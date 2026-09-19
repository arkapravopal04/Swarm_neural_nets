"""
Regression tests for the run-bounding fixes:

  * per-task attempt cap (a task can no longer recycle agents forever),
  * SPAWN rejected in code for a non-decomposer (the prompt said so; only
    the prompt enforced it),
  * requirement inheritance on no keyword overlap (was: inherit everything),
  * an unparseable action falls back to THINK rather than REPORT,
  * a SPAWN batch whose subtasks were copied out of the prompt's own worked
    examples is rejected in code (three rewordings of the prompt failed to
    stop it), including the zero-requirement-overlap case,
  * "ACTION: ACTION: SPAWN" parses as SPAWN, not as the label,
  * a SPAWN whose JSON object was never labelled "PAYLOAD:" is recovered and
    run instead of downgraded to a wasted THINK cycle,
  * a REPORT cut mid-sentence by its generation budget is walked back to the
    last complete sentence before the existing report-trim sees it, and a
    REPORT that never completed a sentence is left alone for the judge's
    empty-answer checks to catch.

Each of these was previously only observable by reading a 4,000-line run
log, which is exactly how they survived as long as they did.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import (
    EXEMPLAR_SUBTASK_DESCRIPTIONS,
    _ACTION_LINE_RE,
    Agent,
)
from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode
from text_utils import drop_incomplete_tail
from agent_node import _ActionPayloadStop


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


def _make_orchestrator():
    return Orchestrator(
        ColonyState(initial_budget=1000, goal_embedding=None),
        TaskGraph(),
        Messenger(),
        memory_store=_NullMemoryStore(),
    )


def _register(orch, agent_id, role="executor", parent_id=None, task="do a thing"):
    orch.colony.register_agent(
        AgentNode(agent_id=agent_id, role=role, status="running",
                  parent_id=parent_id, task=task)
    )


# --------------------------------------------------------------- attempt cap

def test_task_is_abandoned_once_the_attempt_cap_is_exceeded():
    orch = _make_orchestrator()
    orch.MAX_TASK_ATTEMPTS = 3
    orch.task_graph.add_task(TaskNode(task_id="task-1", description="do a thing", status=1))
    orch.last_partial_result["task-1"] = "a half-finished answer"

    for i in range(orch.MAX_TASK_ATTEMPTS + 1):
        _register(orch, f"agent-{i}")
        orch._kill_and_respawn(f"agent-{i}", "task-1", "executor", None)

    # The cap stops the counter climbing past MAX_TASK_ATTEMPTS.
    assert orch.respawn_counts["task-1"] == orch.MAX_TASK_ATTEMPTS
    assert "task-1" in orch.abandoned_tasks

    task = orch.task_graph.tasks["task-1"]
    assert task.status == 3, "an abandoned task must be marked failed"
    assert "ABANDONED" in str(task.result)
    assert "a half-finished answer" in str(task.result), (
        "the best partial result must be salvaged into the abandonment record"
    )


def test_abandoning_a_task_notifies_the_parent_and_releases_dependents():
    orch = _make_orchestrator()
    orch.MAX_TASK_ATTEMPTS = 1
    orch.task_graph.add_task(TaskNode(task_id="task-1", description="do a thing", status=1))
    dependent = TaskNode(task_id="task-2", description="use the thing",
                         dependencies=["task-1"])
    orch.task_graph.add_task(dependent)
    assert dependent.in_degree == 1

    for i in range(orch.MAX_TASK_ATTEMPTS + 1):
        _register(orch, f"agent-{i}", parent_id="parent-1")
        orch._kill_and_respawn(f"agent-{i}", "task-1", "executor", "parent-1")

    assert dependent.in_degree == 0, (
        "a dependent of an abandoned task must be released, not stranded"
    )
    assert dependent.status != 3, "abandonment must not cascade-fail the graph"

    notifications = [e for e in orch.messenger.drain()
                     if e.type == "parent_notification"]
    assert notifications, "the parent must be told its child was abandoned"
    assert "ABANDONED" in str(notifications[-1].payload["result"])


def test_abandoning_the_root_stops_the_run():
    orch = _make_orchestrator()
    orch.MAX_TASK_ATTEMPTS = 1
    orch.root_task_id = "root"
    orch.task_graph.add_task(TaskNode(task_id="root", description="solve it", status=1))

    for i in range(orch.MAX_TASK_ATTEMPTS + 1):
        _register(orch, f"agent-{i}")
        orch._kill_and_respawn(f"agent-{i}", "root", "decomposer", None)

    assert orch.task_graph.tasks["root"].status == 3
    assert orch.tick() is False, (
        "an abandoned root must end the loop so terminate() can synthesize "
        "the partial answer, instead of spinning to energy death"
    )


# ------------------------------------------------------- executor SPAWN ban

def test_executor_spawn_is_rejected_and_rerouted():
    orch = _make_orchestrator()
    orch.task_graph.add_task(TaskNode(task_id="task-1", description="do a thing", status=1))
    _register(orch, "exec-1", role="executor")
    orch.colony.get_agent("exec-1").task_id = "task-1"

    orch.messenger.push_event(
        "spawn_request", "exec-1",
        {"parent_id": "exec-1", "role": "executor", "task_id": "a smuggled subtask"},
    )
    orch._route_events(orch.messenger.drain())

    assert not orch.task_graph.tasks.get("a smuggled subtask")
    assert len(orch.task_graph.tasks) == 1, "no child task may be created"

    failures = [e for e in orch.messenger.drain() if e.type == "failure_request"]
    assert failures, "the rejected SPAWN must be rerouted to the failure path"
    assert str(failures[0].payload["result"]).startswith("TASK TOO LARGE:")


def test_decomposer_spawn_is_still_allowed():
    orch = _make_orchestrator()
    _register(orch, "dec-1", role="decomposer")

    orch.handle_spawn_allowed = orch._reject_spawn_from_non_decomposer(
        type("E", (), {"from_agent": "dec-1", "payload": {}})()
    )
    assert orch.handle_spawn_allowed is True


# ------------------------------------------------------ requirement filter

def test_no_keyword_overlap_inherits_no_requirements():
    orch = _make_orchestrator()
    requirements = [
        "Operating temperature must exceed 1450C",
        "Coating thickness must be under 200 microns",
    ]
    kept = orch._filter_requirements_for_task(
        "Implement the binomial probability mass function", requirements
    )
    assert kept == [], (
        "a subtask sharing no vocabulary with any requirement must inherit "
        "none of them -- inheriting all of them is what made unrelated "
        "constraints unsatisfiable"
    )


def test_keyword_overlap_still_inherits_the_matching_requirement():
    orch = _make_orchestrator()
    requirements = [
        "Operating temperature must exceed 1450C",
        "Coating thickness must be under 200 microns",
    ]
    kept = orch._filter_requirements_for_task(
        "Choose a coating and check its thickness", requirements
    )
    assert kept == ["Coating thickness must be under 200 microns"]


def test_partial_stem_overlap_inherits_at_most_two():
    orch = _make_orchestrator()
    requirements = [
        "Cooling channels must not intersect the trailing edge",
        "Coatings must survive thermal cycling",
        "Documentation must be in English",
    ]
    kept = orch._filter_requirements_for_task("Design the cooled channel layout", requirements)
    assert 0 < len(kept) <= 2


# ------------------------------------------------------------ code detector

def test_code_output_bypasses_tier_two():
    assert Orchestrator._looks_like_code("def f(n):\n    return n * 2\n")
    assert Orchestrator._looks_like_code("```python\nprint(1)\n```")
    assert not Orchestrator._looks_like_code(
        "The operating temperature must exceed 1450 degrees for the coating to bond."
    )
    assert not Orchestrator._looks_like_code("")


# ------------------------------------------------- copied-exemplar subtasks

_TURBINE_REQUIREMENTS = [
    "The turbine blade must survive an operating temperature above 1450C",
    "Coolant flow rate must not exceed 2.1 kg/s",
    "Blade mass must stay under 380 g",
]


def _turbine_orchestrator():
    orch = _make_orchestrator()
    orch.spec = {"goal": "Design a turbine blade", "requirement": list(_TURBINE_REQUIREMENTS)}
    return orch


def test_every_prompt_exemplar_is_recognized_as_copied():
    """The guard reads the same constant the prompt is built from, so no
    exemplar can be added to the prompt without the guard learning it."""
    assert EXEMPLAR_SUBTASK_DESCRIPTIONS, "the prompt exemplars must be discoverable"
    for exemplar in EXEMPLAR_SUBTASK_DESCRIPTIONS:
        assert Orchestrator._matching_exemplar(exemplar) is not None, exemplar


def test_exemplar_match_survives_padding_and_punctuation():
    assert Orchestrator._matching_exemplar(
        "First, independent part A of your task, then proceed"
    ) is not None
    assert Orchestrator._matching_exemplar("  <ONE   FOCUSED PIECE OF YOUR TASK>  ") is not None
    # The model often drops the angle brackets when copying a placeholder.
    assert Orchestrator._matching_exemplar("step that needs the result of part A") is not None


def test_a_real_subtask_extending_a_short_exemplar_is_not_rejected():
    """The exemplars are placeholders, so a genuine project subtask must
    never be mistaken for one."""
    assert Orchestrator._matching_exemplar(
        "Select the base material for the turbine blade given the 1450C floor"
    ) is None
    assert Orchestrator._matching_exemplar(
        "Compute the coolant mass flow rate at the 2.1 kg/s ceiling"
    ) is None


def test_copied_subtask_rejects_the_whole_batch_and_respawns():
    orch = _turbine_orchestrator()
    orch.task_graph.add_task(TaskNode(task_id="task-1", description="design a blade", status=1))
    _register(orch, "dec-1", role="decomposer")
    orch.colony.get_agent("dec-1").task_id = "task-1"

    orch.messenger.push_event(
        "spawn_request", "dec-1",
        {"parent_id": "dec-1", "subtasks": [
            {"role": "executor", "task": "Compute the coolant flow rate margin",
             "dependencies": []},
            {"role": "executor", "task": "<independent part B of YOUR task>",
             "dependencies": []},
        ]},
    )
    orch._route_events(orch.messenger.drain())

    assert len(orch.task_graph.tasks) == 1, (
        "one copied subtask invalidates the plan -- its clean siblings must "
        "not be spawned either"
    )
    failures = [e for e in orch.messenger.drain() if e.type == "failure_request"]
    assert failures, "the rejected batch must reroute the decomposer to a respawn"
    assert "REJECTED" in str(failures[0].payload["result"])


def test_zero_requirement_overlap_alone_rejects_the_batch():
    """Belt to the exemplar guard's braces: a subtask sharing no vocabulary
    with any requirement is rejected even if it matches no known exemplar."""
    orch = _turbine_orchestrator()
    assert Orchestrator._matching_exemplar("Design the newsletter signup widget") is None
    assert orch._has_no_requirement_overlap("Design the newsletter signup widget") is True
    assert orch._has_no_requirement_overlap("Compute the coolant flow rate margin") is False


def test_zero_overlap_guard_is_inert_without_requirements():
    orch = _make_orchestrator()
    orch.spec = {"goal": "do something", "requirement": []}
    assert orch._has_no_requirement_overlap("literally anything at all") is False


def test_a_clean_batch_still_spawns():
    orch = _turbine_orchestrator()
    orch.task_graph.add_task(TaskNode(task_id="task-1", description="design a blade", status=1))
    _register(orch, "dec-1", role="decomposer")
    orch.colony.get_agent("dec-1").task_id = "task-1"

    orch.messenger.push_event(
        "spawn_request", "dec-1",
        {"parent_id": "dec-1", "subtasks": [
            {"role": "executor", "task": "Compute the coolant flow rate margin",
             "dependencies": []},
            {"role": "executor", "task": "Check the blade mass against the 380 g limit",
             "dependencies": []},
        ]},
    )
    orch._route_events(orch.messenger.drain())

    assert len(orch.task_graph.tasks) == 3, "both clean subtasks must be spawned"


# ------------------------------------------------------- ACTION: line parse

def test_repeated_action_label_parses_the_token_not_the_label():
    assert _ACTION_LINE_RE.search("ACTION: ACTION: SPAWN").group(1) == "SPAWN"
    assert _ACTION_LINE_RE.search("ACTION: ACTION: ACTION: TOOL").group(1) == "TOOL"
    assert _ACTION_LINE_RE.search("ACTION : ACTION SPAWN").group(1) == "SPAWN"


def test_ordinary_action_lines_are_unaffected():
    for text, expected in [
        ("ACTION: SPAWN", "SPAWN"),
        ("ACTION:REPORT", "REPORT"),
        ("some preamble\nACTION: DIE\nPAYLOAD: done", "DIE"),
    ]:
        assert _ACTION_LINE_RE.search(text).group(1).upper() == expected


# ------------------------------------------------- unlabelled SPAWN payload

def test_unlabelled_spawn_payload_is_recovered():
    text = ('ACTION: SPAWN\n'
            '{"subtasks": [{"role": "executor", "task": "Size the coolant channels"}]}\n')
    recovered = Agent._recover_unlabelled_payload("SPAWN", text)
    assert isinstance(recovered, dict) and "subtasks" in recovered


def test_unlabelled_tool_payload_is_recovered_past_unrelated_objects():
    text = ('ACTION: TOOL\n'
            'considering {"note": "not a payload"} first\n'
            '{"tool_name": "run_code", "args": {"code_string": "print(1)"}}\n')
    recovered = Agent._recover_unlabelled_payload("TOOL", text)
    assert recovered["tool_name"] == "run_code"


def test_wrong_shaped_object_is_not_passed_off_as_a_payload():
    assert Agent._recover_unlabelled_payload("SPAWN", 'ACTION: SPAWN\n{"note": "hmm"}') is None
    assert Agent._recover_unlabelled_payload("SPAWN", "ACTION: SPAWN\nno object here") is None
    assert Agent._recover_unlabelled_payload("REPORT", '{"task": "x"}') is None


# --------------------------------------------------- REPORT budget + walkback

def test_report_budget_is_wide_enough_to_survive_a_preamble():
    """80 tokens was the length an answer wants to BE, which only works if
    the answer starts at token 1 -- a preamble sentence consumed the whole
    allowance before any answer was generated."""
    assert _ActionPayloadStop.REPORT_MAX_NEW_TOKENS >= 200


def test_mid_sentence_stump_is_walked_back():
    raw = ("Fifteen minutes per round. Allow five minutes for setup and ten "
           "for the actual acti")
    assert drop_incomplete_tail(raw) == "Fifteen minutes per round."


def test_complete_text_is_left_alone():
    for text in ["Fifteen minutes per round.", "Why not?", "Stop!"]:
        assert drop_incomplete_tail(text) == text


def test_closing_quote_travels_with_its_terminator():
    assert drop_incomplete_tail('He said "stop." Then we') == 'He said "stop."'


def test_text_with_no_terminator_is_never_emptied():
    """The judge's empty-answer checks are the only reason this failure mode
    is visible; the walk-back must not swallow the evidence."""
    preamble = ("The icebreaker activity prompt and timing guide are "
                "structured as follows:")
    assert drop_incomplete_tail(preamble) == preamble
    assert drop_incomplete_tail("- glass\n- paper\n- plas") == "- glass\n- paper\n- plas"
    assert drop_incomplete_tail("") == ""
    assert drop_incomplete_tail(None) is None


def test_walkback_feeds_the_existing_report_trim():
    """The two are ordered, not alternatives: the walk-back decides where the
    text ends, report-trim decides how much of it the judge reads."""
    orch = _make_orchestrator()
    orch.task_graph.add_task(TaskNode(task_id="task-1", description="time the icebreaker", status=1))
    _register(orch, "exec-1", role="executor")
    orch.colony.get_agent("exec-1").task_id = "task-1"

    raw = ("Fifteen minutes per round. Allow five minutes for setup and ten "
           "for discussion. Then rotate the groups. Finally collect the "
           "cards and rev")
    orch.messenger.push_event(
        "completion_request", "exec-1",
        {"task_id": "task-1", "result": raw},
    )
    orch._route_events(orch.messenger.drain())

    stored = orch.last_partial_result["task-1"]
    assert stored.endswith("."), f"a stump reached the judge: {stored!r}"
    assert "rev" not in stored.split(".")[-1]


# ------------------------------------------------ last action block wins

from agent_node import _last_action_match, _payload_match_for


def test_last_action_block_wins_over_an_abandoned_draft():
    text = ('ACTION: REPORT\nPAYLOAD: draft answer\n'
            'Actually this needs splitting.\n'
            'ACTION: SPAWN\nPAYLOAD: {"role": "executor", "task": "x"}')
    match = _last_action_match(text)
    assert match.group(1).upper() == "SPAWN"
    assert _payload_match_for(text, match).group(1).startswith('{"role"')


def test_prose_action_label_in_a_payload_does_not_displace_the_real_one():
    text = "ACTION: REPORT\nPAYLOAD: Done. Recommended action: reduce load."
    assert _last_action_match(text).group(1).upper() == "REPORT"


class _TextTokenizer:
    """decode() returns a fixed generation; ids only need a length."""
    def __init__(self, text):
        self.text = text

    def decode(self, ids, skip_special_tokens=True):
        return self.text


def _stop_check(text):
    """(action the stop check parses, whether it ends generation)."""
    stop = _ActionPayloadStop(_TextTokenizer(text), prompt_len=0,
                              extract_balanced_object=Agent._extract_first_balanced_object,
                              min_new_tokens=0)
    return stop._parsed_action(text), stop([[0] * 50], None)


def test_action_word_inside_a_payload_line_does_not_hijack_the_action():
    cases = [
        ("ACTION: REPORT\nPAYLOAD: The recommended action: DIE if pressure > 5 bar.",
         "REPORT", "The recommended action: DIE"),
        ("ACTION: THINK\nPAYLOAD: Next action: TOOL to compute density.",
         "THINK", "Next action: TOOL"),
        # colon + action-like word, twice, mid-payload
        ("ACTION: REPORT\nPAYLOAD: Step 1 -- action: SPAWN helpers; step 2 -- ACTION: DIE.",
         "REPORT", "Step 1"),
    ]
    for text, expected, payload_start in cases:
        match = _last_action_match(text)
        assert match.group(1).upper() == expected, text
        assert _payload_match_for(text, match).group(1).startswith(payload_start), text
        assert _stop_check(text) == (expected, False), text
        # A finished plain-text payload still ends generation on the real block.
        assert _stop_check(text + "\n\n") == (expected, True), text


def test_markdown_decorated_label_still_parses():
    assert _last_action_match("**ACTION: SPAWN**\nPAYLOAD: {}").group(1).upper() == "SPAWN"
    assert _last_action_match("reasoning\n- ACTION: REPORT\nPAYLOAD: x").group(1).upper() == "REPORT"


def test_bare_keyword_fallback_ignores_prose_lines():
    assert _last_action_match("Spawn\n{}").group(1).upper() == "SPAWN"
    text = "SPAWN\n{}\nThink about the dependencies first."
    assert _last_action_match(text).group(1).upper() == "SPAWN"
    assert _last_action_match("some reasoning\nREPORT\nthe answer").group(1) == "REPORT"


# ------------------------------------------------ role-scoped action menu

def _prompt_for(role, generation=1):
    from colony_state import AgentNode
    node = AgentNode(agent_id="a-1", role=role, status="active", parent_id="root",
                     task="do the thing", task_id="t-1", generation=generation)
    return Agent(tokeniser=None, model=None, message=None, node=node)


def test_only_decomposers_are_offered_spawn():
    for role in ("executor", "verifier"):
        agent = _prompt_for(role)
        prompt = agent._build_prompt(available_tools=["run_code"], requirements=[])
        seed = agent._build_thinking_seed(requirements=[], available_tools=["run_code"])
        assert "- SPAWN" not in prompt and "If SPAWN" not in prompt, role
        assert "ACTION: SPAWN" not in prompt, role
        assert "SPAWN" not in seed.split("Real actions")[1], role

    agent = _prompt_for("decomposer")
    prompt = agent._build_prompt(available_tools=["run_code"], requirements=[])
    assert "- SPAWN" in prompt and "ACTION: SPAWN" in prompt
    assert "- TOOL" not in prompt
    assert "Give the answer itself first" not in prompt


def test_task_too_large_escape_hatch_is_executor_only():
    for role, expected in (("executor", True), ("verifier", False), ("decomposer", False)):
        agent = _prompt_for(role)
        for tools in (["run_code"], []):
            prompt = agent._build_prompt(available_tools=tools, requirements=[])
            seed = agent._build_thinking_seed(requirements=[], available_tools=tools)
            assert ("TASK TOO LARGE" in prompt) is expected, (role, tools)
            assert ("TASK TOO LARGE" in seed) is expected, (role, tools)


# ------------------------------------ TOOL access: prompt and enforcement agree

def _tool_orchestrator(agent, monkeypatch):
    import orchestrator as orchestrator_module
    executed = []
    monkeypatch.setattr(
        orchestrator_module.ToolRegistry, "execute",
        staticmethod(lambda name, args, **kw: executed.append(name) or {"status": "ok", "output": "1"}),
    )
    orch = Orchestrator.__new__(Orchestrator)
    orch.spec = {"domain": "General Discourse"}
    orch.judge = None
    orch.live_agents = {agent.agent_id: agent}
    orch.messenger = Messenger()
    return orch, executed


def _tool_request_from(agent):
    from event_queue import Event
    event = Event(type="tool_request", from_agent=agent.agent_id)
    event.payload.update({"agent_id": agent.agent_id, "tool_name": "run_code",
                          "args": {"code_string": "print(1)"}})
    return event


def test_tool_prompt_and_orchestrator_enforcement_agree(monkeypatch):
    from agent_node import role_may_use_tools
    for role in ("decomposer", "executor", "verifier"):
        agent = _prompt_for(role)
        prompt = agent._build_prompt(available_tools=["run_code"], requirements=[])
        offered = "- TOOL" in prompt
        assert offered is role_may_use_tools(role), role

        orch, executed = _tool_orchestrator(agent, monkeypatch)
        orch.handle_tool_request(_tool_request_from(agent))
        assert bool(executed) is offered, role

    assert not role_may_use_tools("decomposer")


def test_refused_decomposer_tool_call_does_not_trip_the_circuit_breaker(monkeypatch):
    agent = _prompt_for("decomposer")
    orch, executed = _tool_orchestrator(agent, monkeypatch)
    for _ in range(agent.MAX_CONSECUTIVE_TOOL_FAILURES + 1):
        orch.handle_tool_request(_tool_request_from(agent))
    assert executed == []
    assert not agent.tool_circuit_open
    assert "REJECTED" in agent.fail_reason and "SPAWN" in agent.fail_reason
