"""
Regressions from run 3 (book-club voting goal, TERMINATED [SUCCESS] with 8
subtasks abandoned):

  1. TaskNode.parent_task_id / TaskGraph.direct_children: a view of a task's
     own subtasks that reparenting (ColonyState.unregister_agent) cannot
     pollute. The agent-level `children` list is left as it was.
  2. A decomposer whose SPAWN is refused for energy is failed, not released
     to REPORT, unless at least one DIRECT subtask completed. The root's
     structural gate counts direct subtasks. The end-of-run prompt states
     how many subtasks completed, and a decomposer with none gets DIE only.
  3. The YOUR SUBTASKS block: subtask status in the decide() prompt, built
     from the graph, outside the 500-character thought window.
  4. The [PARTIAL RESULT] banner on a SUCCESS run that abandoned subtasks.
  5. The exit guard's artifact cut (code fence, JSON tail, separator run)
     and the unfinished-sentence walk-back after a cut.
  6. The prompt's worked REPORT: rejected before the judge when it is the
     whole REPORT, cut out wherever it is echoed alongside real content.
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent, EXEMPLAR_REPORT_TEXTS, _EX_WORKED_REPORT
from colony_state import AgentNode, ColonyState
from event_queue import Event, Messenger
from orchestrator import Orchestrator
from synthesizer import Synthesizer
from task_graph import TaskGraph, TaskNode
from text_utils import is_exemplar_echo, trim_artifact_tail


WORKED = _EX_WORKED_REPORT
DELIMITER = "---END-- -- --- ----"
ANSWER = "Hold one ranked-choice vote per month and drop the lowest title each round."
BATCH = {"subtasks": [{"label": "a", "role": "executor", "task": "compute x",
                       "dependencies": []}]}

# Run 3's shipped final answer, from its last sentence on.
RUN3_ANSWER = (
    "Disputes escalate through a structured conflict resolution step. It was "
    "never explicitly designed to handle unresolvable deadlocks, though a "
    "fallback is implicitly built into the conflict escalation framework. "
    "--- --- --- ----\n```json\n{\"status\": \"complete\", \"output\": "
    "\"Disputes escalate through a structured conflict resolution step."
)
RUN3_KEPT = RUN3_ANSWER[:RUN3_ANSWER.index(" ---")]


class _Store:
    def __init__(self):
        self.writes = []

    def write(self, record_type, text, metadata):
        self.writes.append((record_type, text, metadata))

    def get_success_cache(self, description, task_id=None):
        return None

    def save_ghosts(self):
        pass


class _Judge:
    def __init__(self, verdict="promote"):
        self.verdict = verdict
        self.seen = []

    def decide(self, agent_node, **kwargs):
        self.seen.append(kwargs.get("output"))
        return {"verdict": self.verdict, "reason": "ok"}


def _orch(budget=1000, judge=None):
    return Orchestrator(ColonyState(initial_budget=budget, goal_embedding=None),
                        TaskGraph(), Messenger(), judge=judge, memory_store=_Store())


def _run3_tree(orch, live_root=False):
    """Root r on "root" whose only subtask t-mid was abandoned; t-mid's own
    subtask t-leaf completed. Retiring t-mid's agent m hands l to r -- the
    shape that let run 3's root pass its structural gate."""
    orch.root_task_id = "root"
    graph = orch.task_graph
    graph.add_task(TaskNode(task_id="root", description="choose books", agent_id="r", status=1))
    root = AgentNode(agent_id="r", role="decomposer", status="running", parent_id=None,
                     task="choose books", task_id="root", generation=0, has_spawned=True)
    orch.colony.register_agent(root)
    graph.add_task(TaskNode(task_id="t-mid", description="plan the vote", agent_id="m",
                            status=3, parent_task_id="root"))
    orch.abandoned_tasks.add("t-mid")
    orch.colony.register_agent(AgentNode(agent_id="m", role="decomposer", status="running",
                                         parent_id="r", task="plan the vote",
                                         task_id="t-mid", generation=1))
    graph.add_task(TaskNode(task_id="t-leaf", description="count the ballots", agent_id="l",
                            status=2, parent_task_id="t-mid"))
    graph.tasks["t-leaf"].result = "Count ballots by hand."
    orch.colony.register_agent(AgentNode(agent_id="l", role="executor", status="completed",
                                         parent_id="m", task="count the ballots",
                                         task_id="t-leaf", generation=2))
    orch.colony.unregister_agent("m")
    if live_root:
        agent = Agent(tokeniser=None, model=None, message=orch.messenger, node=root)
        orch.live_agents["r"] = agent
        return agent
    return root


def _spawning_decomposer(budget):
    orch = _orch(budget=budget)
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan", agent_id="p",
                                      status=1))
    node = AgentNode(agent_id="p", role="decomposer", status="running", parent_id=None,
                     task="plan", task_id="t-parent", generation=1)
    orch.colony.register_agent(node)
    agent = Agent(tokeniser=None, model=None, message=orch.messenger, node=node)
    orch.live_agents["p"] = agent
    return orch, agent


def _spawn(orch, agent, payload):
    agent.request_spawn(payload)
    orch._route_events(orch.messenger.drain())


# ------------------------------------------------------ 1. direct children

def test_direct_children_ignore_reparented_grandchildren():
    orch = _orch()
    _run3_tree(orch)
    # Reparenting itself is unchanged...
    assert "l" in orch.colony.get_agent("r").children
    # ...and the task graph's view is not affected by it.
    assert [t.task_id for t in orch.task_graph.direct_children("root")] == ["t-mid"]
    assert [t.task_id for t in orch.task_graph.direct_children("t-mid")] == ["t-leaf"]
    assert orch.task_graph.direct_children(None) == []


def test_a_spawned_subtask_records_its_parent_task_and_label():
    orch, agent = _spawning_decomposer(budget=1000)
    _spawn(orch, agent, BATCH)
    (child,) = orch.task_graph.direct_children("t-parent")
    assert child.parent_task_id == "t-parent" and child.label == "a"
    assert child.description == "compute x"


# ------------------------------------------------- 2. SPAWN-refusal REPORT

def test_out_of_energy_with_only_abandoned_subtasks_is_failed_not_released():
    orch, agent = _spawning_decomposer(budget=0)
    agent.node.has_spawned = True
    orch.task_graph.add_task(TaskNode(task_id="t-gone", description="compute w", status=3,
                                      parent_task_id="t-parent"))
    orch.abandoned_tasks.add("t-gone")

    _spawn(orch, agent, BATCH)

    failures = [e for e in orch.messenger.drain() if e.type == "failure_request"]
    assert [e.from_agent for e in failures] == ["p"]
    assert agent.awaiting == "children" and not agent.spawn_closed
    assert "none of its 1 earlier subtask(s) completed" in agent.fail_reason
    assert orch.colony.verdict_counts["spawn_refused_no_completed_child"] == 1


def test_an_adopted_grandchild_does_not_count_as_a_completed_subtask():
    orch = _orch(budget=0)
    root = _run3_tree(orch, live_root=True)
    _spawn(orch, root, BATCH)
    failures = [e for e in orch.messenger.drain() if e.type == "failure_request"]
    assert [e.from_agent for e in failures] == ["r"]


def test_open_means_in_flight_and_the_tally_splits_closed():
    orch = _orch()
    _run3_tree(orch)
    orch.task_graph.add_task(TaskNode(task_id="t-run", description="assign roles", status=1,
                                      parent_task_id="root"))
    assert orch._open_child_task_ids("r") == ["t-run"]
    assert orch._direct_child_tally("root") == {"open": 1, "completed": 0, "failed": 1}

    orch.task_graph.tasks["t-run"].status = 3
    assert orch._open_child_task_ids("r") == []
    assert orch._direct_child_tally("root") == {"open": 0, "completed": 0, "failed": 2}


def _root_report(orch, text):
    event = Event(type="completion_request", from_agent="r")
    event.payload.update({"task_id": "root", "result": text})
    orch.handle_completion(event)


def test_root_gate_counts_direct_subtasks_not_adopted_ones(capsys):
    orch = _orch()
    _run3_tree(orch)
    _root_report(orch, "Voting selected a top three pool. The remaining title is confirmed.")
    assert orch.task_graph.tasks["root"].status != 2
    assert "REJECT (structural)" in capsys.readouterr().out


def test_root_gate_passes_with_a_completed_direct_subtask():
    orch = _orch()
    _run3_tree(orch)
    orch.task_graph.tasks["t-mid"].status = 2
    _root_report(orch, ANSWER)
    assert orch.task_graph.tasks["root"].status == 2


# ------------------------------- A. the root is never served from the cache

class _HitStore(_Store):
    """Every lookup returns an accepted entry from another task."""

    def get_success_cache(self, description, task_id=None):
        return SimpleNamespace(hit={"result": "a donor task's answer",
                                    "task_id": "t-donor", "outcome": "accepted"},
                               hit_score=0.95, negative_count=0,
                               negative_outcomes={}, cross_task_below_threshold=0)


def test_a_failed_root_is_never_completed_from_the_cache(capsys):
    orch = _orch()
    orch.memory_store = _HitStore()
    _run3_tree(orch)

    orch._kill_and_respawn("r", "root", "decomposer", None)

    assert orch.task_graph.tasks["root"].status != 2
    assert orch.colony.results.get("root") != "a donor task's answer"
    assert orch.colony.verdict_counts["root_cache_lookup_skipped"] == 1
    assert orch.colony.verdict_counts["root_cache_hit_withheld"] == 1
    assert "cache_hit_served" not in orch.colony.verdict_counts
    assert "would have completed it" in capsys.readouterr().out


def test_a_non_root_task_is_still_served_from_the_cache():
    orch = _orch()
    orch.memory_store = _HitStore()
    _run3_tree(orch)
    orch.task_graph.add_task(TaskNode(task_id="t-sub", description="count votes", agent_id="s",
                                      status=1, parent_task_id="root"))
    orch.colony.register_agent(AgentNode(agent_id="s", role="executor", status="running",
                                         parent_id="r", task="count votes", task_id="t-sub"))

    orch._kill_and_respawn("s", "t-sub", "executor", "r")

    assert orch.task_graph.tasks["t-sub"].status == 2
    assert orch.task_graph.tasks["t-sub"].result == "a donor task's answer"
    assert orch.colony.verdict_counts["cache_hit_served"] == 1
    assert "root_cache_lookup_skipped" not in orch.colony.verdict_counts


# ------------------------ B. the gate covers every decomposer, on every path

def _decomposer_colony(judge=None):
    orch = _orch(judge=judge or _Judge())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan", agent_id="p",
                                      status=1))
    orch.colony.register_agent(AgentNode(agent_id="p", role="decomposer", status="running",
                                         parent_id=None, task="plan", task_id="t-parent"))
    orch.task_graph.add_task(TaskNode(task_id="t-mid2", description="settle disputes",
                                      agent_id="d1", status=1, parent_task_id="t-parent"))
    orch.colony.register_agent(AgentNode(agent_id="d1", role="decomposer", status="running",
                                         parent_id="p", task="settle disputes",
                                         task_id="t-mid2", generation=1))
    return orch


def _mid_report(orch, text):
    event = Event(type="completion_request", from_agent="d1")
    event.payload.update({"task_id": "t-mid2", "result": text})
    orch.handle_completion(event)


def test_a_non_root_decomposer_report_without_a_completed_subtask_is_rejected(capsys):
    orch = _decomposer_colony()
    orch.task_graph.add_task(TaskNode(task_id="t-dead", description="draft rules", status=3,
                                      parent_task_id="t-mid2"))
    orch.abandoned_tasks.add("t-dead")

    _mid_report(orch, ANSWER)

    assert orch.judge.seen == []
    assert orch.task_graph.tasks["t-mid2"].status != 2
    # Not salvaged either: the parent must not get this text if the task is
    # abandoned later.
    assert "t-mid2" not in orch.last_partial_result
    assert orch.colony.verdict_counts["decomposer_report_no_completed_child"] == 1
    out = capsys.readouterr().out
    assert "REJECT (structural)" in out and "root decomposer" not in out


def test_a_decomposer_report_with_a_completed_subtask_still_reaches_the_judge():
    orch = _decomposer_colony()
    orch.task_graph.add_task(TaskNode(task_id="t-done2", description="draft rules", status=2,
                                      parent_task_id="t-mid2"))

    _mid_report(orch, ANSWER)

    assert orch.judge.seen == [ANSWER]
    assert orch.task_graph.tasks["t-mid2"].status == 2
    assert "decomposer_report_no_completed_child" not in orch.colony.verdict_counts


def test_an_executor_report_is_untouched_by_the_decomposer_gate():
    orch = _executor_colony()
    _report(orch, ANSWER)
    assert orch.judge.seen == [ANSWER]
    assert orch.task_graph.tasks["t-1"].status == 2


class _Inputs(dict):
    def to(self, device):
        return self


class _Tokeniser:
    eos_token_id = 0

    def __init__(self, reply):
        self.reply = reply

    def __call__(self, prompt, return_tensors=None):
        return _Inputs(input_ids=torch.zeros((1, 3), dtype=torch.long))

    def decode(self, tokens, skip_special_tokens=True):
        return self.reply


class _Model:
    device = "cpu"

    def generate(self, input_ids=None, **kwargs):
        return torch.zeros((1, 5), dtype=torch.long)


MIXED = [
    {"task": "Define the vote count rules", "status": "completed", "label": "a",
     "excerpt": "Plurality first, then a runoff."},
    {"task": "Define role assignment rules", "status": "abandoned", "label": "b",
     "excerpt": None},
    {"task": "Create a conflict protocol", "status": "not started", "label": None,
     "excerpt": None},
]
NONE_COMPLETED = [dict(row, status="abandoned", excerpt=None) for row in MIXED]


def _decomposer(rows, capped=True, spawn_closed=False, reply=None):
    node = AgentNode(agent_id="d", role="decomposer", status="running", parent_id="root",
                     task="plan it", task_id="t-d", generation=1, has_spawned=True)
    agent = Agent(tokeniser=None, model=None, message=Messenger(), node=node)
    agent.child_status_fn = lambda: rows
    if capped:
        agent.non_terminal_cycles = agent.MAX_NON_TERMINAL_CYCLES
    agent.spawn_closed = spawn_closed
    if reply is not None:
        agent.tokeniser, agent.model = _Tokeniser(reply), _Model()
    return agent


def test_final_prompt_states_how_many_subtasks_completed():
    agent = _decomposer(MIXED, capped=False, spawn_closed=True)
    prompt = agent._build_prompt(available_tools=[], requirements=[])
    assert "your subtasks are finished" not in prompt
    assert "1 of your 3 subtask(s) completed; the rest: 1 abandoned, 1 not started" in prompt
    assert agent.final_actions == ("REPORT", "DIE")
    assert "- REPORT" in prompt


def test_a_decomposer_with_no_completed_subtask_is_offered_die_only():
    agent = _decomposer(NONE_COMPLETED)
    assert agent.final_actions == ("DIE",)
    prompt = agent._build_prompt(available_tools=[], requirements=[])
    assert "none of your 3 subtask(s) completed (3 abandoned)" in prompt
    for closed in ("- REPORT", "If REPORT", "ACTION: REPORT", "- SPAWN", "- THINK"):
        assert closed not in prompt, closed
    assert "ACTION: DIE" in prompt


def test_a_report_from_a_decomposer_with_no_completed_subtask_is_turned_into_die():
    agent = _decomposer(NONE_COMPLETED, reply=(
        "ACTION: REPORT\nPAYLOAD: Voting mechanics selected a top 3 candidate pool."))
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "DIE" and "none of its subtasks completed" in payload


def test_without_a_colony_view_the_prompt_claims_nothing():
    agent = _decomposer(None)
    agent.child_status_fn = None
    prompt = agent._build_prompt(available_tools=[], requirements=[])
    assert "your subtasks are finished" not in prompt
    assert "[YOUR SUBTASKS" not in prompt
    assert agent.final_actions == ("REPORT", "DIE")


# ------------------------------------------------ 3. YOUR SUBTASKS block

def test_subtask_block_survives_a_full_thought_window():
    agent = _decomposer(MIXED, capped=False)
    agent.thought_process = ("\n[CHILD RESULT - agent_x]: [ABANDONED after 1 attempt(s)]\n"
                             + "more planning notes " * 100)
    prompt = agent._build_prompt(available_tools=[], requirements=[])
    assert "[CHILD RESULT" not in prompt  # pushed out of the 500-char window
    assert '- a "Define the vote count rules": completed -- Plurality first, then a runoff.' in prompt
    assert '- b "Define role assignment rules": abandoned' in prompt
    assert '- "Create a conflict protocol": not started' in prompt
    assert "[END OF YOUR SUBTASKS]" in prompt


def test_orchestrator_rows_label_status_and_excerpt():
    orch = _orch()
    _run3_tree(orch)
    orch.task_graph.add_task(TaskNode(task_id="t-ok", description="assign roles", status=2,
                                      parent_task_id="root", label="roles"))
    orch.task_graph.tasks["t-ok"].result = "word " * 60
    orch.task_graph.add_task(TaskNode(task_id="t-new", description="protocol", status=3,
                                      parent_task_id="root"))
    orch.unstarted_tasks.add("t-new")
    rows = {row["task"]: row for row in orch._direct_child_statuses("root")}
    assert rows["plan the vote"]["status"] == "abandoned"
    assert rows["protocol"]["status"] == "not started"
    assert rows["assign roles"]["label"] == "roles"
    assert rows["assign roles"]["excerpt"].endswith("...")
    assert len(rows["assign roles"]["excerpt"]) <= Orchestrator.CHILD_EXCERPT_CHARS + 3
    assert "count the ballots" not in rows  # grandchild


# -------------------------------------------- 4. partial banner on SUCCESS

def test_a_success_with_abandoned_subtasks_carries_the_partial_banner(capsys):
    orch = _orch()
    orch.root_task_id = "root_task_0"
    orch.task_graph.add_task(TaskNode(task_id="root_task_0", description="plan", status=2))
    orch.colony.results["final_spec"] = ANSWER
    orch._final_spec_synthesized = True
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="a", status=2,
                                      parent_task_id="root_task_0"))
    orch.task_graph.tasks["t-1"].result = "Monthly."
    orch.task_graph.add_task(TaskNode(task_id="t-2", description="b", status=3,
                                      parent_task_id="root_task_0"))
    orch.abandoned_tasks.add("t-2")
    orch.abandon_reasons["t-2"] = "no energy to respawn"

    out = orch.terminate()

    assert out == ("[PARTIAL RESULT -- one or more subtasks were abandoned (no energy to "
                   "respawn) before every subtask finished. 1 subtask(s) completed and are "
                   "synthesized below; anything not mentioned was not reached.]\n\n" + ANSWER)
    printed = capsys.readouterr().out
    assert "System Status: TERMINATED [SUCCESS]" in printed
    assert "PARTIAL: 1 subtask(s) were abandoned" in printed
    assert orch.colony.verdict_counts["partial_banner_on_success"] == 1


def test_a_clean_success_has_no_banner():
    orch = _orch()
    orch.root_task_id = "root_task_0"
    orch.task_graph.add_task(TaskNode(task_id="root_task_0", description="plan", status=2))
    orch.colony.results["final_spec"] = ANSWER
    assert orch.terminate() == ANSWER


# ----------------------------------------------------- 5. artifact guard

def _book_club(orch):
    orch.spec = {"raw_text": "How should our book club choose books by voting?"}
    return orch


def test_exit_guard_cuts_run3_fence_json_and_separator_tail():
    orch = _book_club(_orch())
    assert orch._guard_final_answer(RUN3_ANSWER) == RUN3_KEPT
    counts = orch.colony.verdict_counts
    assert counts["final_answer_artifact_fence"] == 1
    assert counts["final_answer_artifact_separator"] == 1
    assert counts["final_answer_cut"] == 1


def test_exit_guard_cuts_a_bare_json_tail():
    orch = _book_club(_orch())
    out = orch._guard_final_answer(ANSWER + ' {"status": "complete", "output": "Hold one"}')
    assert out == ANSWER
    assert orch.colony.verdict_counts["final_answer_artifact_json"] == 1


def test_exit_guard_walks_back_a_stump_the_cut_exposed():
    orch = _book_club(_orch())
    out = orch._guard_final_answer(ANSWER + " Then book the rooms for\n```json\n{\"a\": 1}")
    assert out == ANSWER
    assert orch.colony.verdict_counts["final_answer_incomplete_tail_after_cut"] == 1


def test_exit_guard_keeps_code_on_a_software_request():
    orch = _orch()
    orch.spec = {"raw_text": "Write a Python script that parses a CSV file and prints the totals."}
    answer = "Use this script.\n```python\nprint(1)\n```"
    assert orch._guard_final_answer(answer) == answer


def test_the_goal_delimiter_is_a_separator_run():
    goal = ("Choose books by voting, assign roles before meeting starts, remove disliked "
            "titles via majority vote during discussion." + DELIMITER)
    kept, reasons = trim_artifact_tail(goal)
    assert kept.endswith("during discussion.") and reasons == ["separator run"]


def test_prose_endings_are_not_separators():
    for text in ("Keep it short —", "We will see...", "| a | b |\n|---|---|\n| 1 | 2 |",
                 "Cost is 10 - 20"):
        assert trim_artifact_tail(text) == (text, []), text


def test_synthesis_walks_back_a_stump_its_own_cut_exposed():
    synth = Synthesizer(llm_call_fn=lambda prompt: (
        ANSWER + " Another step follows\nACTION REQUESTED: RUN OR ABORT?"))
    out = synth.format_output([{"task_id": "t1", "description": "d", "result": "r"}], "p")
    assert out == ANSWER
    assert synth.trim_counts == {"cut": 1, "incomplete_tail_after_cut": 1}


# -------------------------------------------------- 6. worked-example echo

def test_exemplar_echo_detection():
    assert EXEMPLAR_REPORT_TEXTS == (WORKED,)
    assert is_exemplar_echo(WORKED + DELIMITER, EXEMPLAR_REPORT_TEXTS)
    assert is_exemplar_echo("Final answer: " + WORKED.lower(), EXEMPLAR_REPORT_TEXTS)
    assert not is_exemplar_echo(ANSWER + " " + WORKED, EXEMPLAR_REPORT_TEXTS)
    assert not is_exemplar_echo(ANSWER, EXEMPLAR_REPORT_TEXTS)


def _executor_colony(role="executor", judge=None):
    orch = _orch(judge=judge or _Judge())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan", agent_id="p",
                                      status=1))
    orch.colony.register_agent(AgentNode(agent_id="p", role="decomposer", status="running",
                                         parent_id=None, task="plan", task_id="t-parent"))
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="settle disputes",
                                      agent_id="a1", status=1, parent_task_id="t-parent"))
    orch.colony.register_agent(AgentNode(agent_id="a1", role=role, status="running",
                                         parent_id="p", task="settle disputes",
                                         task_id="t-1"))
    return orch


def _report(orch, text):
    event = Event(type="completion_request", from_agent="a1")
    event.payload.update({"task_id": "t-1", "result": text})
    orch.handle_completion(event)


def test_a_report_that_is_the_worked_example_is_rejected_before_the_judge():
    for role in ("executor", "decomposer"):
        orch = _executor_colony(role)
        _report(orch, WORKED + DELIMITER)
        assert orch.judge.seen == [], role
        assert orch.task_graph.tasks["t-1"].status != 2, role
        assert "t-1" not in orch.last_partial_result, role
        assert orch.colony.verdict_counts["report_exemplar_echo_rejected"] == 1, role


def test_an_echo_appended_to_a_real_answer_is_cut_at_promotion():
    orch = _executor_colony()
    _report(orch, ANSWER + " " + WORKED)
    assert orch.task_graph.tasks["t-1"].status == 2
    assert orch.task_graph.tasks["t-1"].result == ANSWER


def test_synthesis_cuts_an_echoed_worked_example():
    synth = Synthesizer(llm_call_fn=lambda prompt: ANSWER + " " + WORKED)
    out = synth.format_output([{"task_id": "t1", "description": "d", "result": "r"}], "p")
    assert out == ANSWER
