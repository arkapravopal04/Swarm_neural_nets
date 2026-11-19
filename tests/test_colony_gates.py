"""Colony gate harness (TRAINING_SPEC §10 G4-G6): the executor-root option
G4 runs on, and the P1 root-split count the bench summary records."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_runner  # noqa: E402
from colony_state import ColonyState  # noqa: E402
from event_queue import Messenger  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from task_graph import TaskGraph, TaskNode  # noqa: E402


def _orch(**kw):
    return Orchestrator(ColonyState(initial_budget=500, goal_embedding=None), TaskGraph(), Messenger(), **kw)


@pytest.mark.parametrize("role", ["decomposer", "executor"])
def test_root_role_sets_the_root_task_and_its_agent(role):
    orch = _orch(root_role=role)
    orch.initialize_colony("A 230 V heater draws 10 A. What is its power in watts?")
    root = orch.task_graph.tasks[orch.root_task_id]
    assert root.required_role == role
    agent = orch.colony.get_agent(root.agent_id)
    assert agent is not None and agent.role == role and agent.generation == 0


def test_default_root_is_still_a_decomposer():
    assert _orch().root_role == "decomposer"


def test_unknown_root_role_is_refused():
    with pytest.raises(ValueError):
        _orch(root_role="verifier")


def test_summary_counts_the_roots_children():
    graph = TaskGraph()
    graph.add_task(TaskNode(task_id="root", description="q", required_role="decomposer"))
    graph.add_task(TaskNode(task_id="a", description="part a", required_role="executor", parent_task_id="root"))
    graph.add_task(TaskNode(task_id="b", description="part b", required_role="decomposer", parent_task_id="root"))
    graph.add_task(TaskNode(task_id="c", description="part of b", required_role="executor", parent_task_id="b"))
    orch = SimpleNamespace(task_graph=graph, root_task_id="root", colony=None, spec={})
    row = bench_runner._summary({"id": "T3-01", "tier": 3}, orch, "ok", 1.0, None)
    assert row["root_children"] == ["executor", "decomposer"]


def test_summary_survives_a_colony_that_never_built():
    row = bench_runner._summary({"id": "T3-01"}, None, "error", 0.0, None, error="boom")
    assert row["root_children"] == []


def test_school_check_finds_all_four_parts_in_a_right_answer():
    import colony_gates
    right = ("Take the bus: 40,000 rupees against 900 x 50 = 45,000 for the train. Meals cost about "
             "12,500 rupees a day, under the 15,000 limit. Make 5 groups of 9 students, one teacher each. "
             "If a student falls ill, a teacher takes them to a doctor and calls the parents.")
    assert all(f(right) for f in colony_gates.SCHOOL_CHECKS.values())


def test_school_check_flags_the_wrong_transport_and_groups():
    import colony_gates
    wrong = ("Take the train: 900 x 45 = 40,500 rupees. Meals cost 16,000 rupees a day. "
             "Make 9 groups of 5. If a student is ill, call a doctor.")
    flags = {k: bool(f(wrong)) for k, f in colony_gates.SCHOOL_CHECKS.items()}
    assert not flags["transport: the bus"] and not flags["groups: 5 groups of 9"]
    assert not flags["meals: a daily figure under 15,000"]


class _NoChildrenSynth:
    """What Synthesizer.run returns for a root with no children."""
    def run(self, *a, **k):
        return "No subtask results were produced -- there is nothing to synthesize."


def _executor_root():
    # Stand-in model and tokeniser: the orchestrator only builds live agents when it has both.
    orch = _orch(root_role="executor", synthesizer=_NoChildrenSynth(), model=object(), tokeniser=object())
    orch.initialize_colony("A 230 V heater draws 10 A. What is its power in watts?")
    agent_id = orch.task_graph.tasks[orch.root_task_id].agent_id
    return orch, orch.live_agents[agent_id]


def test_a_promoted_executor_root_returns_its_own_report():
    orch, agent = _executor_root()
    agent.report("The power is 230 x 10 = 2300 W.")
    orch._route_events(orch.messenger.drain())
    assert orch.colony.results.get("final_spec") == "The power is 230 x 10 = 2300 W."


def test_an_executor_root_is_respawned_not_converted_on_task_too_large():
    orch, agent = _executor_root()
    agent.die("TASK TOO LARGE: it mixes 2 separate tasks (power, cost).")
    orch._route_events(orch.messenger.drain())
    root = orch.task_graph.tasks[orch.root_task_id]
    assert root.required_role == "executor"
    assert orch.colony.verdict_counts.get("executor_root_conversion_refused") == 1
    assert not [t for t in orch.task_graph.tasks.values() if t.parent_task_id == orch.root_task_id]


def test_an_executor_root_is_not_offered_task_too_large():
    orch, agent = _executor_root()
    assert "TASK TOO LARGE" not in agent._build_prompt(["decomposer", "executor"], ["run_code"], [])


def test_built_on_results_are_the_tasks_dependencies_and_children():
    orch = _orch()
    g = orch.task_graph
    g.add_task(TaskNode(task_id="root", description="plan", required_role="decomposer"))
    g.add_task(TaskNode(task_id="a", description="count badges", status=2, result="246 badges", parent_task_id="root"))
    g.add_task(TaskNode(task_id="b", description="split rooms", status=1, parent_task_id="root"))  # not accepted
    g.add_task(TaskNode(task_id="x", description="elsewhere", status=2, result="99", parent_task_id="a"))  # grandchild
    assert orch._built_on_results("root") == ["count badges: 246 badges"]


def test_an_abandoned_executor_root_returns_its_last_report_without_the_banner():
    orch, agent = _executor_root()
    orch.last_partial_result[orch.root_task_id] = "The power is 2300 W."
    orch.abandoned_tasks.add(orch.root_task_id) if isinstance(orch.abandoned_tasks, set) \
        else orch.abandoned_tasks.append(orch.root_task_id)
    orch.task_graph.tasks[orch.root_task_id].result = "[ABANDONED after 4 attempt(s) -- attempt cap.]"
    answer = str(orch.terminate())
    assert "The power is 2300 W." in answer and "ABANDONED" not in answer


T3_01 = ("Minimize f(x,y) = (x-2)^2 + 4(y-1)^2 subject to x+y = 1 and x,y >= 0. Return the "
         "minimum function value, not the minimizing coordinates.")


def _t3_01(monkeypatch, serving_mode, tool_result):
    if serving_mode:
        monkeypatch.setenv("HIVE_SERVING", serving_mode)
    else:
        monkeypatch.delenv("HIVE_SERVING", raising=False)
    orch = _orch()
    orch.spec = {"raw_text": T3_01, "supplies_data": True}
    orch.task_graph.add_task(TaskNode(task_id="q", description=T3_01, agent_id="a", status=1))
    node = SimpleNamespace(last_tool_result=tool_result)
    return orch._ungrounded_figures("The minimum function value is 3.2.", "q",
                                    tool_output=orch._clean_tool_output(node))


def test_v2_a_figure_the_agents_own_code_printed_is_grounded(monkeypatch):
    # Gate run T3-01: scipy printed 3.2, the REPORT said 3.2, and it was listed
    # as "from nowhere" three times.
    assert _t3_01(monkeypatch, "v2", "[TOOL RESULT - run_code]: SUCCESS: 3.2") == []


def test_v2_a_failed_tool_call_grounds_nothing(monkeypatch):
    crash = "[TOOL RESULT - run_code]: ERROR (script_crash): CRASH VERDICT: TypeError: 3.2"
    assert _t3_01(monkeypatch, "v2", crash) == ["3.2"]


def test_legacy_still_keeps_tool_output_out_of_the_grounding(monkeypatch):
    assert _t3_01(monkeypatch, None, "[TOOL RESULT - run_code]: SUCCESS: 3.2") == ["3.2"]


def test_v2_report_wiring_reads_the_live_agents_tool_output(monkeypatch):
    # The orchestrator must take the printout from the LIVE agent: the colony's
    # registry record has no last_tool_result (gate run T3-01, second try).
    monkeypatch.setenv("HIVE_SERVING", "v2")
    from judge import Judge
    prompts = []
    orch = _orch(root_role="executor", synthesizer=_NoChildrenSynth(), model=object(),
                 tokeniser=object(),
                 judge=Judge(llm_call_fn=lambda p: prompts.append(p) or " accept"))
    orch.initialize_colony(T3_01)
    orch.spec = {"raw_text": T3_01, "supplies_data": True}
    agent = orch.live_agents[orch.task_graph.tasks[orch.root_task_id].agent_id]
    agent.last_tool_result = "[TOOL RESULT - run_code]: SUCCESS: 3.2"
    agent.last_tool_args = {"code_string": "print(3.2)"}
    agent.report("The minimum function value is 3.2.")
    orch._route_events(orch.messenger.drain())
    assert prompts, "the REPORT never reached the critique"
    assert "FIGURES TO CHECK" not in prompts[-1]
    assert "CODE THE AGENT RAN PRINTED: SUCCESS: 3.2" in prompts[-1]
    assert "THAT CODE:\nprint(3.2)" in prompts[-1]


def test_a_failed_call_shows_no_code(monkeypatch):
    orch, agent = _executor_root()
    agent.last_tool_result = "[TOOL RESULT - run_code]: ERROR (script_crash): CRASH VERDICT: x"
    agent.last_tool_args = {"code_string": "print(1/0)"}
    assert orch._tool_code_for_critique(agent) is None


def _tool_event(agent_id, code):
    from event_queue import Event
    ev = Event(type="tool_request", from_agent=agent_id)
    ev.payload.update({"agent_id": agent_id, "tool_name": "run_code", "args": {"code_string": code}})
    return ev


def test_the_code_is_recorded_with_its_result_not_at_dispatch(monkeypatch):
    import orchestrator as orch_mod
    orch, agent = _executor_root()
    monkeypatch.setattr(orch_mod.ToolRegistry, "execute",
                        staticmethod(lambda *a, **k: {"status": "success", "output": "0.24"}))
    orch.handle_tool_request(_tool_event(agent.agent_id, "print(F(3, 5) * F(2, 5))"))
    assert agent.last_tool_args == {"code_string": "print(F(3, 5) * F(2, 5))"}
    # Reviewer case: a corrected call the energy ceiling skips never ran, so
    # the critique must keep seeing the code that printed 0.24.
    monkeypatch.setattr(orch, "_tool_round_would_cross_ceiling", lambda *a: True)
    orch.handle_tool_request(_tool_event(agent.agent_id, "print(F(3, 5) * F(2, 4))"))
    assert agent.last_tool_args == {"code_string": "print(F(3, 5) * F(2, 5))"}


def test_a_call_without_code_shows_no_code():
    orch, agent = _executor_root()
    agent.last_tool_result = "[TOOL RESULT - verify_math]: SUCCESS: True"
    agent.last_tool_args = {"expression": "2 + 2 = 4", "mode": "computational"}
    assert orch._tool_code_for_critique(agent) is None
