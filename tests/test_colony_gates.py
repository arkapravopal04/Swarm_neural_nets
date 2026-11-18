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
