"""
Regression tests for the run-bounding fixes:

  * per-task attempt cap (a task can no longer recycle agents forever),
  * SPAWN rejected in code for a non-decomposer (the prompt said so; only
    the prompt enforced it),
  * requirement inheritance on no keyword overlap (was: inherit everything),
  * an unparseable action falls back to THINK rather than REPORT.

Each of these was previously only observable by reading a 4,000-line run
log, which is exactly how they survived as long as they did.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description):
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
