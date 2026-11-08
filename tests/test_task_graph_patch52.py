"""PATCH 52: a dependency target added after the task waiting on it is
linked back; dependency cycles are broken deterministically (the task added
first keeps running, its wait on the later one is dropped)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_graph import TaskGraph, TaskNode


def _node(task_id, deps=None, **kw):
    return TaskNode(task_id=task_id, description=f"task {task_id}",
                    dependencies=list(deps or []), **kw)


def test_forward_reference_is_linked_when_target_added():
    g = TaskGraph()
    g.add_task(_node("B", ["A"]))
    g.add_task(_node("A"))
    b, a = g.tasks["B"], g.tasks["A"]
    assert b.in_degree == 1
    assert a.dependents == ["B"]
    assert g.complete_task("A") == ["B"]
    assert b.in_degree == 0


def test_forward_reference_with_agent_starts_running_on_completion():
    g = TaskGraph()
    g.add_task(_node("B", ["A"]))
    g.assign_agent("B", "agent_b")
    assert g.tasks["B"].status == 0
    g.add_task(_node("A"))
    assert g.tasks["B"].status == 0
    g.complete_task("A")
    assert g.tasks["B"].status == 1


def test_two_task_cycle_keeps_first_added_running():
    g = TaskGraph()
    g.add_task(_node("A", ["B"]))
    g.add_task(_node("B", ["A"]))
    a, b = g.tasks["A"], g.tasks["B"]
    assert a.dependencies == []
    assert b.dependencies == ["A"]
    assert a.dependents == ["B"]
    assert b.dependents == []
    assert a.in_degree == 0
    assert b.in_degree == 1
    assert g.complete_task("A") == ["B"]
    assert b.in_degree == 0


def test_two_task_cycle_with_agent_releases_first_added():
    g = TaskGraph()
    g.add_task(_node("A", ["B"]))
    g.assign_agent("A", "agent_a")
    assert g.tasks["A"].status == 0
    g.add_task(_node("B", ["A"]))
    assert g.tasks["A"].status == 1


def test_three_task_cycle_does_not_stick():
    g = TaskGraph()
    g.add_task(_node("A", ["B"]))
    g.add_task(_node("B", ["C"]))
    g.add_task(_node("C", ["A"]))
    assert any(t.in_degree == 0 for t in g.tasks.values())
    done = set()
    for _ in range(3):
        ready = g.get_ready_tasks()
        assert ready, f"stuck with {sorted(set(g.tasks) - done)} unfinished"
        g.complete_task(ready[0].task_id)
        done.add(ready[0].task_id)
    assert done == {"A", "B", "C"}
    assert all(t.in_degree == 0 for t in g.tasks.values())


def test_self_dependency_is_dropped():
    g = TaskGraph()
    g.add_task(_node("A", ["A"]))
    a = g.tasks["A"]
    assert a.dependencies == []
    assert a.dependents == []
    assert a.in_degree == 0
    assert a in g.get_ready_tasks()


def test_duplicate_dependency_entries_unchanged():
    g = TaskGraph()
    g.add_task(_node("A"))
    g.add_task(_node("B", ["A", "A"]))
    a, b = g.tasks["A"], g.tasks["B"]
    assert b.in_degree == 2
    assert a.dependents == ["B", "B"]
    g.complete_task("A")
    assert b.in_degree == 0


def test_duplicate_forward_reference_also_resolves():
    g = TaskGraph()
    g.add_task(_node("B", ["A", "A"]))
    g.add_task(_node("A"))
    assert g.tasks["B"].in_degree == 2
    assert g.tasks["A"].dependents == ["B", "B"]
    g.complete_task("A")
    assert g.tasks["B"].in_degree == 0


def test_target_added_already_closed_releases_waiter():
    g = TaskGraph()
    g.add_task(_node("B", ["A"]))
    assert g.tasks["B"].in_degree == 1
    g.add_task(_node("A", status=2))
    assert g.tasks["B"].in_degree == 0
    assert g.tasks["A"].dependents == []


def test_target_added_already_failed_releases_waiter_with_agent():
    g = TaskGraph()
    g.add_task(_node("B", ["A"]))
    g.assign_agent("B", "agent_b")
    g.add_task(_node("A", status=3))
    assert g.tasks["B"].in_degree == 0
    assert g.tasks["B"].status == 1


def test_spawn_batch_listed_out_of_order_completes():
    # Mirrors orchestrator.handle_spawn: ids pre-assigned, siblings added
    # in list order, the first depends on the second.
    g = TaskGraph()
    g.add_task(_node("root"))
    g.add_task(_node("t1", ["t2"], parent_task_id="root"))
    g.assign_agent("t1", "a1")
    g.add_task(_node("t2", parent_task_id="root"))
    g.assign_agent("t2", "a2")
    assert g.tasks["t1"].status == 0
    assert g.tasks["t2"].status == 1
    assert g.complete_task("t2") == ["t1"]
    assert g.tasks["t1"].status == 1


def test_no_cycle_warning_after_break(capsys):
    g = TaskGraph()
    g.add_task(_node("A", ["B"]))
    capsys.readouterr()
    g.add_task(_node("B", ["A"]))
    out = capsys.readouterr().out
    assert "no longer waits" in out
    assert "introduces a dependency cycle" not in out
