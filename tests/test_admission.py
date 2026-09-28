"""
Energy admission: new work -- the root's bootstrap, a SPAWN batch, a TASK TOO
LARGE conversion, a respawn -- is let in only while

    spent + every unfinished task's remaining reservation + the new work
        <= budget - ADMISSION_FLOOR

Reservations are Orchestrator.RESERVE_* (see __init__); the evaluation behind
the numbers is sims/reservation_formula/REPORT.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


def _decomposer(budget):
    """A generation-1 decomposer on t-parent (children are executors)."""
    orch = Orchestrator(ColonyState(initial_budget=budget, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=_NullMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan", agent_id="p",
                                      status=1, required_role="decomposer"))
    node = AgentNode(agent_id="p", role="decomposer", status="running", parent_id=None,
                     task="plan", task_id="t-parent", generation=1)
    orch.colony.register_agent(node)
    agent = Agent(tokeniser=None, model=None, message=orch.messenger, node=node)
    orch.live_agents["p"] = agent
    return orch, agent


def _spawn(orch, agent, payload):
    agent.request_spawn(payload)
    orch._route_events(orch.messenger.drain())


def _children(orch):
    return {tid: t for tid, t in orch.task_graph.tasks.items() if tid != "t-parent"}


def _limit(orch):
    return orch.colony.starting_budget - orch.ADMISSION_FLOOR


def _budget_fitting(k):
    """Smallest budget whose limit admits a k-executor batch from a fresh
    generation-1 decomposer, and no more."""
    o = Orchestrator(ColonyState(initial_budget=0, goal_embedding=None), TaskGraph(),
                     Messenger())
    own = o.RESERVE_DECOMPOSER_BASE + o.RESERVE_PER_CHILD * k
    return int(own + k * o.RESERVE_EXECUTOR) + o.ADMISSION_FLOOR + 1


# ------------------------------------------------------------- SPAWN batches

def test_a_batch_that_fits_is_admitted_whole():
    orch, agent = _decomposer(budget=1000)
    _spawn(orch, agent, {"subtasks": [
        {"role": "executor", "task": "list the supplies", "dependencies": []},
        {"role": "executor", "task": "assign the roles", "dependencies": []},
    ]})
    assert len(_children(orch)) == 2
    res = orch._reservation("t-parent")
    assert res.spawned and res.k == 2
    assert "admission_subtasks_dropped" not in orch.colony.verdict_counts


def test_a_batch_is_cut_to_what_fits_and_the_decomposer_is_told():
    orch, agent = _decomposer(budget=_budget_fitting(1))
    _spawn(orch, agent, {"subtasks": [
        {"role": "executor", "task": "list the supplies", "dependencies": []},
        {"role": "executor", "task": "assign the roles", "dependencies": []},
    ]})
    (child,) = _children(orch).values()
    assert child.description == "list the supplies"          # order kept, tail dropped
    assert orch._reservation("t-parent").k == 1
    assert orch.colony.verdict_counts["admission_subtasks_dropped"] == 1
    assert "[BUDGET]" in agent.thought_process and "assign the roles" in agent.thought_process
    assert agent.awaiting == "children"                       # still waits on the kept one
    assert orch.committed_energy() <= _limit(orch)


def test_a_dependency_on_a_dropped_subtask_is_removed_not_left_pending():
    orch, agent = _decomposer(budget=_budget_fitting(1))
    _spawn(orch, agent, {"subtasks": [
        {"role": "executor", "task": "combine the plan", "dependencies": ["b"]},
        {"label": "b", "role": "executor", "task": "list the supplies", "dependencies": []},
    ]})
    (child,) = _children(orch).values()
    assert child.dependencies == [] and child.in_degree == 0
    assert child.status == 1


def test_queued_overflow_is_reserved_when_its_batch_is_admitted():
    orch, agent = _decomposer(budget=1000)
    _spawn(orch, agent, {"subtasks": [
        {"role": "executor", "task": f"part {i} of the plan", "dependencies": []}
        for i in range(4)
    ]})
    assert len(orch.pending_overflow["p"]) == 2               # fan-out cap 2
    assert orch._reservation("t-parent").k == 4
    queued = 2 * orch._new_task_reservation("executor")
    spent = orch.colony.starting_budget - orch.colony.budget_remaining
    running = sum(orch._remaining_reservation(t) for t in orch.task_graph.tasks)
    assert orch.committed_energy() == spent + running + queued


def test_admission_can_be_switched_off():
    orch, agent = _decomposer(budget=_budget_fitting(1))
    orch.ADMISSION_CONTROL = False
    _spawn(orch, agent, {"subtasks": [
        {"role": "executor", "task": "list the supplies", "dependencies": []},
        {"role": "executor", "task": "assign the roles", "dependencies": []},
    ]})
    assert len(_children(orch)) == 2


# ------------------------------------------------------ conversion / respawn

def _executor_under(orch):
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="plan the whole day",
                                      agent_id="a1", status=1, required_role="executor"))
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id="p", task="plan the whole day",
                                         task_id="t-1"))
    orch._reservation("t-1").attempts = 1


def _too_large(orch):
    orch.messenger.push_event("failure_request", "a1", {
        "task_id": "t-1", "role": "executor", "parent_id": "p",
        "result": "TASK TOO LARGE: needs splitting into parts."})
    orch._route_events(orch.messenger.drain())


def test_task_too_large_converts_when_the_decomposition_fits():
    orch, _ = _decomposer(budget=1000)
    _executor_under(orch)
    orch.colony.debit_energy("a1", 20, category="think_tick")
    _too_large(orch)

    (agent,) = [a for a in orch.colony.agents.values() if a.task_id == "t-1"]
    assert agent.role == "decomposer"
    res = orch._reservation("t-1")
    assert res.kind == "decomposer" and res.conv_sunk == 20 and res.attempts == 2


def test_task_too_large_is_closed_when_decomposing_it_does_not_fit():
    orch, _ = _decomposer(budget=80)
    _executor_under(orch)
    orch.colony.debit_energy("a1", 20, category="think_tick")
    _too_large(orch)

    assert orch.task_graph.tasks["t-1"].status == 3
    assert orch.abandon_reasons["t-1"] == "admission"
    assert not [a for a in orch.colony.agents.values() if a.task_id == "t-1"]
    assert orch.colony.verdict_counts["admission_conversion_refused"] == 1
    assert orch._reservation("t-1").kind == "executor"       # re-pricing rolled back


def test_a_respawn_reserves_a_whole_attempt_and_is_refused_when_it_cannot():
    orch, _ = _decomposer(budget=80)
    _executor_under(orch)
    orch.colony.debit_energy("a1", 30, category="think_tick")

    orch._kill_and_respawn("a1", "t-1", "executor", "p")

    assert orch.task_graph.tasks["t-1"].status == 3
    assert orch.abandon_reasons["t-1"] == "no energy to respawn"
    assert orch.colony.verdict_counts["admission_respawn_refused"] == 1


def test_a_respawn_that_fits_resets_the_attempt_reservation():
    orch, _ = _decomposer(budget=1000)
    _executor_under(orch)
    orch.colony.debit_energy("a1", 30, category="think_tick")

    orch._kill_and_respawn("a1", "t-1", "executor", "p")

    res = orch._reservation("t-1")
    assert res.sunk == 30 and res.attempts == 2
    # A full attempt reserved again, less the new agent's spawn cost -- which
    # is the first thing that attempt spends.
    spawn_cost = orch.energy_when_new_by_role["executor"]
    assert orch._remaining_reservation("t-1") == orch.RESERVE_EXECUTOR - spawn_cost


# ------------------------------------------------------------------ bootstrap

def test_root_is_closed_at_bootstrap_when_even_its_cheapest_plan_does_not_fit():
    orch = Orchestrator(ColonyState(initial_budget=50, goal_embedding=None), TaskGraph(),
                        Messenger(), memory_store=_NullMemoryStore())
    orch.initialize_colony("Organise a community park cleanup day")

    root = orch.task_graph.tasks["root_task_0"]
    assert root.status == 3 and root.agent_id is None
    assert orch.abandon_reasons["root_task_0"] == "admission"
    assert orch.tick() is False


def test_root_starts_at_the_smallest_phaser_budget():
    """100 is the phaser's floor: root + one decomposer + one executor has to fit."""
    orch = Orchestrator(ColonyState(initial_budget=100, goal_embedding=None), TaskGraph(),
                        Messenger(), memory_store=_NullMemoryStore())
    orch.initialize_colony("Organise a community park cleanup day")
    assert orch.task_graph.tasks["root_task_0"].agent_id is not None
    assert orch.committed_energy() <= _limit(orch)


# ------------------------------------------------------------- energy trace

def test_terminate_appends_one_energy_trace_line_per_run(tmp_path):
    import json
    path = tmp_path / "trace.jsonl"
    for _ in range(2):
        orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None), TaskGraph(),
                            Messenger(), memory_store=_NullMemoryStore(),
                            energy_trace_path=str(path))
        orch.initialize_colony("Organise a community park cleanup day")
        orch.terminate()
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    (root,) = [t for t in lines[0]["tasks"] if t["task_id"] == "root_task_0"]
    assert root["kind"] == "root" and root["attempts"] == 1
    assert lines[0]["params"]["e"] == orch.RESERVE_EXECUTOR


def test_energy_trace_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None), TaskGraph(),
                        Messenger(), memory_store=_NullMemoryStore())
    orch.initialize_colony("Organise a community park cleanup day")
    orch.terminate()
    assert list(tmp_path.iterdir()) == []


def test_ledger_marks_an_overrun_with_a_queued_verdict_as_pending(capsys):
    """The one-cycle window: the REPORT that crossed the line is judged next
    tick. A mid-run ledger printed in between must not call it a leak."""
    orch, _ = _decomposer(budget=1000)
    _executor_under(orch)
    orch.colony.debit_energy("a1", orch.task_energy_ceiling("t-1") + 1, category="think_tick")
    orch.messenger.push_event("completion_request", "a1", {"task_id": "t-1", "result": "x"})
    capsys.readouterr()

    orch._print_energy_report()
    row = next(l for l in capsys.readouterr().out.splitlines() if "t-1" in l and "agents=" in l)
    assert "VERDICT PENDING" in row and "NOT STOPPED" not in row
