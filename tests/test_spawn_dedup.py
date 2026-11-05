"""
PATCH 35. Duplicate subtasks removed at spawn time.

Before a subtask is spawned it is compared with every task that is live or
completed. A match is not
spawned: the existing task becomes the dependency in its place -- for the
siblings that named it, and for the decomposer that asked for it. A
decomposer that SPAWNs again (second batch, or a respawned agent on the
same task) therefore only starts the parts no completed subtask covers.

The tool cap (Agent.MAX_TOOL_ATTEMPTS_PER_AGENT) is untouched at 4.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent_node import Agent
from colony_state import AgentNode, ColonyState
from event_queue import Event, Messenger
from orchestrator import SPAWN_DEDUP_COSINE, Orchestrator
from task_graph import TaskGraph, TaskNode


class _Live:
    """Stands in for a live Agent: the fields handle_spawn's release path
    and handle_parent_notification touch."""

    def __init__(self, task_id, role="decomposer"):
        self.task_id = task_id
        self.role = role
        self.awaiting = "children"
        self.fail_reason = None
        self.thought_process = ""
        self.non_terminal_cycles = 0
        self.cycles_capped = False
        self.parent_id = None


def _orch(embed_model=None):
    orch = Orchestrator(
        ColonyState(initial_budget=10000, goal_embedding=None),
        TaskGraph(),
        Messenger(),
        embed_model=embed_model,
    )
    orch.spec = {"goal": "g", "raw_text": "g", "requirement": []}
    return orch


def _add(orch, task_id, description, status=1, parent_task_id=None,
         dependencies=None, result=None, agent_id=None):
    node = TaskNode(task_id=task_id, description=description, status=status,
                    parent_task_id=parent_task_id, dependencies=dependencies or [],
                    result=result, agent_id=agent_id)
    orch.task_graph.add_task(node)
    node.status = status  # add_task does not touch status; keep it explicit
    return node


def _decomposer(orch, agent_id, task_id, live=True):
    orch.colony.register_agent(AgentNode(agent_id=agent_id, role="decomposer",
                                         status="running", parent_id=None,
                                         task="t", task_id=task_id))
    orch.task_graph.tasks[task_id].agent_id = agent_id
    if live:
        orch.live_agents[agent_id] = _Live(task_id)
    return orch.live_agents.get(agent_id)


def _wire_spawn_recorder(orch, monkeypatch):
    """Strip the gates that need a model or a budget model, and record
    what would be spawned instead of spawning it."""
    spawned = []
    monkeypatch.setattr(orch, "_screen_derived_subtask_batch",
                        lambda event, descs, originals: (True, []))
    monkeypatch.setattr(orch, "_screen_goal_drift_batch", lambda prepared: [])
    monkeypatch.setattr(orch, "_admit_spawn_batch",
                        lambda task_id, roles: len(roles))

    def record(**kw):
        spawned.append(kw)
        _add(orch, kw["task_id"], kw["description"], status=1,
             parent_task_id=orch.colony.get_agent(kw["parent_id"]).task_id,
             dependencies=kw.get("dependencies"))
    monkeypatch.setattr(orch, "_spawn_child_task", record)
    return spawned


def _spawn(orch, parent_id, subtasks):
    orch.handle_spawn(Event(type="spawn_request", from_agent=parent_id,
                            payload={"parent_id": parent_id, "subtasks": subtasks}))


def _tree(orch):
    """root -> A (decomposer, running), root -> B (completed elsewhere)."""
    _add(orch, "root", "the whole project", status=1)
    orch.root_task_id = "root"
    _add(orch, "A", "plan part A", status=1, parent_task_id="root")
    _add(orch, "B", "Measure the blade temperature", status=2,
         parent_task_id="root", result="1450C at the tip")


# ------------------------------------------------------------ matching

def test_completed_duplicate_is_linked_not_spawned(monkeypatch):
    orch = _orch()
    _tree(orch)
    _decomposer(orch, "dec-A", "A")
    spawned = _wire_spawn_recorder(orch, monkeypatch)

    _spawn(orch, "dec-A", [
        {"role": "executor", "task": "measure the blade temperature.", "label": "temp",
         "dependencies": []},
        {"role": "executor", "task": "Size the cooling channels", "label": "cool",
         "dependencies": ["temp"]},
    ])

    assert [s["description"] for s in spawned] == ["Size the cooling channels"]
    # The sibling that depended on the duplicate depends on B instead.
    assert spawned[0]["dependencies"] == ["B"]
    assert orch.borrowed_tasks == {"A": ["B"]}
    notes = [e for e in orch.messenger.drain() if e.type == "parent_notification"]
    assert [(n.payload["parent_id"], n.payload["task_id"]) for n in notes] == [("dec-A", "B")]
    assert orch.colony.verdict_counts.get("spawn_dedup_completed") == 1


def test_live_duplicate_is_waited_on_and_delivers_on_close(monkeypatch):
    orch = _orch()
    _tree(orch)
    _add(orch, "C", "Pick the coating", status=1, parent_task_id="root")
    _decomposer(orch, "dec-A", "A")
    spawned = _wire_spawn_recorder(orch, monkeypatch)

    _spawn(orch, "dec-A", [{"role": "executor", "task": "pick the coating",
                            "dependencies": []}])

    assert spawned == []
    assert orch._open_child_task_ids("dec-A") == ["C"]
    assert orch.live_agents["dec-A"].awaiting == "children", \
        "a live reused task is still in flight: the decomposer keeps waiting"

    orch.messenger.drain()
    orch.task_graph.tasks["C"].status = 2
    orch._notify_borrowers("C", "YSZ, 300um")
    notes = [e for e in orch.messenger.drain() if e.type == "parent_notification"]
    assert [n.payload["parent_id"] for n in notes] == ["dec-A"]
    assert orch._open_child_task_ids("dec-A") == []
    assert orch._has_completed_direct_child("A")


def test_whole_batch_reused_releases_decomposer_to_report(monkeypatch):
    orch = _orch()
    _tree(orch)
    live = _decomposer(orch, "dec-A", "A")
    spawned = _wire_spawn_recorder(orch, monkeypatch)

    _spawn(orch, "dec-A", [{"role": "executor", "task": "Measure the blade temperature",
                            "dependencies": []}])

    assert spawned == []
    assert live.awaiting is None
    assert "Do not SPAWN them again" in live.fail_reason
    assert orch.colony.get_agent("dec-A").has_spawned, \
        "REPORT must be on its menu, not SPAWN-again"
    assert not [e for e in orch.messenger.drain() if e.type == "failure_request"]


def test_respawned_decomposer_only_spawns_uncovered_parts(monkeypatch):
    """A decomposer on task A failed after its first child completed; its
    successor re-plans the whole task. The finished part is linked, not
    redone, and is not double-listed as borrowed (it is a direct child)."""
    orch = _orch()
    _tree(orch)
    _add(orch, "A1", "Collect the load cases", status=2, parent_task_id="A",
         result="12 load cases")
    _decomposer(orch, "dec-A2", "A")
    spawned = _wire_spawn_recorder(orch, monkeypatch)

    _spawn(orch, "dec-A2", [
        {"role": "executor", "task": "Collect the load cases", "label": "loads",
         "dependencies": []},
        {"role": "executor", "task": "Run the fatigue check", "dependencies": ["loads"]},
    ])

    assert [s["description"] for s in spawned] == ["Run the fatigue check"]
    assert spawned[0]["dependencies"] == ["A1"]
    assert orch.borrowed_tasks == {}
    rows = orch._direct_child_statuses("A")
    assert [r["task"] for r in rows] == ["Collect the load cases", "Run the fatigue check"]


def test_own_task_ancestors_and_root_are_never_matched():
    orch = _orch()
    _tree(orch)
    _decomposer(orch, "dec-A", "A")
    assert orch._find_duplicate_task("plan part A", None, "A") is None
    assert orch._find_duplicate_task("the whole project", None, "A") is None


def test_live_task_waiting_on_the_spawner_is_not_matched():
    """D depends on A. A linking to D would have A wait on D and D on A."""
    orch = _orch()
    _tree(orch)
    _add(orch, "D", "Write the report", status=0, parent_task_id="root",
         dependencies=["A"])
    assert orch._find_duplicate_task("write the report", None, "A") is None
    # Any other spawner may still reuse it.
    _add(orch, "E", "plan part E", status=1, parent_task_id="root")
    assert orch._find_duplicate_task("write the report", None, "E") == ("D", 1.0)


def test_completed_match_preferred_over_live():
    orch = _orch()
    _tree(orch)
    _add(orch, "L", "Measure the blade temperature", status=1, parent_task_id="root")
    assert orch._find_duplicate_task("Measure the blade temperature", None, "A")[0] == "B"


def test_failed_tasks_are_not_matched():
    orch = _orch()
    _tree(orch)
    _add(orch, "F", "Pick a supplier", status=3, parent_task_id="root")
    assert orch._find_duplicate_task("Pick a supplier", None, "A") is None


class _AngleEmbedder:
    """'<cos>|text' embeds to a unit vector at that cosine from (1, 0)."""

    def encode(self, text, convert_to_numpy=True):
        c = float(str(text).split("|", 1)[0])
        return np.array([c, np.sqrt(max(0.0, 1 - c * c))], dtype="float32")


def test_cosine_threshold():
    orch = _orch(embed_model=_AngleEmbedder())
    _tree(orch)
    base = _add(orch, "X", "1.0|anchor", status=2, parent_task_id="root", result="r")
    base.description_embedding = orch._embed_description("1.0|anchor")
    hi = orch._embed_description(f"{SPAWN_DEDUP_COSINE + 0.01}|reworded")
    lo = orch._embed_description(f"{SPAWN_DEDUP_COSINE - 0.05}|sibling")
    assert orch._find_duplicate_task("reworded", hi, "A")[0] == "X"
    assert orch._find_duplicate_task("sibling", lo, "A") is None


# ------------------------------------------------------------ prompt + caps

def test_subtask_block_tells_a_respawn_to_cover_only_the_gaps():
    block = Agent._child_status_block([
        {"task": "Collect the load cases", "status": "completed", "excerpt": "12"},
        {"task": "Measure temp", "status": "completed", "reused": True},
    ])
    assert "SPAWN only the parts of your task that no completed subtask" in block
    assert "(reused from elsewhere in the colony)" in block
    no_completed = Agent._child_status_block([{"task": "x", "status": "running"}])
    assert "SPAWN only the parts" not in no_completed


def test_tool_cap_stays_at_four():
    assert Agent.MAX_TOOL_ATTEMPTS_PER_AGENT == 4


# ------------------------------------------------------------ PATCH 36

def _dec_agent(rows, has_spawned=True):
    node = AgentNode(agent_id="d-1", role="decomposer", status="running",
                     parent_id="root", task="t", task_id="t-1",
                     generation=1, has_spawned=has_spawned)
    agent = Agent(tokeniser=None, model=None, message=Messenger(), node=node)
    agent.child_status_fn = lambda: rows
    return agent


def test_decomposer_with_completed_subtasks_cannot_spawn_again():
    agent = _dec_agent([{"task": "a", "status": "completed"},
                        {"task": "b", "status": "completed"}])
    assert agent.final_only
    assert agent.final_actions == ("REPORT", "DIE")
    # decide()'s coercion turns a SPAWN into REPORT/DIE on this menu.
    action, _ = agent._coerce_final_action(
        "SPAWN", {"subtasks": [{"role": "executor", "task": "redo a"}]})
    assert action in ("REPORT", "DIE")


def test_decomposer_keeps_spawn_to_redo_a_failed_subtask():
    agent = _dec_agent([{"task": "a", "status": "completed"},
                        {"task": "b", "status": "abandoned"}])
    assert not agent.final_only


def test_unspawned_decomposer_still_spawns():
    agent = _dec_agent([{"task": "a", "status": "completed"}], has_spawned=False)
    assert not agent.final_only
