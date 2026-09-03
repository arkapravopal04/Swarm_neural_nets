"""
Smoke test for the kill/ghost/respawn path in orchestrator.py.

Regression target: _kill_and_respawn() calls `ghost_extractor.extract(...)`
under a bare `except Exception:` (memory-write guard). A prior version of
this file called the bare name `extract(...)` instead of the
module-qualified `ghost_extractor.extract(...)` -- a NameError on every
single kill, silently swallowed by the broad except and reported only as
a generic "Warning: failed writing ghost record" line indistinguishable
from a legitimate MemoryStore failure. Nothing asserted that
memory_store.write("ghost", ...) actually ran, so that regression could
(and did) ship unnoticed.

This test does not exercise respawn itself (that needs a live model to
spawn a real Agent) -- it kills an agent with no task_id, which returns
immediately after the ghost-write block, and just asserts the ghost
record was actually written.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph


class _RecordingMemoryStore:
    def __init__(self):
        self.writes = []

    def write(self, record_type, text, metadata):
        self.writes.append((record_type, text, metadata))
        return len(self.writes)

    def get_success_cache(self, description):
        return None


def _make_orchestrator(memory_store):
    return Orchestrator(
        ColonyState(initial_budget=100, goal_embedding=None),
        TaskGraph(),
        Messenger(),
        memory_store=memory_store,
    )


def test_kill_and_respawn_writes_ghost_record():
    memory_store = _RecordingMemoryStore()
    orch = _make_orchestrator(memory_store)

    dead = AgentNode(
        agent_id="agent-1",
        role="executor",
        status="running",
        parent_id=None,
        task="add two numbers",
        fail_reason="crashed mid-tool-call",
    )
    orch.colony.register_agent(dead)

    # No task_id -> _kill_and_respawn returns right after the ghost-write
    # block, before ever touching spawn_agent (which needs a live model).
    orch._kill_and_respawn(
        agent_id="agent-1", task_id=None, role="executor", parent_id=None
    )

    ghost_writes = [w for w in memory_store.writes if w[0] == "ghost"]
    assert len(ghost_writes) == 1, (
        f"expected exactly one ghost record written on the kill path, "
        f"got {memory_store.writes!r}"
    )

    _, ghost_task, ghost_record = ghost_writes[0]
    assert ghost_record["agent_id"] == "agent-1"
    assert ghost_record["failure_reason"] == "crashed mid-tool-call"

    # The dead agent is gone from the colony either way.
    assert orch.colony.get_agent("agent-1") is None


def test_kill_and_respawn_survives_ghost_extract_failure():
    """
    A broken ghost-write path (e.g. a NameError in the extract call) must
    not crash the kill/respawn flow -- it should be caught, logged, and
    the agent still torn down. This pins that resilience contract in
    place while test_kill_and_respawn_writes_ghost_record (above) pins
    the *happy* path, so a regression that silently breaks ghost writing
    can't hide behind "well, nothing crashed."
    """

    class _BrokenMemoryStore(_RecordingMemoryStore):
        def write(self, record_type, text, metadata):
            if record_type == "ghost":
                raise RuntimeError("simulated ghost-write failure")
            return super().write(record_type, text, metadata)

    memory_store = _BrokenMemoryStore()
    orch = _make_orchestrator(memory_store)

    dead = AgentNode(
        agent_id="agent-2",
        role="executor",
        status="running",
        parent_id=None,
        task="subtract two numbers",
        fail_reason="crashed mid-tool-call",
    )
    orch.colony.register_agent(dead)

    # Must not raise.
    orch._kill_and_respawn(
        agent_id="agent-2", task_id=None, role="executor", parent_id=None
    )

    assert orch.colony.get_agent("agent-2") is None


# ── _run_live_agents(): crash path ────────────────────────────────────────
#
# Regression target: an exception out of Agent.run() used to `continue`
# straight past both the liveness stamp and the energy debit. A crash-looping
# agent therefore cost the colony nothing (so the energy-death backstop never
# fired) while its stale last_active also made the deadlock watchdog queue a
# *second*, conflicting kill on top of whatever else was happening -- and
# nothing ever escalated it, so it was retried on the same broken state every
# tick forever.

from task_graph import TaskNode  # noqa: E402  (kept next to its users)


class _CrashingAgent:
    """Stand-in for a live Agent whose run() always raises (a real Agent
    needs a loaded model)."""

    def __init__(self, agent_id, task_id, role="executor", parent_id=None):
        self.agent_id = agent_id
        self.task_id = task_id
        self.role = role
        self.parent_id = parent_id
        self.awaiting = None
        self.thought_process = ""

    def run(self, *args, **kwargs):
        raise RuntimeError("boom")


def _orchestrator_with_crashing_agent(agent_id="agent-1", register=True):
    orch = _make_orchestrator(_RecordingMemoryStore())
    task = TaskNode(task_id="task-1", description="do a thing", agent_id=agent_id, status=1)
    orch.task_graph.add_task(task)
    if register:
        orch.colony.register_agent(
            AgentNode(agent_id=agent_id, role="executor", status="running",
                      parent_id=None, task="do a thing")
        )
    orch.live_agents[agent_id] = _CrashingAgent(agent_id, "task-1")
    return orch


def test_crash_debits_energy_and_stamps_liveness():
    orch = _orchestrator_with_crashing_agent()
    node = orch.colony.get_agent("agent-1")
    node.last_active = 0.0
    budget_before = orch.colony.budget_remaining

    orch._run_live_agents()

    assert orch.colony.budget_remaining < budget_before, "crash must not be free"
    assert node.energy_spent > 0
    assert node.last_active > 0.0, "crash must refresh the liveness stamp"
    assert node.crash_count == 1
    assert "boom" in (node.fail_reason or "")


def test_repeated_crashes_escalate_to_failure_request():
    orch = _orchestrator_with_crashing_agent()
    node = orch.colony.get_agent("agent-1")

    for _ in range(orch.MAX_CONSECUTIVE_CRASHES - 1):
        orch._run_live_agents()
    assert orch.messenger.drain() == [], "must not escalate before the threshold"

    orch._run_live_agents()
    events = orch.messenger.drain()
    assert len(events) == 1
    event = events[0]
    assert event.type == "failure_request"
    assert event.from_agent == "agent-1"
    assert event.payload["task_id"] == "task-1"
    assert event.payload["role"] == "executor"
    assert node.crash_count == orch.MAX_CONSECUTIVE_CRASHES


def test_successful_tick_resets_crash_counter():
    orch = _orchestrator_with_crashing_agent()
    node = orch.colony.get_agent("agent-1")
    orch._run_live_agents()
    assert node.crash_count == 1

    orch.live_agents["agent-1"].run = lambda *a, **k: None
    orch._run_live_agents()
    assert node.crash_count == 0, "the counter tracks *consecutive* crashes"


def test_unregistered_crashing_agent_escalates_immediately():
    """With no colony node there is nowhere to debit energy or keep a crash
    count, so such an agent would crash-loop for free forever -- it must be
    routed out on the first crash instead."""
    orch = _orchestrator_with_crashing_agent(register=False)

    orch._run_live_agents()

    events = orch.messenger.drain()
    assert len(events) == 1
    assert events[0].type == "failure_request"
    assert events[0].from_agent == "agent-1"
