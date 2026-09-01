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
