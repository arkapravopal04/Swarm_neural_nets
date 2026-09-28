"""
The success cache records the judging outcome on every entry and serves only
accepted ones. A failed attempt (DIE, judge EXECUTE, tier-3 reject) is written
as a negative entry with no result, and a lookup that matches only negative
entries is a miss the caller can still count -- no task is ever closed as
completed off one.

No faiss here: the store is a fake that records writes and hands back a
SuccessLookup-shaped answer, which is all _kill_and_respawn reads.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


class _Store:
    def __init__(self, lookup=None):
        self.writes = []
        self.lookup = lookup

    def write(self, record_type, text, metadata):
        self.writes.append((record_type, text, metadata))
        return len(self.writes)

    def get_success_cache(self, description, task_id=None):
        return self.lookup


def _orchestrator(store):
    orch = Orchestrator(ColonyState(initial_budget=1000, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=store)
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="list the waste types",
                                      agent_id="a1", status=1))
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id=None, task="list the waste types",
                                         task_id="t-1"))
    return orch


def _cache_writes(store):
    return [m for kind, _, m in store.writes if kind == "success"]


def test_a_die_writes_a_negative_entry_with_no_result():
    store = _Store()
    orch = _orchestrator(store)

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    assert _cache_writes(store) == [{"result": None, "task_id": "t-1", "outcome": "die"}]
    assert orch.colony.verdict_counts["cache_write_negative_die"] == 1
    assert "cache_write_positive" not in orch.colony.verdict_counts


def test_judge_rejections_are_recorded_by_tier():
    for verdict, outcome in [
        ({"verdict": "execute", "tier": 3}, "tier3_reject"),
        ({"verdict": "execute", "tier": 2}, "tier2_execute"),
        ({"verdict": "execute", "tier": 1}, "tier1_execute"),
        ({"verdict": "warn", "tier": 2}, "warn_exhausted"),
        ({"verdict": "execute", "reason": "software framing"}, "structural_reject"),
    ]:
        store = _Store()
        orch = _orchestrator(store)
        orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=verdict)
        assert [m["outcome"] for m in _cache_writes(store)] == [outcome]


def test_a_match_on_negative_entries_only_is_a_counted_miss():
    lookup = SimpleNamespace(hit=None, hit_score=0.0, negative_count=3,
                             negative_outcomes={"die": 2, "tier3_reject": 1})
    orch = _orchestrator(_Store(lookup))

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    assert orch.task_graph.tasks["t-1"].status != 2
    assert orch.colony.verdict_counts["cache_miss_negative_match"] == 1
    assert "cache_hit_served" not in orch.colony.verdict_counts


def test_a_hit_without_an_accepted_outcome_is_never_served():
    # A store that hands back an entry with no outcome (written before the
    # field existed) must not close the task: fail closed.
    lookup = SimpleNamespace(hit={"result": "someone else's answer", "task_id": "t-9"},
                             hit_score=0.8, negative_count=0, negative_outcomes={})
    orch = _orchestrator(_Store(lookup))

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    assert orch.task_graph.tasks["t-1"].status != 2
    assert orch.task_graph.tasks["t-1"].result != "someone else's answer"
    assert "cache_hit_served" not in orch.colony.verdict_counts


def test_an_accepted_hit_is_served_and_counted_as_cross_task():
    lookup = SimpleNamespace(hit={"result": "donor answer", "task_id": "t-9",
                                  "outcome": "accepted"},
                             hit_score=0.8, negative_count=1, negative_outcomes={"die": 1})
    orch = _orchestrator(_Store(lookup))

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    assert orch.task_graph.tasks["t-1"].status == 2
    assert orch.task_graph.tasks["t-1"].result == "donor answer"
    assert orch.colony.verdict_counts["cache_hit_served"] == 1
    assert orch.colony.verdict_counts["cache_hit_served_cross_task"] == 1
    assert "cache_miss_negative_match" not in orch.colony.verdict_counts
