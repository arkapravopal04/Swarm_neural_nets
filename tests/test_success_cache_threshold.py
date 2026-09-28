"""
Cross-task success-cache donors need CROSS_TASK_SUCCESS_CACHE_THRESHOLD;
a task's own entries keep SUCCESS_CACHE_THRESHOLD.

memory_state imports faiss and sentence_transformers at module level, so
both are stubbed for the import; the lookup under test only calls
self.query, which is replaced with fixed neighbours.
"""
import importlib
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


@pytest.fixture
def memory_state(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=None))
    monkeypatch.delitem(sys.modules, "memory_state", raising=False)
    module = importlib.import_module("memory_state")
    yield module
    sys.modules.pop("memory_state", None)


def _store(module, hits):
    store = object.__new__(module.MemoryStore)
    store.success_index = types.SimpleNamespace(ntotal=len(hits))
    store.query = lambda text, record_type, top_k=5: hits
    return store


def _hit(score, task_id, outcome="accepted"):
    return {"id": 0, "score": score,
            "metadata": {"task_id": task_id, "outcome": outcome, "result": f"from {task_id}"}}


def test_cross_task_donor_below_its_threshold_is_not_served(memory_state):
    lookup = _store(memory_state, [_hit(0.67, "t-other")]).get_success_cache("d", task_id="t-me")
    assert lookup.hit is None
    assert lookup.cross_task_below_threshold == 1


def test_cross_task_donor_above_its_threshold_is_served(memory_state):
    lookup = _store(memory_state, [_hit(0.9, "t-other")]).get_success_cache("d", task_id="t-me")
    assert lookup.hit["task_id"] == "t-other" and lookup.hit_score == 0.9


def test_same_task_entry_keeps_the_base_threshold(memory_state):
    assert memory_state.SUCCESS_CACHE_THRESHOLD < 0.5 < memory_state.CROSS_TASK_SUCCESS_CACHE_THRESHOLD
    lookup = _store(memory_state, [_hit(0.5, "t-me")]).get_success_cache("d", task_id="t-me")
    assert lookup.hit["task_id"] == "t-me"


def test_blocked_cross_task_donor_does_not_shadow_an_own_entry_behind_it(memory_state):
    hits = [_hit(0.7, "t-other"), _hit(0.6, "t-me")]
    lookup = _store(memory_state, hits).get_success_cache("d", task_id="t-me")
    assert lookup.hit["task_id"] == "t-me"
    assert lookup.cross_task_below_threshold == 1


def test_without_task_id_every_entry_uses_the_base_threshold(memory_state):
    lookup = _store(memory_state, [_hit(0.67, "t-other")]).get_success_cache("d")
    assert lookup.hit["task_id"] == "t-other"


class _RecordingStore:
    def __init__(self):
        self.calls = []

    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        self.calls.append((description, task_id))
        return types.SimpleNamespace(hit=None, hit_score=0.0, negative_count=0,
                                     negative_outcomes={}, cross_task_below_threshold=2)


def test_orchestrator_looks_up_with_its_task_id_and_counts_blocked_donors():
    store = _RecordingStore()
    orch = Orchestrator(ColonyState(initial_budget=1000, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=store)
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="list the waste types",
                                      agent_id="a1", status=1))
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id=None, task="list the waste types",
                                         task_id="t-1"))

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    assert store.calls == [("list the waste types", "t-1")]
    assert orch.colony.verdict_counts["cache_cross_task_below_threshold"] == 1
