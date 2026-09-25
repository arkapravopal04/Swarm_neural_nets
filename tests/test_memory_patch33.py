"""
PATCH 33 -- memory store: the success cache is cleared at the start of every
colony run, and the ghost index is read back on every respawn, MEASURE-ONLY.

No faiss and no model download: the orchestrator tests use a small in-memory
store with the MemoryStore interface and a deterministic bag-of-words stub
embedder (cosine on normalised vectors, like IndexFlatIP on MiniLM output).
The tests against the real MemoryStore take the same stub embedder and are
skipped where faiss / sentence_transformers are not installed.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import orchestrator as orchestrator_module
from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


DIM = 64


class _StubEmbedder:
    """Deterministic: each word hashes to a fixed unit vector; a text is the
    normalised sum. Shared words -> high cosine, disjoint words -> ~0."""

    def encode(self, text, normalize_embeddings=True):
        vec = np.zeros(DIM, dtype="float32")
        for word in str(text).lower().split():
            seed = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
            vec += np.random.default_rng(seed).standard_normal(DIM).astype("float32")
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class _StubStore:
    """MemoryStore's interface, backed by lists instead of FAISS."""

    def __init__(self):
        self._embedder = _StubEmbedder()
        self.ghosts = []      # (vector, metadata)
        self.successes = []   # (vector, metadata)
        self.clear_calls = 0
        self.ghost_queries = []

    def write(self, record_type, text, metadata):
        entry = (self._embedder.encode(text), {**metadata, "task_description": text})
        (self.ghosts if record_type == "ghost" else self.successes).append(entry)
        return len(self.ghosts) + len(self.successes)

    def _query(self, entries, text, top_k):
        q = self._embedder.encode(text)
        scored = sorted(((float(np.dot(q, v)), m) for v, m in entries),
                        key=lambda s: s[0], reverse=True)
        return [{"id": i, "score": s, "metadata": m}
                for i, (s, m) in enumerate(scored[:top_k])]

    def query_ghosts(self, task_description, top_k=3):
        self.ghost_queries.append((task_description, top_k))
        return self._query(self.ghosts, task_description, top_k)

    def get_success_cache(self, description, task_id=None):
        return None   # never serves: every respawn in these tests goes through

    def clear_session(self):
        self.clear_calls += 1
        self.successes = []


TASK = "list the hazardous waste types collected at the park cleanup"


def _orchestrator(store, budget=1000):
    orch = Orchestrator(ColonyState(initial_budget=budget, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=store)
    orch.task_graph.add_task(TaskNode(task_id="t-1", description=TASK,
                                      agent_id="a1", status=1))
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id=None, task=TASK, task_id="t-1",
                                         fail_reason="invented business metrics for the park"))
    return orch


def _capture_respawn(orch, monkeypatch):
    """Record what _retire_agent returned and what spawn_agent was handed."""
    seen = {}
    real_retire = orch._retire_agent

    def retire(*args, **kwargs):
        seen["retired_ghost_context"] = real_retire(*args, **kwargs)
        return seen["retired_ghost_context"]

    def spawn(role, task_id, parent_id=None, ghost_context=None):
        seen["spawn"] = {"role": role, "task_id": task_id, "parent_id": parent_id,
                         "ghost_context": ghost_context}
        return "a2"

    monkeypatch.setattr(orch, "_retire_agent", retire)
    monkeypatch.setattr(orch, "spawn_agent", spawn)
    return seen


def _earlier_ghost(store, agent_id="old-7", task=TASK,
                   reason="invented business metrics nobody supplied"):
    store.write("ghost", task, {"agent_id": agent_id, "task": task,
                                "failure_type": "tier3_reject",
                                "failure_reason": reason})


# ------------------------------------------------------------ change 1

def test_initialize_colony_clears_the_success_cache():
    store = _StubStore()
    store.write("success", TASK, {"result": "an answer from the previous prompt",
                                  "task_id": "t-1", "outcome": "accepted"})
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None), TaskGraph(),
                        Messenger(), memory_store=store)

    orch.initialize_colony("Organise a community park cleanup day")

    assert store.clear_calls == 1
    assert store.successes == []


def test_ghosts_survive_initialize_colony():
    store = _StubStore()
    _earlier_ghost(store)
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None), TaskGraph(),
                        Messenger(), memory_store=store)

    orch.initialize_colony("Organise a community park cleanup day")

    assert len(store.ghosts) == 1
    assert store.ghosts[0][1]["agent_id"] == "old-7"


def test_initialize_colony_tolerates_a_store_without_clear_session():
    class _NoClear:
        def write(self, record_type, text, metadata):
            return 1

        def get_success_cache(self, description, task_id=None):
            return None

    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None), TaskGraph(),
                        Messenger(), memory_store=_NoClear())
    orch.initialize_colony("Organise a community park cleanup day")
    assert "root_task_0" in orch.task_graph.tasks


# ------------------------------------------------------------ change 2

def test_a_similar_earlier_ghost_is_counted_and_ghost_context_is_unchanged(monkeypatch, capsys):
    store = _StubStore()
    _earlier_ghost(store)
    orch = _orchestrator(store)
    seen = _capture_respawn(orch, monkeypatch)

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    assert orch.colony.verdict_counts["ghost_lessons_available"] == 1
    assert "ghost_lessons_none" not in orch.colony.verdict_counts
    # Measure-only: spawn_agent gets exactly what _retire_agent produced.
    assert seen["spawn"]["task_id"] == "t-1"
    assert seen["spawn"]["ghost_context"] is seen["retired_ghost_context"]
    assert "invented business metrics nobody supplied" not in repr(seen["spawn"]["ghost_context"])
    out = capsys.readouterr().out
    assert "[ghost-memory] t-1: 1 earlier failure(s) on similar tasks" in out
    assert "tier3_reject" in out


def test_the_ghost_of_the_agent_just_retired_is_excluded(monkeypatch, capsys):
    store = _StubStore()
    orch = _orchestrator(store)
    _capture_respawn(orch, monkeypatch)

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    # _retire_agent wrote a1's own ghost (score ~1.0 against its task) ...
    assert [m["agent_id"] for _, m in store.ghosts] == ["a1"]
    assert store.ghost_queries, "the ghost index was never read back"
    # ... and it is not an "earlier lesson".
    assert orch.colony.verdict_counts["ghost_lessons_none"] == 1
    assert "ghost_lessons_available" not in orch.colony.verdict_counts
    assert "[ghost-memory]" not in capsys.readouterr().out


def test_a_ghost_below_the_threshold_is_not_a_lesson(monkeypatch):
    store = _StubStore()
    _earlier_ghost(store, task="compose a sonnet about orbital mechanics",
                   reason="rhymed badly")
    orch = _orchestrator(store)
    _capture_respawn(orch, monkeypatch)

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    assert orch.colony.verdict_counts["ghost_lessons_none"] == 1
    assert "ghost_lessons_available" not in orch.colony.verdict_counts


def test_query_ghosts_raising_warns_and_the_respawn_still_happens(monkeypatch, capsys):
    class _Broken(_StubStore):
        def query_ghosts(self, task_description, top_k=3):
            raise RuntimeError("simulated ghost index failure")

    store = _Broken()
    orch = _orchestrator(store)
    seen = _capture_respawn(orch, monkeypatch)

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    assert seen["spawn"]["task_id"] == "t-1"
    assert "Warning: ghost read-back failed for t-1" in capsys.readouterr().out
    assert "ghost_lessons_available" not in orch.colony.verdict_counts
    assert "ghost_lessons_none" not in orch.colony.verdict_counts


def test_long_failure_reasons_are_trimmed_in_the_print(monkeypatch, capsys):
    store = _StubStore()
    _earlier_ghost(store, reason="invented business metrics " + "x" * 400)
    orch = _orchestrator(store)
    _capture_respawn(orch, monkeypatch)

    orch._kill_and_respawn("a1", "t-1", "executor", None, verdict=None)

    line = [l for l in capsys.readouterr().out.splitlines() if "tier3_reject" in l][0]
    assert "x" * 120 not in line
    assert "..." in line


def test_ledger_row_reports_respawns_with_lessons(capsys):
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None), TaskGraph(),
                        Messenger(), memory_store=_StubStore())
    for _ in range(2):
        orch.colony.record_verdict("ghost_lessons_available")
    for _ in range(3):
        orch.colony.record_verdict("ghost_lessons_none")

    orch._print_energy_report()

    assert ("ghost memory: respawns with earlier lessons available : 2 of 5"
            in capsys.readouterr().out)


# ------------------------------------------------------ real MemoryStore

def test_threshold_matches_memory_state():
    memory_state = pytest.importorskip("memory_state", exc_type=ImportError)
    assert orchestrator_module.GHOST_LESSON_THRESHOLD == memory_state.SUCCESS_CACHE_THRESHOLD


def test_real_store_clear_session_keeps_ghosts(tmp_path):
    memory_state = pytest.importorskip("memory_state", exc_type=ImportError)
    store = memory_state.MemoryStore(ghost_persist_path=str(tmp_path / "ghosts"),
                                     embedding_dim=DIM, embed_model=_StubEmbedder())
    store.write("success", TASK, {"result": "r", "task_id": "t-1", "outcome": "accepted"})
    store.write("ghost", TASK, {"agent_id": "old-7", "failure_type": "die"})

    store.clear_session()

    assert store.success_index.ntotal == 0
    assert store.get_success_cache(TASK).hit is None
    hits = store.query_ghosts(TASK, top_k=3)
    assert [h["metadata"]["agent_id"] for h in hits] == ["old-7"]
    assert hits[0]["score"] >= orchestrator_module.GHOST_LESSON_THRESHOLD
