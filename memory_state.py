"""
memory_store.py

Persistent vector memory for Project Hive.

Two independent FAISS indices:
    - ghost_index   : failure records. Persists across sessions (written to disk).
    - success_index : completed-subtask cache. Lives only for the current session
                      (never touches disk, wiped on clear_session()).
"""

import os
import json
import time
import threading

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

SUCCESS_CACHE_THRESHOLD = 0.45


class MemoryStore:
    def __init__(self, ghost_persist_path=None, embedding_dim=EMBEDDING_DIM, embed_model=None):
        self.embedding_dim = embedding_dim
        self._lock = threading.Lock()
        self._embedder = embed_model or SentenceTransformer(EMBEDDING_MODEL_NAME)

        self.ghost_persist_path = ghost_persist_path
        self.ghost_index = None
        self.ghost_metadata = {}
        self._ghost_next_id = 0

        self.success_index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        self.success_metadata = {}
        self._success_next_id = 0

        self._load_ghosts()

    def _embed(self, text: str) -> np.ndarray:
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return np.asarray([vec], dtype="float32")

    def write(self, record_type: str, text: str, metadata: dict) -> int:
        if record_type not in ("ghost", "success"):
            raise ValueError(f"Unknown record_type: {record_type!r}")

        vec = self._embed(text)

        with self._lock:
            if record_type == "ghost":
                if self.ghost_index is None:
                    self.ghost_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
                record_id = self._ghost_next_id
                self._ghost_next_id += 1
                self.ghost_index.add_with_ids(vec, np.array([record_id], dtype="int64"))
                self.ghost_metadata[record_id] = {
                    **metadata,
                    "task_description": text,
                    "timestamp": time.time(),
                }
                self._save_ghosts_unlocked()
            else:
                record_id = self._success_next_id
                self._success_next_id += 1
                self.success_index.add_with_ids(vec, np.array([record_id], dtype="int64"))
                self.success_metadata[record_id] = {
                    **metadata,
                    "task_description": text,
                    "timestamp": time.time(),
                }

        return record_id

    def query(self, text: str, record_type: str, top_k: int = 5) -> list:
        if record_type not in ("ghost", "success"):
            raise ValueError(f"Unknown record_type: {record_type!r}")

        index = self.ghost_index if record_type == "ghost" else self.success_index
        metadata_store = self.ghost_metadata if record_type == "ghost" else self.success_metadata

        if index is None or index.ntotal == 0:
            return []

        vec = self._embed(text)
        k = min(top_k, index.ntotal)
        scores, ids = index.search(vec, k)

        results = []
        for score, rid in zip(scores[0], ids[0]):
            if rid == -1:
                continue
            results.append({
                "id": int(rid),
                "score": float(score),
                "metadata": metadata_store.get(int(rid), {}),
            })
        return results

    def get_success_cache(self, task_description: str, threshold: float = SUCCESS_CACHE_THRESHOLD):
        hits = self.query(task_description, "success", top_k=1)
        if not hits:
            return None
        best = hits[0]
        if best["score"] >= threshold:
            return best["metadata"]
        return None

    def query_ghosts(self, task_description: str, top_k: int = 3) -> list:
        return self.query(task_description, "ghost", top_k=top_k)

    def clear_session(self):
        with self._lock:
            self.success_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
            self.success_metadata = {}
            self._success_next_id = 0

    def _save_ghosts_unlocked(self):
        """Persist ghost_index + ghost_metadata to disk. Caller must hold self._lock.

        FIX: confirmed via a real crash -- this previously wrote both files
        directly (`open(path, "w")` / `faiss.write_index(path)`), with no
        atomicity at all. Ghosts are saved on every single write() call, so
        any interruption during a save (Kaggle kernel timeout, OOM kill, or
        manual stop -- all common during a colony run that's actively
        spiraling through repeated agent deaths) can leave a truncated,
        half-written JSON file on disk. Confirmed exactly this: a
        subsequent run's __init__ -> _load_ghosts() -> json.load() crashed
        immediately with "JSONDecodeError: Expecting value" at a mid-object
        byte offset, taking down colony construction for a COMPLETELY
        unrelated later problem before a single tick ever ran.

        Now writes to a temporary path first, then atomically renames over
        the real path with os.replace() -- which on both POSIX and Windows
        either fully succeeds or leaves the original file completely
        untouched. There is no window where a reader can observe a
        partially-written file at the real path.
        """
        if not self.ghost_persist_path:
            return
        os.makedirs(os.path.dirname(self.ghost_persist_path) or ".", exist_ok=True)

        faiss_path = self.ghost_persist_path + ".faiss"
        json_path = self.ghost_persist_path + ".json"

        if self.ghost_index is not None:
            tmp_faiss_path = faiss_path + ".tmp"
            faiss.write_index(self.ghost_index, tmp_faiss_path)
            os.replace(tmp_faiss_path, faiss_path)

        tmp_json_path = json_path + ".tmp"
        with open(tmp_json_path, "w") as f:
            json.dump({
                "next_id": self._ghost_next_id,
                "metadata": self.ghost_metadata,
            }, f)
        os.replace(tmp_json_path, json_path)

    def save_ghosts(self):
        with self._lock:
            self._save_ghosts_unlocked()

    def _load_ghosts(self):
        """Load ghost_index + ghost_metadata from disk if they exist.

        FIX: confirmed via a real crash -- this previously had zero error
        handling around faiss.read_index()/json.load(). A ghost file
        corrupted by a prior interrupted write crashed MemoryStore.__init__()
        outright, which crashed build_orchestrator() before a single tick of
        a totally unrelated future run ever executed. A corrupted ghost
        file is lost history, not a reason to refuse to start -- degrade to
        an empty ghost index and let the run proceed. Combined with the
        atomic-write fix above, this should only ever trigger on a file
        corrupted before this fix was deployed.
        """
        if not self.ghost_persist_path:
            return
        faiss_path = self.ghost_persist_path + ".faiss"
        json_path = self.ghost_persist_path + ".json"
        if os.path.exists(faiss_path) and os.path.exists(json_path):
            try:
                loaded_index = faiss.read_index(faiss_path)
                with open(json_path, "r") as f:
                    data = json.load(f)
                loaded_metadata = {int(k): v for k, v in data["metadata"].items()}
                loaded_next_id = data["next_id"]
            except Exception as e:
                print(
                    f"Warning: ghost persistence at {self.ghost_persist_path!r} is "
                    f"corrupted or unreadable ({e!r}) -- starting with an empty "
                    f"ghost index instead of crashing colony construction. The "
                    f"corrupted file(s) will be overwritten on the next ghost write."
                )
                return
            self.ghost_index = loaded_index
            self.ghost_metadata = loaded_metadata
            self._ghost_next_id = loaded_next_id