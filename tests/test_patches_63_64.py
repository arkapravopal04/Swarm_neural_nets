"""PATCH 63 (retries use the request's own words), the same-part dedup bar,
and PATCH 64 (budget floor from the request's parts), against run 21."""
import importlib.machinery
import importlib.util
import os
import sys
import types
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if importlib.util.find_spec("sentence_transformers") is None:
    _stub = types.ModuleType("sentence_transformers")
    _stub.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    _stub.SentenceTransformer = object
    sys.modules.setdefault("sentence_transformers", _stub)

from colony_state import AgentNode, ColonyState  # noqa: E402
from event_queue import Event, Messenger  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from problem_phaser import Problem_Phaser  # noqa: E402
from task_graph import TaskGraph, TaskNode  # noqa: E402

SCHOOL = ("A school is taking 45 students and 5 teachers on a 3-day trip with a budget of "
          "180,000 rupees. Choose between a bus that costs 40,000 rupees in total and a train at "
          "900 rupees per person, plan meals so that each day stays under 15,000 rupees, split the "
          "students into groups with one teacher each, and decide what to do if a student falls ill "
          "during the trip.")
MEAL_PART = "Plan meals so that each day stays under 15,000 rupees."
RUN21_MEAL = ("Allocate a daily food budget of less than 15k per meal type while respecting "
              "nutritional requirements for 45+5 people over three days.")

_TOPICS = (("bus", "train", "transport"), ("meal", "food"), ("group",), ("ill", "sick", "medical"))


class _TopicEmbedder:
    """One dimension per topic plus a per-text dimension at weight 0.7: two
    different texts on one topic sit at cosine 0.67, texts on different
    topics near 0 -- the run 21 shape (0.64-0.78 vs <= 0.4)."""

    def encode(self, text, convert_to_numpy=True):
        t = str(text).lower()
        v = np.zeros(len(_TOPICS) + 64, dtype="float32")
        for i, words in enumerate(_TOPICS):
            if any(w in t for w in words):
                v[i] = 1.0
        v[len(_TOPICS) + zlib.crc32(t.encode()) % 64] = 0.7
        return v / np.linalg.norm(v)


def _orch():
    orch = Orchestrator(ColonyState(initial_budget=10000, goal_embedding=None), TaskGraph(),
                        Messenger(), embed_model=_TopicEmbedder())
    orch.spec = {"goal": SCHOOL, "raw_text": SCHOOL, "requirement": []}
    orch.root_task_id = "root"
    _add(orch, "root", SCHOOL, status=1)
    return orch


def _add(orch, task_id, description, status=1, parent_task_id="root"):
    node = TaskNode(task_id=task_id, description=description, status=status,
                    parent_task_id=None if task_id == "root" else parent_task_id)
    node.description_embedding = orch._embed_description(description)
    orch.task_graph.add_task(node)
    node.status = status
    return node


# ---------------- PATCH 63 ----------------

def test_run21_meal_wording_maps_to_the_meal_part():
    assert _orch()._request_part_for(RUN21_MEAL) == MEAL_PART[0].lower() + MEAL_PART[1:-1]


def test_no_clear_part_means_no_match():
    assert _orch()._request_part_for("Write a short summary of the trip") is None


def test_a_failed_task_is_reworded_once_to_the_request():
    orch = _orch()
    node = _add(orch, "t-meal", RUN21_MEAL)
    assert orch._anchor_task_to_request(node, "retry") is True
    assert node.description == MEAL_PART
    assert node.original_description == RUN21_MEAL
    assert orch._anchor_task_to_request(node, "retry") is False
    assert orch.colony.verdict_counts["retry_anchored"] == 1


def test_the_root_is_never_reworded():
    orch = _orch()
    assert orch._anchor_task_to_request(orch.task_graph.tasks["root"], "retry") is False


def _root_decomposer(orch):
    orch.colony.register_agent(AgentNode(agent_id="d0", role="decomposer", status="running",
                                         parent_id=None, task=SCHOOL, task_id="root"))
    return "d0"


def test_a_replan_of_an_abandoned_part_starts_from_the_request():
    orch = _orch()
    _add(orch, "t-old", RUN21_MEAL, status=3)
    batch = [({"task": "Compute meals with a 15% buffer margin"}, "Compute meals with a 15% "
              "buffer margin", "t-new"),
             ({"task": "Compute food costs with a buffer"}, "Compute food costs with a buffer",
              "t-new2"),
             ({"task": "Compare bus and train cost"}, "Compare bus and train cost", "t-bus")]
    out = orch._anchor_replanned_parts(_root_decomposer(orch), batch)
    assert out[0][1] == MEAL_PART and out[0][0]["task"] == MEAL_PART
    assert out[1:] == batch[1:]  # one rewording per part; other parts untouched


def test_deeper_tasks_are_never_reworded():
    orch = _orch()
    _add(orch, "t-meals", "Plan the meals", status=1)
    leaf = _add(orch, "t-breakfast", "Calculate the breakfast food cost per day",
                parent_task_id="t-meals")
    assert orch._anchor_task_to_request(leaf, "retry") is False
    orch.colony.register_agent(AgentNode(agent_id="d1", role="decomposer", status="running",
                                         parent_id="d0", task="Plan the meals",
                                         task_id="t-meals"))
    _add(orch, "t-lunch", "Price lunch food per day", status=3, parent_task_id="t-meals")
    batch = [({"task": "Price dinner food per day"}, "Price dinner food per day", "t-dinner")]
    assert orch._anchor_replanned_parts("d1", batch) == batch


def test_rejected_retry_runs_on_the_request_wording():
    class _Reject:
        def decide(self, agent_node, **kwargs):
            return {"verdict": "execute", "reason": "invented 18000", "tier": 3}

    class _Memory:
        def write(self, *a, **k):
            return 1

        def get_success_cache(self, *a, **k):
            return None

    orch = _orch()
    orch.judge, orch.memory_store = _Reject(), _Memory()
    _add(orch, "t-meal", RUN21_MEAL)
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id=None, task=RUN21_MEAL, task_id="t-meal"))
    event = Event(type="completion_request", from_agent="a1")
    event.payload.update({"agent_id": "a1", "parent_id": None, "task_id": "t-meal",
                          "result": "Final answer: 18000 rupees for the meals."})
    orch.handle_completion(event)
    assert orch.respawn_counts["t-meal"] == 1
    assert orch.task_graph.tasks["t-meal"].description == MEAL_PART


# ---------------- dedup bar on the same request part ----------------

def test_replan_of_finished_transport_work_is_linked():
    orch = _orch()
    _add(orch, "t-bus", "Calculate the total transportation cost of bus and train", status=2)
    desc = "Compare the total transportation cost of the shuttle versus train options"
    found = orch._find_duplicate_task(desc, orch._embed_description(desc), "root")
    assert found is not None and found[0] == "t-bus" and 0.6 <= found[1] < 0.92


def test_different_part_at_the_same_cosine_is_not_linked():
    orch = _orch()
    _add(orch, "t-bus", "Calculate the total transportation cost of the bus", status=2)
    desc = "Calculate the total transportation and meal cost"  # transport + meal: no one part
    assert orch._request_part_for(desc) is None
    assert orch._find_duplicate_task(desc, orch._embed_description(desc), "root") is None


def test_the_lower_bar_is_for_root_replans_only():
    orch = _orch()
    _add(orch, "t-bus", "Calculate the total transportation cost of bus and train", status=2)
    _add(orch, "t-sub", "Plan the trip logistics", status=1)
    desc = "Compare the total transportation cost of the shuttle versus train options"
    assert orch._find_duplicate_task(desc, orch._embed_description(desc), "t-sub") is None


def test_live_tasks_keep_the_high_bar():
    orch = _orch()
    _add(orch, "t-bus", "Calculate the total transportation cost of bus and train", status=1)
    desc = "Compare the total transportation cost of the shuttle versus train options"
    assert orch._find_duplicate_task(desc, orch._embed_description(desc), "root") is None


# ---------------- PATCH 64 ----------------

def _low_score_spec(raw_text):
    return {"raw_text": raw_text, "requirement": [], "domain": "General Discourse > Stub",
            "goal_vector": np.array([1.0, 0.0]), "context_vector": np.array([1.0, 0.0])}


def test_four_part_request_gets_the_floor(capsys):
    spec = Problem_Phaser.__new__(Problem_Phaser).estimate_complexity(_low_score_spec(SCHOOL))
    assert spec["colony_budget"] == 4 * Problem_Phaser.BUDGET_PER_REQUEST_PART
    assert "PATCH 64" in capsys.readouterr().out


def test_floor_never_lowers_and_needs_a_request():
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    base = phaser.estimate_complexity(_low_score_spec(""))["colony_budget"]
    assert base < 4 * Problem_Phaser.BUDGET_PER_REQUEST_PART  # no parts, no floor
    one_part = phaser.estimate_complexity(_low_score_spec("Plan a picnic."))["colony_budget"]
    assert one_part == max(base, Problem_Phaser.BUDGET_PER_REQUEST_PART)
