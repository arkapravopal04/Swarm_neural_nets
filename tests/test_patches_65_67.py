"""PATCH 65 (no PARTIAL banner when a later task answered the abandoned
part) and PATCH 67 (the critique is told when every figure checked out),
against run 22."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_patches_63_64 import MEAL_PART, RUN21_MEAL, SCHOOL, _add, _orch  # noqa: E402

from colony_state import AgentNode  # noqa: E402
from event_queue import Event  # noqa: E402
from judge import Judge  # noqa: E402


# ---------------- PATCH 65 ----------------

def test_abandoned_part_answered_later_needs_no_banner():
    orch = _orch()
    _add(orch, "t-old", MEAL_PART, status=3)
    orch.abandoned_tasks.add("t-old")
    _add(orch, "t-new", MEAL_PART, status=2).result = "300 rupees per person per day."
    assert orch._unanswered_abandoned() == []


def test_abandoned_part_left_unanswered_keeps_the_banner():
    orch = _orch()
    _add(orch, "t-old", RUN21_MEAL, status=3)
    orch.abandoned_tasks.add("t-old")
    _add(orch, "t-bus", "Compare bus and train cost", status=2).result = "Take the bus."
    assert orch._unanswered_abandoned() == ["t-old"]


def test_a_completed_leaf_does_not_answer_the_part():
    orch = _orch()
    _add(orch, "t-meals", "Plan the meals for all three days", status=3)
    _add(orch, "t-day1", "Plan day 1 meals under the daily food cap", status=2,
         parent_task_id="t-meals").result = "Day 1: 300 per person."
    orch.abandoned_tasks.add("t-meals")
    assert orch._unanswered_abandoned() == ["t-meals"]


def test_a_failed_check_is_not_reported_as_all_clear(monkeypatch):
    import text_utils
    monkeypatch.setattr(text_utils, "classify_figures", lambda *a, **k: 1 / 0)
    orch = _orch()
    orch.spec["supplies_data"] = True
    _add(orch, "t-meal", MEAL_PART)
    assert orch._ungrounded_figures("300 rupees per person per day.", "t-meal") is None


def test_an_abandoned_root_or_unmatched_task_counts_as_unanswered():
    orch = _orch()
    _add(orch, "t-odd", "Write a short summary of the trip", status=3)
    orch.abandoned_tasks.update({"t-odd", "root"})
    assert sorted(orch._unanswered_abandoned()) == ["root", "t-odd"]


# ---------------- PATCH 67 ----------------

class _Agent:
    role = "executor"
    warning_count = 0
    fail_reason = ""
    task = MEAL_PART


def _critique_prompt(**kwargs):
    captured = []
    j = Judge(llm_call_fn=lambda p: captured.append(p) or " accept\nok")
    j.semantic_check = lambda a, b: 0.9
    j.decide(_Agent(), "out", "text", output_embedding=np.ones(4),
             target_embedding=np.ones(4), needs_deep_check=True, **kwargs)
    return captured[0]


def test_critique_is_told_when_the_figures_check_out():
    assert "FIGURES CHECKED: an automatic check" in _critique_prompt(figures_follow=True)


def test_no_note_without_the_check_or_with_flagged_figures():
    assert "FIGURES CHECKED" not in _critique_prompt()
    prompt = _critique_prompt(figures_follow=True, ungrounded_figures=["18000"])
    assert "FIGURES CHECKED" not in prompt and "FIGURES TO CHECK" in prompt


class _Recording:
    def __init__(self):
        self.kwargs = None

    def decide(self, agent_node, **kwargs):
        self.kwargs = kwargs
        return {"verdict": "promote", "reason": "ok", "tier": 3}


class _Memory:
    def write(self, *a, **k):
        return 1

    def get_success_cache(self, *a, **k):
        return None


def _complete(result):
    orch = _orch()
    orch.spec["supplies_data"] = True
    orch.judge, orch.memory_store = _Recording(), _Memory()
    _add(orch, "t-meal", MEAL_PART)
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id=None, task=MEAL_PART, task_id="t-meal"))
    event = Event(type="completion_request", from_agent="a1")
    event.payload.update({"agent_id": "a1", "parent_id": None, "task_id": "t-meal",
                          "result": result})
    orch.handle_completion(event)
    return orch.judge.kwargs


def test_run22_correct_meal_answer_reaches_the_critique_marked_as_checked():
    kwargs = _complete("15,000 rupees per day divided by 50 people equals about 300 rupees "
                       "per person per day.")
    assert kwargs["ungrounded_figures"] == [] and kwargs["figures_follow"] is True


def test_invented_figures_are_not_marked_as_checked():
    kwargs = _complete("Each meal costs 120 rupees, so 150 meals cost 18,000 rupees.")
    assert kwargs["ungrounded_figures"] and kwargs["figures_follow"] is False
