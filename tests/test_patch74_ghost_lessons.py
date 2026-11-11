"""PATCH 74 -- a tier-3 reject's own unverified equations stay out of the
ghost lesson and the respawn hint; tier-3 ghosts are labelled TIER_3_REJECT."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ghost_extractor  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402


def _drop(reason, report):
    return Orchestrator._drop_unverified_judge_equations(
        Orchestrator.__new__(Orchestrator), reason, report)


REPORT = "Bus: 900 people / 50 per bus = 18 buses. Train: 900 x 50 = $45,000."


def test_equation_not_in_report_is_dropped_other_sentence_kept():
    lesson, dropped = _drop(
        "Train costs 900*46=41400, which is more than the bus. The reasoning is vague.",
        REPORT)
    assert dropped == 1
    assert lesson == "The reasoning is vague."
    assert "41400" not in lesson


def test_equation_whose_result_is_in_the_report_is_kept():
    reason = "Train costs 900*50=45,000, which is too much. The reasoning is vague."
    lesson, dropped = _drop(reason, REPORT)
    assert dropped == 0
    assert lesson == reason


def test_all_sentences_dropped_yields_the_generic_line():
    lesson, dropped = _drop(
        "Train costs 900*46=41400. It should be approximately 900*45=40,500.", REPORT)
    assert dropped == 2
    assert lesson == "the reviewer disputed the arithmetic."


def test_currency_abbreviation_does_not_split_the_equation():
    lesson, dropped = _drop("The train is 900 x 45 = Rs. 40,500, not enough.",
                            "The train costs 45,000 rupees.")
    assert dropped == 1 and "40,500" not in lesson


def _dead(warnings=0):
    return SimpleNamespace(agent_id="a1", role="executor", task="cost of the trip",
                           warning_count=warnings, fail_reason=None, energy_spent=5)


def test_tier3_verdict_is_labelled_tier3_reject():
    rec = ghost_extractor.extract(_dead(), {"verdict": "execute", "tier": 3,
                                            "reason": "The reasoning is vague."})
    assert rec["failure_type"] == ghost_extractor.TIER_3_REJECT == "TIER_3_REJECT"


def test_tier1_and_untiered_labels_unchanged():
    assert ghost_extractor.extract(
        _dead(), {"verdict": "execute", "tier": 1, "reason": "empty"}
    )["failure_type"] == "TIER_1_CRASH"
    assert ghost_extractor.extract(
        _dead(), {"verdict": "execute", "reason": "echo"})["failure_type"] == "TIER_1_CRASH"
    assert ghost_extractor.extract(
        _dead(3), {"verdict": "execute", "tier": 2, "reason": "drift"}
    )["failure_type"] == "SEMANTIC_DRIFT"
    assert ghost_extractor.extract(_dead(), None)["failure_type"] == "SELF_REPORTED"
