"""PATCH 71: labelled arithmetic instead of a blessing, one re-critique when a
reject's arithmetic ground is checkably false, and the agent's tool output
shown to the critique."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_patches_63_64 import SCHOOL, _orch  # noqa: E402

from judge import Judge  # noqa: E402
from text_utils import request_figure_labels  # noqa: E402

GARDEN = ("A community garden has 12 plots and 20 members who want one, decide how to "
          "allocate plots fairly, set a yearly fee so that 9,000 rupees of costs are "
          "covered, split watering duty across the summer months, and decide what to do "
          "with members who stop tending their plot.")
WRONG = "The train costs 45 x 900 = 40,500 rupees, Rs 500 more than the bus at 40,000."
RIGHT = "The train would cost 900 x 50 = 45,000, so the bus at 40,000 is cheaper."


# ---------------- (a) labels ----------------

def test_labels_for_the_school_and_garden_requests():
    school = request_figure_labels(SCHOOL)
    assert school[45.0] == "45 students" and school[5.0] == "5 teachers"
    assert school[900.0] == "900 rupees per person"
    assert school[40000.0] == "40,000 rupees in total"
    garden = request_figure_labels(GARDEN)
    assert garden == {12.0: "12 plots", 20.0: "20 members", 9000.0: "9,000 rupees of costs"}


def test_derivations_are_labelled_from_the_request():
    orch = _orch()
    orch.spec["supplies_data"] = True
    # run 24: P62 passed the 500 chain; the 40,500 step was never shown
    assert orch._ungrounded_figures(WRONG, "t-none") == ["500 = 40500 - 40000"]
    derivations = orch._labelled_derivations(request_figure_labels(SCHOOL), WRONG)
    assert "40,500 = 45 [students] x 900 [rupees per person]" in derivations


def test_labelled_arithmetic_shows_beside_the_p62_derivations():
    _, prompts = _decide([" accept\nok"], ungrounded_figures=["500 = 40500 - 40000"],
                         derivations_checked=["40,500 = 45 [students] x 900 [rupees per person]"])
    assert "DERIVATIONS TO CHECK: 500 = 40500 - 40000" in prompts[0]
    assert "45 [students] x 900 [rupees per person]" in prompts[0]


class _Agent:
    role = "executor"
    warning_count = 0
    fail_reason = ""
    task = "Choose between the bus and the train."


def _decide(replies, **kwargs):
    prompts = []
    replies = list(replies)
    j = Judge(llm_call_fn=lambda p: prompts.append(p) or replies.pop(0))
    j.semantic_check = lambda a, b: 0.9
    v = j.decide(_Agent(), "out", "text", output_embedding=np.ones(4),
                 target_embedding=np.ones(4), needs_deep_check=True, **kwargs)
    return v, prompts


def test_prompt_shows_labelled_arithmetic_and_no_blessing():
    _, prompts = _decide([" accept\nok"], figures_follow=True,
                         derivations_checked=["40,500 = 45 [students] x 900 [rupees per person]"],
                         request_labels=["45 students", "5 teachers"])
    assert "ARITHMETIC CHECKED BY CODE (the OUTPUT's own equations): 40,500 = 45 [students] x 900" in prompts[0]
    assert "The request states: 45 students, 5 teachers." in prompts[0]
    assert "Do not reject for arithmetic alone" not in prompts[0]


# ---------------- (b) unfounded critique grounds ----------------

REP45 = "The train costs 900 x 45 = 40,500 rupees."


def test_unfounded_grounds_are_found():
    orch = _orch()
    assert "46" in orch._critique_unfounded("Train costs 900*46=41400.", RIGHT)
    assert "45,000" in orch._critique_unfounded("Train costs 900*50=41400.", RIGHT)


def test_symmetric_dispute_and_plain_rejects_are_left_alone():
    orch = _orch()
    assert orch._critique_unfounded(
        "It ignores the teachers: 45 + 5 = 50 people, so 900 * 50 = 45,000.", REP45) is None
    assert orch._critique_unfounded(
        "It ignores the 5 teachers; train is 900 * 50 = 45,000.", REP45) is None
    assert orch._critique_unfounded("should be 900 * 45 = 40,500", RIGHT) is None
    assert orch._critique_unfounded("It omits the remaining budget.", RIGHT) is None


def test_quoting_the_reports_equation_is_a_dispute_not_a_note():
    orch = _orch()
    # review: case (3) made this a note and reintroduced the false accept
    assert orch._critique_unfounded(
        "It ignores the teachers; 45 x 900 = 40,500 counts only the students.", WRONG) is None
    assert orch._critique_unfounded(
        "It should compute 900 * 50 = 45,000 first.", RIGHT) is None
    # run 23's critique against the wrong REPORT
    assert orch._critique_unfounded(
        'The agent miscomputed the train price. It says "train costs 45000" when the '
        "correct calculation should be 900 * 45 = 40,500.",
        "The train costs 45000 RUPEES while the bus is fixed at 40000 RUPEES.") is None
    # one step from grounded figures is grounded
    assert orch._critique_unfounded("Savings are 45,000 - 40,000 = 5,000", WRONG) is None


def test_percent_ranges_and_bad_input():
    orch = _orch()
    assert orch._critique_unfounded("Savings 5,000 / 45,000 = 11%", RIGHT) is None
    assert orch._critique_unfounded("between 10-15 = 5 people", RIGHT) is None
    assert orch._critique_unfounded("Day 1-2 = 1 night", RIGHT) is None
    assert orch._critique_unfounded("50 / 0 = 50", RIGHT) is None
    assert orch._eval_claim("9**9**9") == []
    assert orch._eval_claim("45+5 x 900") == [4545.0, 45000.0]


def test_critique_check_that_raises_is_no_note():
    def boom(reason):
        raise ZeroDivisionError
    v, prompts = _decide([" reject\nbad"], critique_check=boom)
    assert len(prompts) == 1 and v["verdict"] == "execute"


def test_correct_report_gets_its_own_working_labelled():
    orch = _orch()
    orch.spec["supplies_data"] = True
    assert orch._ungrounded_figures(RIGHT, "t-none") == []
    derivations = orch._labelled_derivations(request_figure_labels(SCHOOL), RIGHT)
    assert derivations == ["45,000 = 900 [rupees per person] x 50"]
    _, prompts = _decide([" accept\nok"], figures_follow=True, derivations_checked=derivations)
    assert "3 [trip] x 15,000" not in prompts[0]
    assert "it is not a ruling on the OUTPUT" in prompts[0]


def test_recheck_is_debited_and_the_ledger_reconciles():
    import test_patches_68_69 as p69
    replies = [" reject\nTrain costs 900*46=41400" + " pad" * 60, " accept\nfine"]

    def judge(_reason):
        j = Judge(llm_call_fn=lambda p: replies.pop(0))
        j.semantic_check = lambda a, b: 0.9
        return j
    saved, p69._Reject = p69._Reject, judge
    try:
        orch = p69._complete(RIGHT, "")
    finally:
        p69._Reject = saved
    colony = orch.colony
    assert colony.verdict_counts["tier3_arith_recheck_accept"] == 1
    assert colony.energy_ledger["tier3_recheck"] == 2  # ~260 chars of first reason
    assert sum(colony.energy_ledger.values()) == (colony.starting_budget
                                                  - colony.budget_remaining)


def test_decide_recritiques_once_with_the_note():
    v, prompts = _decide([" reject\n900*46=41400", " accept\nfine"],
                         critique_check=lambda reason: "46 is not in the request")
    assert len(prompts) == 2 and "A code check found: 46 is not in the request." in prompts[1]
    assert v["verdict"] == "promote"
    assert v["arith_recheck"]["note"] == "46 is not in the request"
    assert v["arith_recheck"]["verdict"] == "accept"


def test_second_reject_is_final_and_accepts_are_never_rechecked():
    calls = []
    v, prompts = _decide([" reject\na", " reject\nb"],
                         critique_check=lambda r: calls.append(r) or "note")
    assert len(prompts) == 2 and len(calls) == 1 and v["verdict"] == "execute"
    v, prompts = _decide([" accept\nok"], critique_check=lambda r: calls.append(r) or "note")
    assert len(prompts) == 1 and len(calls) == 1 and "arith_recheck" not in v
    v, prompts = _decide([" reject\nbad"], critique_check=lambda r: None)
    assert len(prompts) == 1 and "arith_recheck" not in v


# ---------------- (c) tool output ----------------

def test_tool_output_line_appears_when_given():
    _, prompts = _decide([" accept\nok"], tool_output="Train cost: 45000")
    assert ("CODE THE AGENT RAN PRINTED: Train cost: 45000 (computed; inputs not checked)"
            in prompts[0])
    _, prompts = _decide([" accept\nok"])
    assert "CODE THE AGENT RAN PRINTED" not in prompts[0]


def test_tool_output_is_tagless_capped_and_mismatch_printed(capsys):
    class _Node:
        last_tool_result = "[TOOL RESULT - run_code]: Train cost: 45000" + " pad" * 100
    out = _orch()._tool_output_for_critique(_Node(), WRONG, "a1", "t-bus")
    assert out.startswith("Train cost: 45000") and len(out) <= 203
    assert "[tool-mismatch] a1/t-bus" in capsys.readouterr().out
    _orch()._tool_output_for_critique(_Node(), RIGHT, "a1", "t-bus")
    assert "[tool-mismatch]" not in capsys.readouterr().out


# ---------------- integration review ----------------

def test_percent_already_times_100_is_not_a_false_note():
    orch = _orch()
    assert orch._critique_unfounded("45,000 / 60,000 x 100 = 75%", RIGHT) is None
    note = orch._critique_unfounded("45,000 / 60,000 x 100 = 80%", RIGHT)
    assert note and "= 75%" in note and "7,500" not in note


def test_common_constants_are_not_ungrounded():
    orch = _orch()
    assert orch._critique_unfounded("Per hour 45,000 / 24 = 1,875", RIGHT) is None
    assert orch._critique_unfounded("40,000 / 45,000 x 100 = 88.9%", RIGHT) is None


def test_currency_prefixed_results_are_claims():
    orch = _orch()
    for text in ("900 x 46 = $ 41,400", "900 x 46 = ₹ 41,400", "900 x 46 = Rs. 41,400",
                 "900 x 46 = INR 41,400"):
        assert [c for _, c, _, _, _ in orch._arith_claims(text)] == [41400.0], text


def test_traceback_is_not_shown_as_printed_output():
    class _Node:
        last_tool_result = ("[TOOL RESULT - run_code]: Traceback (most recent call last):\n"
                            "ZeroDivisionError: division by zero 45000")
    assert _orch()._tool_output_for_critique(_Node(), RIGHT, "a1", "t") is None

