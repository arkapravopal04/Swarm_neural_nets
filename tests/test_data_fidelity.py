"""
PATCH 22 (phaser data fidelity) and PATCH 25 (LaTeX is not code).

Run 8's request said Rs. 9,000, 12 plots, 20 applicants and summer months,
and asked four things. The phaser's goal said 9,567, lost the 12 and the
fourth ask, and said "six months"; a phaser constraint said Rs. 9,600, and
the root decomposer copied it into its first SPAWN.

RUN8_REQUEST below is a RECONSTRUCTION -- the log never printed the request.
Its figures and its four parts are the real ones; the wording is not.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import sys
import types

import numpy as np
import pytest

if importlib.util.find_spec("sentence_transformers") is None:
    _stub = types.ModuleType("sentence_transformers")
    _stub.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    _stub.SentenceTransformer = object
    sys.modules.setdefault("sentence_transformers", _stub)

from colony_state import ColonyState  # noqa: E402
from event_queue import Messenger  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from problem_phaser import Problem_Phaser  # noqa: E402
from task_graph import TaskGraph  # noqa: E402
from text_utils import (  # noqa: E402
    ask_coverage, figure_fidelity, figure_values, goal_fidelity,
    looks_like_source_code, repair_figures, request_asks, software_artifact_reason,
    strip_math,
)


RUN8_REQUEST = (
    "Our community garden has 12 plots and 20 people have applied for them. "
    "Running costs are Rs. 9,000 a year. "
    "How should we decide who gets a plot fairly? "
    "What annual fee should each plot holder pay so the costs are covered? "
    "How should watering duties be shared over the summer months? "
    "What should happen to members who stop tending their plots?")
RUN8_GOAL = ("Allocate plots fairly among 20 applicants while covering 9,567 rupee "
             "operational expenses through annual fees and splitting seasonal upkeep "
             "responsibilities evenly over six months.")
FAITHFUL_GOAL = ("Decide how to allocate the 12 plots fairly among 20 applicants, set an "
                 "annual fee that covers Rs. 9,000 costs, share watering duties over the "
                 "summer months, and decide what happens to members who stop tending plots.")


# --- figures ----------------------------------------------------------------------

def test_figures_compare_by_value_and_number_words_count():
    assert set(figure_values("Rs. 9,000 and 9000 and 9K")) == {9000.0}
    assert set(figure_values("six months, twenty-five members, a dozen eggs")) == {6.0, 25.0, 12.0}
    # "one" is not a quantity in prose.
    assert figure_values("each one of the plots") == {}


def test_run_8_goal_fails_on_figures_both_ways():
    extra, missing = figure_fidelity(RUN8_REQUEST, RUN8_GOAL)
    assert extra == ["9567", "six"]
    assert missing == ["12", "9000"]


def test_run_8_constraints_state_figures_the_request_does_not():
    assert figure_fidelity(RUN8_REQUEST, "Yearly fees must cover Rs. 9,600 total expenses.",
                           require_all=False) == (["9600"], [])
    assert figure_fidelity(RUN8_REQUEST, "Watering duties distributed evenly over six months.",
                           require_all=False) == (["six"], [])
    # A constraint restating the user's figure in words passes.
    assert figure_fidelity("Plan a dinner for two guests.", "Must serve 2 guests.",
                           require_all=False) == ([], [])


# --- parts --------------------------------------------------------------------------

def test_asks_are_questions_list_items_and_requests():
    asks = request_asks(RUN8_REQUEST)
    assert len(asks) == 4 and all(a.endswith("?") for a in asks)
    assert request_asks("Context line.\n1) choose a trail\n2) set a date") == [
        "choose a trail", "set a date"]
    assert request_asks("Decide who hosts. The club meets monthly.") == ["Decide who hosts."]
    assert request_asks("We have 12 plots.") == []


def test_the_dropped_fourth_part_is_found():
    coverage = {ask: covered for ask, _, covered in ask_coverage(request_asks(RUN8_REQUEST), RUN8_GOAL)}
    assert coverage["What should happen to members who stop tending their plots?"] is False
    assert coverage["How should we decide who gets a plot fairly?"] is True


def test_a_faithful_goal_passes_everything():
    assert goal_fidelity(RUN8_REQUEST, FAITHFUL_GOAL)["ok"] is True
    assert goal_fidelity(RUN8_REQUEST, RUN8_GOAL)["ok"] is False


def test_asks_naming_nothing_are_not_checked():
    assert ask_coverage(["How should we do this?"], "Anything at all.") == []


# --- the phaser: regenerate once, then the user's own text --------------------------

def _phaser():
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    return phaser


def _draws(*goals):
    it = iter(goals)
    return lambda: next(it)


def test_first_faithful_draw_is_kept():
    goal, record = _phaser()._faithful_goal(_draws(FAITHFUL_GOAL), RUN8_REQUEST)
    assert goal == FAITHFUL_GOAL and record["outcome"] == "pass"


def test_second_draw_is_used_when_the_first_fails():
    goal, record = _phaser()._faithful_goal(_draws(RUN8_GOAL, FAITHFUL_GOAL), RUN8_REQUEST)
    assert goal == FAITHFUL_GOAL and record["outcome"] == "pass_on_retry"
    assert record["attempts"][0]["extra"] == ["9567", "six"]


def test_two_failures_fall_back_to_the_request_uncapped():
    long_request = RUN8_REQUEST + " " + ("Some background about the garden. " * 12)
    goal, record = _phaser()._faithful_goal(_draws(RUN8_GOAL, RUN8_GOAL), long_request)
    assert record["outcome"] == "fallback_raw_text"
    assert goal == " ".join(long_request.split())
    assert len(goal) > 400           # dedupe_global_and_cap's cap is not applied


def test_constraints_resampled_once_then_offenders_dropped():
    greedy = ["Yearly fees must cover Rs. 9,600 total expenses.",
              "Watering duties distributed evenly over six months.",
              "Plots must go to applicants fairly."]
    sampled = ["Yearly fees must cover Rs. 9,000 total expenses.",
               "Watering duties distributed evenly over six months."]
    calls = []

    def extract(is_sampled):
        calls.append(is_sampled)
        return sampled if is_sampled else greedy

    kept, record = _phaser()._faithful_constraints(extract, RUN8_REQUEST)
    assert calls == [False, True]            # greedy first, ONE sampled retry
    assert kept == ["Yearly fees must cover Rs. 9,000 total expenses."]
    assert record["dropped"] == [{"constraint": sampled[1], "extra": ["six"]}]


def test_clean_constraints_are_extracted_once():
    calls = []

    def extract(is_sampled):
        calls.append(is_sampled)
        return ["Plots must go to applicants fairly."]

    kept, record = _phaser()._faithful_constraints(extract, RUN8_REQUEST)
    assert calls == [False] and record == {"dropped": [], "repaired": [],
                                           "retried": False, "extracted": 1}


# --- PATCH 30 -- repair, don't drop --------------------------------------------------

RUN10_REQUEST = (
    "A community garden has 12 plots and 20 members who want one, decide how to "
    "allocate plots fairly, set a yearly fee so that 9,000 rupees of costs are "
    "covered, split watering duty across the summer months, and decide what to do "
    "with members who stop tending their plot.")
RUN10_FEE = "Fee structure must cover at least Rs9,067 total annual operational expenses."


def test_currency_glued_figures_read_whole():
    # Run 10 logged the 9,067 as '067'; "Rs.9,000" read as '000' -- a faithful
    # constraint written that way would have been dropped as invented.
    assert figure_values(RUN10_FEE) == {9067.0: "9067"}
    assert figure_values("Fees must cover Rs.9,000") == {9000.0: "9000"}
    assert figure_values("INR9000 a year") == {9000.0: "9000"}
    assert figure_fidelity(RUN10_REQUEST, "Fees must cover Rs.9,000", require_all=False)[0] == []


def test_run10_fee_constraint_is_repaired_not_dropped():
    def extract(is_sampled):
        return ["Exactly 12 watered plots per month during the growing season.",
                RUN10_FEE,
                "Members abandoning care will forfeit their allocated spot immediately."]

    kept, record = _phaser()._faithful_constraints(extract, RUN10_REQUEST)
    assert kept[1] == "Fee structure must cover at least Rs 9,000 total annual operational expenses."
    assert len(kept) == 3 and record["dropped"] == []
    assert record["repaired"][0]["repairs"] == [("9,067", "9,000")]
    assert record["extracted"] == 3


def test_repair_needs_one_unambiguous_request_figure():
    # Small counts are not repaired: 11 is not a mis-copied 12 plots or 20 members.
    assert repair_figures(RUN10_REQUEST, "At most 11 members per rota")[0] is None
    # Too far from 9,000 to be a mis-copy of it.
    assert repair_figures(RUN10_REQUEST, "Cover 12,500 rupees")[0] is None
    # Two request figures in range is ambiguous.
    assert repair_figures("Budgets of 9,000 and 9,500.", "Spend 9,200")[0] is None
    # Run 8's constraint figure maps onto 9,000 as well.
    repaired, repairs, _ = repair_figures(RUN10_REQUEST, "Yearly fees must cover Rs. 9,600 total expenses")
    assert repaired == "Yearly fees must cover Rs. 9,000 total expenses"


def test_unrepairable_offender_is_still_dropped():
    def extract(is_sampled):
        return ["Watering over 6 months.", RUN10_FEE]

    kept, record = _phaser()._faithful_constraints(extract, RUN10_REQUEST)
    assert kept == ["Fee structure must cover at least Rs 9,000 total annual operational expenses."]
    assert record["dropped"] == [{"constraint": "Watering over 6 months.", "extra": ["6"]}]


# --- spawn-time check and the ledger (measure-only) ---------------------------------

def _orch(spec):
    orch = Orchestrator(ColonyState(initial_budget=10000, goal_embedding=np.array([1.0, 0.0])),
                        TaskGraph(), Messenger(), embed_model=None)
    orch.spec = spec
    return orch


def test_the_root_decomposers_9600_is_flagged_at_spawn():
    orch = _orch({"supplies_data": True, "raw_text": RUN8_REQUEST})
    orch._flag_unsupplied_figures(
        "task_716fab30", "Allocate Rs. 9,600 total annual fees to the selected 12 plots")
    orch._flag_unsupplied_figures(
        "task_01ac5781", "Determine a fair allocation method for the 12 plots among 20 applicants")
    assert [(t, extra) for t, extra, _ in orch.unsupplied_figure_subtasks] == [
        ("task_716fab30", ["9600"])]
    assert orch.spawned_subtask_count == 2


def test_nothing_is_flagged_on_a_data_free_problem():
    orch = _orch({"supplies_data": False, "raw_text": "How should our club pick books?"})
    orch._flag_unsupplied_figures("t", "Count 9,761 votes")
    assert orch.unsupplied_figure_subtasks == []


def test_ledger_section():
    orch = _orch({
        "supplies_data": True, "raw_text": RUN8_REQUEST,
        "goal_fidelity": {"outcome": "fallback_raw_text", "attempts": [
            {"goal": RUN8_GOAL, "extra": ["9567", "six"], "missing": ["12", "9000"],
             "uncovered": ["What should happen to members who stop tending their plots?"]}]},
        "constraint_fidelity": {"retried": True, "dropped": [
            {"constraint": "Watering duties distributed evenly over six months.", "extra": ["six"]}]},
    })
    orch._flag_unsupplied_figures("task_716fab30", "Allocate Rs. 9,600 total annual fees")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch._print_fidelity_report()
    out = buf.getvalue()
    assert "goal : fallback_raw_text  (1 draw(s))" in out
    assert "figures not in request ['9567', 'six']" in out
    assert "missing part: 'What should happen to members" in out
    assert "constraints : 1 dropped for figures not in the request (after one re-extraction)" in out
    assert "stating figures the request does not : 1 of 1" in out


# --- PATCH 25 ------------------------------------------------------------------------

# Both of run 8's software-framing rejects, verbatim.
RUN8_LATEX = (
    "The average monthly water usage per applicant is $ \\frac{9567}{12} = 832.25 $. "
    "Since this is spread across two months, each applicant uses "
    "$ \\frac{832.25}{2} = 416.13 $. Round up to 417 units per applicant per month. "
    "Total cost of 20*417=Rs. 8340 which fits within the \\$9600 limit.")
RUN8_CODE = (
    "charge_per_plot, watering_cost = round(9600/12/20), round((3574*12)/12/20) "
    "=> (480, 224) Ruppees each plot/month. Total revenue = 480*20=9600")


def test_latex_arithmetic_is_not_software():
    assert software_artifact_reason(RUN8_LATEX) is None
    assert not looks_like_source_code(RUN8_LATEX)


def test_the_real_code_reject_is_still_caught():
    assert software_artifact_reason(RUN8_CODE) == "is written as code"


@pytest.mark.parametrize("text, removed", [
    ("x is $ \\frac{a}{b} $ here", 1),
    ("inline \\( a \\times b \\) and display \\[ \\sqrt{2} \\]", 2),
    ("bare \\frac{9567}{12} and 3 \\times 4", 2),
    # Currency: a $...$ span with no LaTeX command in it is prose.
    ("covers $9,600 total, six payments of $1,600", 0),
    # An escaped dollar is a literal, never a delimiter.
    ("within the \\$9600 limit, \\$260 left", 0),
])
def test_strip_math(text, removed):
    assert strip_math(text)[1] == removed


def test_code_between_two_prices_is_still_code():
    text = "Charge $5 then total = price * 12; return total if x == 1 else $0"
    assert software_artifact_reason(text) == "is written as code"
