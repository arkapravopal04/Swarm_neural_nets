"""
PATCH 23 (measure-only) -- grounded / derivable / near-miss / neither.

Synthetic fixtures only. The run 8 and run 10 logs are lost; the 478.35 /
9567 case below stands in for run 8's.
"""
import io
import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from colony_state import ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode
from text_utils import classify_figures, figure_values


R = ("A community garden has 12 plots and 20 members who want one, decide "
     "how to allocate plots fairly, set a yearly fee so that 9,000 rupees of "
     "costs are covered, split watering duty across the summer months, and "
     "decide what to do with members who stop tending their plot.")

DERIVABLE_ON_R = [8, 32, 240, 0.6, 1.667, 450, 750, 8980, 8988, 9012, 9020,
                  108000, 180000, 0.00133, 0.00222]


def _one(text, request=R, **kw):
    entries = classify_figures(request, text, **kw)
    return [(e["value"], e["class"], e["detail"]) for e in entries]


# --- the classifier ---------------------------------------------------------

def test_request_values():
    assert set(figure_values(R)) == {12.0, 20.0, 9000.0}


def test_exactly_fifteen_derivable_values_on_r():
    # Every one-op result of two distinct request values, negatives dropped.
    req = [12.0, 20.0, 9000.0]
    results = set()
    for a in req:
        for b in req:
            if a == b:
                continue
            for r in (a + b, a - b, a * b, a / b):
                if r >= 0:
                    results.add(round(r, 5))
    assert len(results) == 15
    for value in DERIVABLE_ON_R:
        assert _one(str(value))[0][1] == "derivable", value


def test_grounded_counts():
    assert _one("20 members, 12 plots") == [(20.0, "grounded", ""),
                                            (12.0, "grounded", "")]
    assert _one("costs of 9k") == [(9000.0, "grounded", "")]


def test_derivable_with_detail():
    assert _one("Rs 750 per plot") == [(750.0, "derivable", "750 = 9000 / 12")]
    assert _one("8 members wait-listed") == [(8.0, "derivable", "8 = 20 - 12")]


def test_two_steps_is_neither():
    assert _one("62.5") == [(62.5, "neither", "")]


def test_unrelated_figures_are_neither_not_near_miss():
    assert _one("Rs 960 per plot, 80 per month") == [(960.0, "neither", ""),
                                                     (80.0, "neither", "")]


@pytest.mark.parametrize("text, detail", [
    ("Rs 9,567", "near-miss of 9000 (+6.3%)"),
    ("Rs 9,600", "near-miss of 9000 (+6.7%)"),
    ("Rs9,067", "near-miss of 9000 (+0.7%)"),
])
def test_runs_8_9_10_totals_are_near_misses(text, detail):
    [(_, cls, got)] = _one(text)
    assert (cls, got) == ("near-miss", detail)


def test_trace_annotates_but_stays_neither():
    assert _one("478.35 per member") == [(478.35, "neither", "")]
    assert _one("478.35 per member", near_misses=[9567]) == [
        (478.35, "neither", "traces to 9567 / 20")]


def test_multiplication_sign_is_not_a_figure():
    assert _one("Fee = 12 x plot_cost") == [(12.0, "grounded", "")]


def test_no_self_pairs():
    assert _one("144") == [(144.0, "neither", "")]


def test_half_unit_tolerance_from_ten_up():
    request = "Split 9,000 rupees across 7 plots."
    assert _one("1285.71", request) == [(1285.71, "derivable", "1285.71 = 9000 / 7")]
    assert _one("1286", request) == [(1286.0, "derivable", "1286 = 9000 / 7")]


def test_one_percent_tolerance_below_ten():
    request = "12 plots, 20 members"
    assert _one("0.6", request)[0][1] == "derivable"
    assert _one("1.67", request)[0][1:] == ("derivable", "1.67 = 20 / 12")
    assert _one("0.7", request)[0][1] == "neither"


def test_distinct_values_only():
    assert _one("12 ... 12 ... 12") == [(12.0, "grounded", "")]


def test_entry_shape():
    [entry] = classify_figures(R, "Rs 9,567")
    assert entry == {"value": 9567.0, "written": "9567", "class": "near-miss",
                     "detail": "near-miss of 9000 (+6.3%)"}


# --- the ledger ---------------------------------------------------------------

def _orch(spec, tasks=None):
    orch = Orchestrator(
        ColonyState(initial_budget=10000, goal_embedding=np.array([1.0, 0.0])),
        TaskGraph(), Messenger(), embed_model=None,
    )
    orch.spec = spec
    if tasks is not None:
        orch.task_graph.tasks = {
            tid: SimpleNamespace(status=status, result=result)
            for tid, (status, result) in tasks.items()}
    return orch


def _ledger(orch):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch._print_data_free_report()
    return buf.getvalue()


def test_ledger_supplies_yes_prints_the_p23_block():
    orch = _orch({"supplies_data": True, "raw_text": R}, {
        "t_fee": (2, "Rs 750 per plot, so 9,000 is covered by 12 plots."),
        "t_bad": (2, "Total costs Rs 9,567, so 478.35 per member, 960 per plot."),
        "t_prose": (2, "Rotate watering weekly."),
        "t_failed": (3, "Rs 4412 per plot."),       # not promoted
        "root_task_0": (2, "Fee 750; 8 members wait-listed; 9,567 again."),
    })
    out = _ledger(orch)
    assert "problem supplies data : yes" in out
    assert "promoted REPORTs with figures : 3 of 4" in out
    # Distinct values per REPORT, summed: t_fee {750, 9000, 12}, t_bad
    # {9567, 478.35, 960}, root {750, 8, 9567}.
    assert "figures grounded   : 2" in out
    assert "figures derivable  : 3" in out
    assert "t_fee  750 = 9000 / 12" in out
    assert "root_task_0  8 = 20 - 12" in out
    assert "figures near-miss  : 2" in out
    assert "t_bad  9567  near-miss of 9000 (+6.3%)" in out
    assert "figures neither    : 2   (not counting near-misses)" in out
    assert "t_bad  478.35  traces to 9567 / 20" in out
    assert "t_bad  960" in out
    assert ("grounded [12, 9000]  derivable [8, 750]  near-miss [9567]  "
            "neither [478.35, 960]") in out
    assert "t_failed" not in out and "4412" not in out
    assert "anyway" not in out


def test_ledger_supplies_no_is_the_patch_21_block_unchanged():
    orch = _orch({"supplies_data": False}, {
        "t_ids": (2, "Plot P7 gets 12 hours."),
        "t_prose": (2, "Rotate watering weekly."),
    })
    out = _ledger(orch)
    assert out == (
        "\n  DATA-FREE PROBLEM (PATCH 21, measure-only)\n"
        "    problem supplies data : no  (phaser)\n"
        "    promoted REPORTs asserting figures or IDs anyway : 1 of 2\n"
        "      t_ids  ['12', 'P7']\n")
    assert "PATCH 23" not in out and "grounded" not in out


# --- the final answer ---------------------------------------------------------

def _final(orch, answer):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch._print_final_answer_figures(answer)
    return buf.getvalue()


def test_final_answer_line():
    out = _final(_orch({"supplies_data": True, "raw_text": R}),
                 "Charge Rs 750 per plot for 12 plots; total 9,567; 62.5 spare.")
    assert "FINAL ANSWER figures (PATCH 23):" in out
    assert ("grounded [12]  derivable [750]  near-miss [9567]  "
            "neither [62.5]") in out


def test_final_answer_line_silent_on_a_data_free_problem():
    assert _final(_orch({"supplies_data": False}), "Rs 750") == ""


def test_final_answer_line_on_a_non_string_answer():
    out = _final(_orch({"supplies_data": True, "raw_text": R}), {"x": 1})
    assert "not a text answer" in out


def _terminate_capturing(orch):
    seen = []
    orch._print_final_answer_figures = lambda answer: seen.append(answer)
    with contextlib.redirect_stdout(io.StringIO()):
        returned = orch.terminate()
    return returned, seen


def test_final_answer_line_gets_best_result_on_the_success_path():
    orch = _orch({"supplies_data": True, "raw_text": R})
    orch.root_task_id = "root_task_0"
    orch.task_graph.add_task(TaskNode(task_id="root_task_0", description="plan", status=2))
    orch.colony.results["final_spec"] = "Charge Rs 750 per plot."
    returned, seen = _terminate_capturing(orch)
    assert seen == [returned] and returned == "Charge Rs 750 per plot."


def test_final_answer_line_gets_best_result_on_the_partial_path():
    orch = _orch({"supplies_data": True, "raw_text": R})
    orch.root_task_id = "root_task_0"
    orch.task_graph.add_task(TaskNode(task_id="root_task_0", description="plan", status=3))
    orch.task_graph.tasks["root_task_0"].result = "Charge Rs 9,567 in total."
    returned, seen = _terminate_capturing(orch)
    assert seen == [returned] and "9,567" in returned
