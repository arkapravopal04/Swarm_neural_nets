"""
PATCH 21 (measure-only) -- a data-free problem, and the promoted REPORTs that
assert specific figures or identifiers anyway.

The replays are asserted: run 7's three hits and run 5's two, from the texts
that were actually promoted.
"""
import io
import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from colony_state import ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph
from text_utils import asserted_figures, asserted_identifiers, supplied_data

from test_artifact_and_figures import CE16FD80_3
from test_goal_drift_gate import RUN5_ROOT_REPORT


RUN7_ASSIGNMENT = ("{'assignment': {1004879: ['R1', 'R2'], 9874021: ['R3'], "
                   "2483902: ['R4']}, 'undecided': [2483902]}")


# --- the input side ---------------------------------------------------------------

@pytest.mark.parametrize("request_text", [
    "Our book club argues every month about what to read. Some members never "
    "finish. How should we choose books, assign readings, and handle picks "
    "people dislike?",
    # List markers are structure, not data (PATCH 16's exclusion).
    "Three questions, in order: 1) how to choose 2) how to assign 3) how to "
    "handle dislikes.",
    # Number WORDS are not numerals. A REPORT saying "5 members" on this
    # input is derived, which is why the ledger shows each hit's figures
    # instead of calling them fabricated.
    "Our five members vote on books each month.",
])
def test_data_free_requests(request_text):
    assert supplied_data(request_text) == ([], [])


@pytest.mark.parametrize("request_text, expected", [
    ("We have 12 members and 40 candidate books.", (["12", "40"], [])),
    ("Book B7 got 9K votes.", (["9K"], ["B7"])),
    ("Budget is $300 and 15% goes to snacks.", (["300", "15%"], [])),
])
def test_requests_that_supply_data(request_text, expected):
    assert supplied_data(request_text) == expected


# --- the answer side --------------------------------------------------------------

def test_run_7s_assignment_is_all_figures_and_ids():
    assert asserted_figures(RUN7_ASSIGNMENT) == ["1004879", "9874021", "2483902"]
    assert asserted_identifiers(RUN7_ASSIGNMENT) == ["R1", "R2", "R3", "R4"]


def test_run_5s_promoted_reports_both_assert_figures():
    assert asserted_figures(CE16FD80_3)[:4] == ["9761", "8847", "6951", "226"]
    # figures() rejects "9K" (PATCH 16 compares bare quantities); this does not.
    assert "9K" in asserted_figures(RUN5_ROOT_REPORT)
    assert "6.9K" in asserted_figures(RUN5_ROOT_REPORT)


@pytest.mark.parametrize("text", [
    "Pick the 1st and 3rd options.",          # ordinals are positions
    "Use version v2.1 of the rules.",         # a version, not an ID
    "Steps: 1) choose 2) assign",             # list markers
])
def test_not_counted(text):
    assert asserted_figures(text) == [] and asserted_identifiers(text) == []


# --- the ledger ---------------------------------------------------------------

def _orch(spec, tasks):
    orch = Orchestrator(
        ColonyState(initial_budget=10000, goal_embedding=np.array([1.0, 0.0])),
        TaskGraph(), Messenger(), embed_model=None,
    )
    orch.spec = spec
    orch.task_graph.tasks = {
        tid: SimpleNamespace(status=status, result=result)
        for tid, (status, result) in tasks.items()}
    return orch


def _ledger(orch):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orch._print_data_free_report()
    return buf.getvalue()


def test_ledger_counts_promoted_reports_on_a_data_free_problem():
    orch = _orch({"supplies_data": False}, {
        "t_ids": (2, RUN7_ASSIGNMENT),
        "t_prose": (2, "Choose by majority vote and rotate the picker."),
        "t_failed": (3, "Book 4412 wins."),          # not promoted
        "root_task_0": (2, "Books were chosen by vote."),
    })
    out = _ledger(orch)
    assert "problem supplies data : no  (phaser)" in out
    assert "asserting figures or IDs anyway : 1 of 3" in out
    assert "t_ids  ['1004879', '9874021', '2483902', 'R1', 'R2', 'R3', 'R4']" in out
    assert "t_failed" not in out


def test_ledger_says_yes_and_stops_when_the_problem_supplies_data():
    orch = _orch({"supplies_data": True}, {"t": (2, RUN7_ASSIGNMENT)})
    out = _ledger(orch)
    assert "problem supplies data : yes" in out
    assert "anyway" not in out


def test_ledger_recomputes_from_raw_text_on_an_older_spec():
    orch = _orch({"raw_text": "How should our club pick books?"},
                 {"t": (2, "Book A 12 votes.")})
    out = _ledger(orch)
    assert "no  (raw_text, recomputed)" in out
    assert "1 of 1" in out
