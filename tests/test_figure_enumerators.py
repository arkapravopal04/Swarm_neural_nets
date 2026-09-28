"""
PATCH 16's enumerator exclusion, added after run 6.

Run 6's largest figure divergence was task_6cc287e9 at 62.1x:
['181.7', '186.3'] against ['1', '2', '3'], where the second attempt read
"three tiers of resolution logic: 1) tiebreaker ... 2) fallback ... 3)
conflict arbitration". Those are list markers. The detector was reading
prose structure as data.

The negative cases matter as much: PATCH 16 exists to catch a fabricated
quantity changing between attempts, and every figure excluded here is one it
can no longer see. Run 5's motivating chain ("Book X 35, Book Y 47, ...") is
asserted whole.
"""
import pytest

from text_utils import figures, figure_divergence


# The exact text of run 6's second attempt at task_6cc287e9.
RUN_6_ENUMERATED = (
    "The voting system needs three tiers of resolution logic: 1) tiebreaker "
    "score threshold where votes are split evenly among top contenders, 2) "
    "fallback consensus rule using majority vote after eliminating "
    "underperforming candidates, and 3) conflict arbitration where tied "
    "votes are resolved via a random draw with a fixed probability weight "
    "for the most popular candidate."
)


def test_run_6_inline_enumeration_asserts_no_figures():
    """After a colon, after a comma, and after "and" -- one clause-start rule
    tight enough to be safe for "N." would have caught only the first."""
    assert figures(RUN_6_ENUMERATED) == []


def test_the_62x_catch_is_silenced():
    previous = ("The chosen book receives 186.3 points against 181.7 for the "
                "runner-up.")
    assert figure_divergence(previous, RUN_6_ENUMERATED) is None


@pytest.mark.parametrize("text", [
    "1. Choose books\n2. Assign readings\n3. Resolve conflicts",
    "Steps: 1) vote 2) assign 3) resolve",
    "- 1. first\n- 2. second\n- 3. third",
])
def test_list_markers_are_not_figures(text):
    assert figures(text) == []


@pytest.mark.parametrize("text, expected", [
    # Run 5's motivating chain, untouched.
    ("Book X 35, Book Y 47, Book Z 19, total 104, 2.9% undecided",
     ["35", "47", "19", "104", "2.9%"]),
    ("book_a 9761, book_b 8847, book_c 6951, 226 unresolved",
     ["9761", "8847", "6951", "226"]),
    # Run 6's two surviving catches.
    ("The total time needed is now estimated at 152 hours, 83% complete.",
     ["152", "83%"]),
    ("approximately 2 hours over three weeks, 3500 pages",
     ["2", "3500"]),
    # A figure that simply ends a sentence is not a list marker.
    ("Raise the quorum to 3. That settles it.", ["3"]),
    # A digit closing a parenthesis that opened earlier is content.
    ("The committee (see rule 2) voted 47 to 19.", ["2", "47", "19"]),
    # Thousands separators and percents still canonicalise as before.
    ("Total is 9,761 votes and 19.0% undecided.", ["9761", "19.0%"]),
    # Above ENUMERATOR_MAX: no list enumerates to 21, so this is a quantity.
    ("Section 21. The vote stands.", ["21"]),
])
def test_real_quantities_survive(text, expected):
    assert figures(text) == expected


def test_exclusion_never_invents_a_divergence():
    """Silencing is one-way: a pair that did not fire before must not fire
    now. Excluding figures can only shrink the sets, and a set below
    FIGURE_DIVERGENCE_MIN returns None."""
    previous = "Book X 35, Book Y 47."
    assert figure_divergence(previous, previous) is None


def test_a_prose_dash_is_not_a_bullet():
    """Run 5's agent_ce16fd80 wrote "Book X - 35, Book Y - 47, Book Z - 19."
    A bullet rule that accepted a dash mid-line read "19." as a list marker
    and dropped it -- and "19" is precisely the figure the run-5 chain
    contradicted itself about."""
    text = ("Resolved vote counts: Book X - 35, Book Y - 47, Book Z - 19. "
            "Unresolved votes: 3 remaining pending resolution.")
    assert figures(text) == ["35", "47", "19", "3"]
