"""
cut_instruction_tail -- the same-line scaffold cut added after run 6.

Run 6's goal_vector was built from a 246-character string of which the last
116 were delimiter:

  "...disliked selections.---THESE THREE QUESTIONS NEED TO BE ANSWERED IN
   ORDER--END OF OUTPUT-- -- -- -- ..."

Nothing upstream could reach it. _sanitize_generation's stop markers are
"\nInput:", "\nAnswer:", "\nEXAMPLES", "\nGIVEN TEXT", "\n\n", "Here is
the", "Certainly!" and "Sure," -- every one begins with a newline or is a
conversational opener, and the delimiter arrives on the SAME LINE as the
goal's final full stop. trim_artifact_tail scans backwards from the end and
stops at "OUTPUT--", which is not a separator token.

The negative cases are the point of the test: this runs on the one string
that becomes the colony's reference vector, so a predicate that eats a
legitimate em dash, hyphenated phrase or acronym is worse than no predicate.
"""
import pytest

from text_utils import cut_instruction_tail


RUN_6_GOAL = (
    "Choose books based on member votes; assign readings per person "
    "completion status; find consensus solutions for disliked selections."
    "---THESE THREE QUESTIONS NEED TO BE ANSWERED IN ORDER--END OF OUTPUT"
    "-- -- -- -- \n -- -- -- --- \n -- --- ---- \n -- - - \n - -"
)
RUN_6_GOAL_CLEAN = (
    "Choose books based on member votes; assign readings per person "
    "completion status; find consensus solutions for disliked selections."
)


def test_run_6_goal_is_cut_back_to_the_goal():
    kept, reasons = cut_instruction_tail(RUN_6_GOAL)
    assert kept == RUN_6_GOAL_CLEAN
    assert reasons == ["dash run"]
    assert len(kept) == 131 and len(RUN_6_GOAL) == 254


@pytest.mark.parametrize("text, reason, kept_text", [
    # The dash run, on its own and glued to the sentence it follows.
    ("Choose books based on member votes.---END OF OUTPUT--", "dash run",
     "Choose books based on member votes."),
    ("Summarize the minutes ——— and stop.", "dash run",
     "Summarize the minutes"),
    # The marker without any dashes in front of it.
    ("Work out the weighted vote totals. END OF OUTPUT", "end-of-output marker",
     "Work out the weighted vote totals."),
    ("Rank the options. end of response", "end-of-output marker",
     "Rank the options."),
    # An all-caps instruction appended after a sentence terminator.
    ("Decide a fair way to pick each month's trail. ANSWER ALL THREE PARTS "
     "IN ORDER.", "appended caps block",
     "Decide a fair way to pick each month's trail."),
])
def test_scaffold_is_cut(text, reason, kept_text):
    kept, reasons = cut_instruction_tail(text)
    assert reasons == [reason]
    assert kept == kept_text


@pytest.mark.parametrize("text", [
    # Hyphenated compounds, "--" standing in for an em dash (which this
    # codebase and the model both write), and a real em dash. This is why
    # the dash rule needs THREE hyphens and TWO en/em dashes, not two/one.
    "Draft a state-of-the-art, well-documented plan for the long-term "
    "rollout -- the part the board cares about — including cost.",
    "Pick a trail -- ideally a loop -- for the group's monthly hike.",
    # Negative numbers and a year range.
    "Plot the temperature curve from -40 to -5 degrees over the 2019-2024 "
    "window.",
    # Three consecutive all-caps words MID-sentence. The sentence-terminator
    # requirement is the only thing keeping this one whole.
    "Decide which of the NASA, ESA and JAXA proposals the committee should "
    "fund.",
    # One caps word after a terminator, then prose.
    "Summarize the Q4 report. NATO spending rose 3% year-over-year across "
    "the alliance.",
    # An appended acronym list with no word of 4+ characters: a list, not an
    # instruction.
    "Rank the agencies by budget. FBI CIA NSA.",
])
def test_legitimate_text_survives_untouched(text):
    kept, reasons = cut_instruction_tail(text)
    assert kept == text
    assert reasons == []


def test_never_returns_empty():
    """A marker at offset 0 means the extraction failed, not that the string
    is scaffold. The caller gets the original back so the degenerate parse
    stays visible to the checks downstream instead of being blanked here."""
    for text in ("---END OF OUTPUT---", "END OF OUTPUT", "-- -- --"):
        kept, reasons = cut_instruction_tail(text)
        assert kept == text
        assert reasons == []


def test_empty_and_none_pass_through():
    assert cut_instruction_tail("") == ("", [])
    assert cut_instruction_tail(None) == (None, [])
    assert cut_instruction_tail("   ") == ("   ", [])


def test_is_idempotent():
    once, _ = cut_instruction_tail(RUN_6_GOAL)
    twice, reasons = cut_instruction_tail(once)
    assert twice == once and reasons == []
