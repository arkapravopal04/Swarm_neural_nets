"""
deep_critique's verdict parse. Every tier-3 completion in one run opened with
an invented few-shot pair -- "Example:\\nVERDICT: reject ...\\nVERDICT: accept
..." -- and the last-match parse read the invented accept as the ruling. The
prompt is now prefilled with "VERDICT:", the first verdict wins, and anything
from an invented Example:/VERDICT: line on is cut.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import Judge


def _critique(completion):
    prompts = []

    def llm_call_fn(prompt):
        prompts.append(prompt)
        return completion

    result = Judge(llm_call_fn=llm_call_fn).deep_critique("the output", "the subtask")
    return result, prompts[0]


def test_prompt_ends_with_the_verdict_prefill():
    _, prompt = _critique("accept")
    assert prompt.endswith("VERDICT:")


def test_prefilled_ruling_is_parsed():
    for word in ("accept", "reject"):
        result, _ = _critique(f" {word}\nReasoning: it does what was asked.")
        assert result["verdict"] == word
        assert result["reasoning"].startswith(f"VERDICT: {word}")


def test_invented_example_after_the_ruling_does_not_decide_it():
    result, _ = _critique(
        "reject\nReasoning: restates the task.\n"
        "Example:\nVERDICT: accept\nReasoning: fine.")
    assert result["verdict"] == "reject"
    assert "Example" not in result["reasoning"]
    assert "fine." not in result["reasoning"]


def test_a_second_verdict_line_is_cut():
    result, _ = _critique("reject\nMisses the constraint.\n**VERDICT: accept**\nLooks right.")
    assert result["verdict"] == "reject"
    assert "Looks right" not in result["reasoning"]


def test_the_observed_run_shape_no_longer_reads_as_accept():
    # The model ignored the prefill and opened its own example: there is no
    # ruling of its own, so the parse fails closed instead of taking the
    # invented pair's last line.
    result, _ = _critique("Example:\nVERDICT: reject\nReasoning: x\nVERDICT: accept\nReasoning: y")
    assert result["verdict"] == "reject"


def test_first_verdict_wins_over_later_prose():
    result, _ = _critique("accept\nIt answers the question; nothing to reject here.")
    assert result["verdict"] == "accept"


def test_spaced_spelling_fallback_still_works():
    result, _ = _critique("ac ce pt\nfine")
    assert result["verdict"] == "accept"


def test_nothing_recognisable_fails_closed():
    result, _ = _critique("The output is interesting.")
    assert result["verdict"] == "reject"


def test_inflected_verdict_word_still_counts():
    result, _ = _critique("accepted -- the answer is right.")
    assert result["verdict"] == "accept"


def test_reasoning_that_opens_with_the_word_example_is_kept():
    result, _ = _critique("reject\nExample output lacks the units the task asked for.")
    assert result["verdict"] == "reject"
    assert "lacks the units" in result["reasoning"]
