"""P3: under HIVE_SERVING=v2 the tier-3 judge asks whether the figures are
right, and only a leading "VERDICT: accept|reject" counts (fail closed).
Legacy (the default) is unchanged."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import Judge  # noqa: E402


def _run(completion):
    seen = {}

    def llm_call_fn(prompt):
        seen["prompt"] = prompt
        return completion
    result = Judge(llm_call_fn=llm_call_fn).deep_critique("The current is 3 A.", "Find the current.")
    return result, seen["prompt"]


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setenv("HIVE_SERVING", "v2")


@pytest.fixture
def legacy(monkeypatch):
    monkeypatch.delenv("HIVE_SERVING", raising=False)


def test_v2_prompt_asks_whether_the_figures_are_right(v2):
    _, prompt = _run(" accept")
    assert "is every figure in the OUTPUT right?" in prompt
    assert "confidence calibrated" not in prompt
    assert prompt.endswith(Judge.VERDICT_PREFILL)


def test_legacy_prompt_is_unchanged(legacy):
    _, prompt = _run(" accept")
    assert "is its confidence calibrated" in prompt and "is every figure in the OUTPUT right?" not in prompt


@pytest.mark.parametrize("completion,want", [
    (" accept. 230 / 230 = 1, so 3 A is wrong... no, the output is fine.", "accept"),
    (" reject. 230 / 230 = 1 A, not 3 A.", "reject"),
    (" Accepted.", "accept"),
    (" **accept** -- the figures check out.", "accept"),
    (" 'reject': 230 / 230 = 1 A.", "reject"),
    # Not a leading ruling: the legacy fallback would read "accept" here.
    (" I would not accept this: the current is 1 A.", "reject"),
    ("", "reject"),
])
def test_v2_only_a_leading_ruling_counts(v2, completion, want):
    result, _ = _run(completion)
    assert result["verdict"] == want


def test_legacy_keeps_its_fallback(legacy):
    result, _ = _run(" I would not accept this: the current is 1 A.")
    assert result["verdict"] == "accept"  # the known weakness P3 fixes under v2


def _prompt_with_built_on():
    seen = {}

    def llm_call_fn(prompt):
        seen["prompt"] = prompt
        return " accept"
    Judge(llm_call_fn=llm_call_fn).deep_critique("Print 246 badges.", "Plan the conference.",
                                                 built_on=["Count the badges: 246 badges."])
    return seen["prompt"]


def test_v2_critique_sees_what_the_output_builds_on(v2):
    assert "RESULTS THIS OUTPUT BUILDS ON" in _prompt_with_built_on()


def test_legacy_critique_ignores_built_on(legacy):
    assert "RESULTS THIS OUTPUT BUILDS ON" not in _prompt_with_built_on()


def _prompt_with_code():
    seen = {}

    def llm_call_fn(prompt):
        seen["prompt"] = prompt
        return " reject"
    Judge(llm_call_fn=llm_call_fn).deep_critique(
        "The probability is 0.24.", "Find P(both red), drawn without replacement.",
        tool_output="0.24", tool_code="print(float(F(3, 5) * F(2, 5)))")
    return seen["prompt"]


def test_v2_critique_sees_the_code_behind_a_printout(v2):
    prompt = _prompt_with_code()
    assert "THAT CODE:\nprint(float(F(3, 5) * F(2, 5)))" in prompt
    assert "Check the code's inputs against the figures the request gives" in prompt


def test_legacy_critique_does_not_show_the_code(legacy):
    assert "THAT CODE:" not in _prompt_with_code()
