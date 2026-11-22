"""Think first (THINK_FIRST.md P1): reading ANSWER lines, the vote, the
fallback, the plan, and the batched generate."""
import contextlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import thinker  # noqa: E402
from thinker import Thinker, agree, decide, parse_vote, plan_text  # noqa: E402


def run(answer, capped=False, closed=True):
    head = "<think>\nworking...\n</think>\n\n" if closed else "<think>\nworking..."
    return {"text": f"{head}PLAN:\n1. Set it up.\n2. Solve it.\nANSWER: {answer}", "capped": capped, "tokens": 9}


@pytest.mark.parametrize("value, expected, unit", [
    ("45,000 rupees", 45000.0, None), ("−7.78", -7.78, None), ("1.2e-3 mol/L", 0.0012, "mol/L"),
    ("12.5%", 12.5, "%"), ("$\\frac{1}{3}$", 1 / 3, None), ("about 5763.65 m/s", 5763.65, "m/s"),
    ("x = 4", 4.0, None), ("$\\boxed{20\\text{ mA}}$", 20.0, "mA"),
])
def test_a_number_with_an_optional_unit_votes(value, expected, unit):
    assert parse_vote(value, False) == ("num", pytest.approx(expected), unit)


@pytest.mark.parametrize("value", [
    "NONE", None, "5 groups of 9", "Take the bus: 40,000 rupees", "20 mA per channel",
])
def test_prose_none_and_plans_never_vote(value):
    assert parse_vote(value, False) is None


def test_option_items_vote_with_a_bare_letter_only():
    assert parse_vote("(c)", True) == ("choice", "C") and parse_vote("\\boxed{B}", True) == ("choice", "B")
    assert parse_vote("B because it fits", True) is None and parse_vote("42", True) is None


def test_agreement_rules():
    v = lambda s: parse_vote(s, False)  # noqa: E731
    assert agree(v("20 mA"), v("0.02 A")) and not agree(v("20 mA"), v("20 A"))   # converted
    assert agree(v("20"), v("20 mA")) and not agree(v("20"), v("0.02 A"))        # unitless: raw
    assert not agree(("num", 5.0, "pF"), ("num", 5.0, "nF"))                     # unconvertible units
    assert agree(v("1/3"), v("0.3333")) and not agree(v("0.33"), v("0.3333"))    # rel_tol 1e-3
    assert not agree(v("2e-13"), v("9e-13"))                                     # no absolute floor


def test_agreement_settles_with_a_line_from_an_uncapped_run():
    d = decide([run("0.3333", capped=True), run("7"), run("1/3")], False)
    assert d["answer_line"] == "ANSWER: 1/3" and d["plan_source"] == "plurality"


def test_two_capped_runs_agreeing_do_not_settle_but_are_the_fallback():
    d = decide([run("5", capped=True), run("5", capped=True), run("9")], False)
    assert d["answer_line"] is None and d["fallback_line"] == "ANSWER: 5"


def test_disagreement_falls_back_to_the_first_uncapped_run():
    d = decide([run("1", capped=True), run("2"), run("3")], False)
    assert d["answer_line"] is None and d["fallback_line"] == "ANSWER: 2"
    assert d["fallback_run"] == 1 and d["plan_source"] == "uncapped"


def test_answers_inside_unclosed_thinking_are_not_read():
    d = decide([run("5", closed=False)] * 3, False)
    assert d["answer_line"] is None and d["fallback_line"] is None
    assert d["plan"] is None and d["plan_source"] == "none"


def test_the_last_answer_line_counts():
    assert thinker.answer_value("<think>x</think>\nANSWER: 4\nNo, wait.\nANSWER: 5") == "5"


def test_the_plan_is_capped_and_carries_no_answer_line():
    long_plan = "\n".join(f"{i}. step {i} " + "x" * 120 for i in range(1, 7))
    p = plan_text(f"<think>a</think>\nPLAN:\n{long_plan}\nANSWER: 3")
    assert p.startswith("1. step 1") and len(p) <= thinker.PLAN_MAX_CHARS and "ANSWER" not in p


def test_solve():
    seen = {}

    def fake(prompt, n):
        seen.update(prompt=prompt, n=n)
        return [run("B"), run("b"), run("A")]

    req = "Which?\n\nOptions: A) one; B) two. " + thinker.CHOICE_MARK
    out = Thinker(n=3, generate_fn=fake).solve(req)
    assert seen["n"] == 3 and seen["prompt"].startswith(req) and "PLAN:" in seen["prompt"]
    assert out["answer_line"] == "ANSWER: B" and [r["answer"] for r in out["runs"]] == ["B", "b", "A"]


def test_settle_off_and_n1_never_end_the_run():
    # G5: two school runs ending "ANSWER: 40,000 rupees" must not replace the plan.
    out = Thinker(settle=False, generate_fn=lambda p, n: [run("40,000 rupees")] * 3).solve("school")
    assert out["answer_line"] is None and out["plan"]
    out = Thinker(n=1, generate_fn=lambda p, n: [run("2")]).solve("q")
    assert out["answer_line"] is None and out["fallback_line"] == "ANSWER: 2"


class _Tok:
    """One id per character; ids 0 and 1 are the two stop tokens."""
    pad_token_id = 0

    def convert_tokens_to_ids(self, t):
        return {"<|im_end|>": 1, "<|endoftext|>": 0}[t]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        import torch
        return {"input_ids": torch.tensor([[ord(c) + 10 for c in text]])}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i - 10) for i in (ids.tolist() if hasattr(ids, "tolist") else ids) if i >= 10)


class _Model:
    """Row 0 answers and stops (then pads); row 1 is capped mid-thought."""
    device = "cpu"

    def __init__(self):
        self.adapter_off, self.calls = [], []

    def disable_adapter(self):
        self.adapter_off.append(True)
        return contextlib.nullcontext()

    def generate(self, input_ids, attention_mask, max_new_tokens, **kw):
        import torch
        self.calls.append(kw)
        enc = lambda s: [ord(c) + 10 for c in s]  # noqa: E731
        prompt = input_ids[0].tolist()
        if kw.get("num_return_sequences"):
            done = enc("<think>a</think>\nANSWER: 7") + [1, 0, 0]
            return torch.tensor([prompt + done, prompt + enc("<think>" + "b" * (len(done) - 7))])
        return torch.tensor([prompt + enc("ANSWER: 7") + [1]])


def test_generate_closes_a_capped_run_with_the_adapter_off(monkeypatch):
    monkeypatch.setenv("HIVE_SERVING", "v2")
    monkeypatch.setattr(thinker.serving, "render", lambda tok, prompt, thinking: [ord("Q") + 10])
    model = _Model()
    runs = Thinker(model, _Tok(), n=2)._generate("Q", 2)
    assert [r["capped"] for r in runs] == [False, True]
    assert [thinker.answer_value(r["text"]) for r in runs] == ["7", "7"]  # row 1 closed, then answered
    assert model.adapter_off and model.calls[0]["eos_token_id"] == [1, 0]
    assert model.calls[1]["do_sample"] is False
