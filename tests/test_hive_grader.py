"""
hive_grader.py regression cases, one block per finding of the 2026-10-08
review (five reviewers, repros re-run). Real benchmark items are used where
the finding named one; keys are read from reasoning_benchmark/, never copied.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import hive_grader as hg
from hive_grader import grade, grade_detail

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = hg.load_problems()


def P(pid):
    return PROBLEMS[pid]


def num_item(answer, unit, tolerance=1e-6, pid="X-NUM"):
    return dict(id=pid, answer_type="numeric", answer=answer, unit=unit, tolerance=tolerance)


def choice_item(answer="B"):
    return dict(id="X-CH", answer_type="choice", answer=answer,
                choices={"A": "first", "B": "second", "C": "third", "D": "fourth"})


MA = num_item(20, "mA")
PROB = num_item(1 / 3, "probability")


def status(problem, answer, **kw):
    return grade_detail(problem, answer, **kw)["status"]


# ------------------------------------------------- 1. molar flow factors

def test_mol_per_second_factor():
    t204 = P("T2-04")
    assert t204["unit"] == "mol/min"
    per_s = t204["answer"] / 60
    assert grade(t204, f"{per_s!r} mol/s") == 1.0   # tolerance 1e-6: rounded values still fail
    assert grade(t204, f"{t204['answer'] * 60} mol/s") == 0.0
    assert grade(t204, f"{t204['answer'] * 60} mol/h") == 1.0
    assert grade(t204, f"{t204['answer']} mol per minute") == 1.0


# ------------------------------------------------- 2. unknown / scaled units

@pytest.mark.parametrize("ans", ["20 ohm", "20 Hz", "20 N", "20 Ω", "20 kΩ"])
def test_other_dimension_on_current_item_scores_zero(ans):
    assert grade(MA, ans) == 0.0


def test_unknown_unit_is_flagged_not_ignored():
    assert status(MA, "20 Zq") == "unknown_unit"
    assert grade(MA, "20 Zq") == 0.0


@pytest.mark.parametrize("ans,key,unit", [
    ("396 µJ", 396e-6, "J"), ("396 μJ", 396e-6, "J"), ("1.52 km/s", 1520, "m/s"),
    ("0.125 days", 3, "hours"), ("3 h", 3, "hours"), ("1.2 KW", 1200, "W"),
    ("250 MPA", 250, "MPa"), ("250 N/mm^2", 250, "MPa"), ("5 µmol/L", 5, "micromolar"),
    ("5 μmol/L", 5, "micromolar"), ("2 Mg", 2000, "kg"),
])
def test_scaled_units_convert(ans, key, unit):
    assert grade(num_item(key, unit), ans) == 1.0


def test_mg_is_not_megagram():
    assert grade(num_item(2000, "kg"), "2 mg") == 0.0


def test_compound_unit_not_read_by_prefix():
    assert grade(num_item(1200, "W"), "1.2 kW·h") == 0.0


def test_unit_field_applies_to_numeric_string():
    assert grade_detail(MA, "0.02", unit="A")["score"] == 1.0


def test_count_and_probability_items_ignore_physical_units():
    assert grade(num_item(7, "socks"), "7 socks") == 1.0
    assert grade(num_item(7, "socks"), "7 s") == 1.0
    assert grade(num_item(0.25, "probability"), "0.25 s") == 1.0


def test_percent_vs_fraction_stays_distinct():
    assert status(num_item(0.25, "probability"), "25%") == "unit_mismatch"
    assert grade(num_item(25, "percent"), "25%") == 1.0


def test_bytes_not_from_label_b():
    assert grade(num_item(16, "bits"), "2 bytes") == 1.0
    assert status(num_item(16, "bits"), "2 B") == "unknown_unit"


# ------------------------------------------------- 3. overflow

@pytest.mark.parametrize("ans", ["answer: 1e999", "7 x 10^400", "1" * 400, 10 ** 400, "1" + "0" * 400 + " mA"])
def test_huge_numbers_do_not_crash(ans):
    d = grade_detail(MA, ans)
    assert d["score"] == 0.0


def test_one_bad_item_does_not_sink_the_run(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(hg, "extract_number", boom)
    items = hg.grade_all(PROBLEMS, [{"id": "T1-01", "answer": "7"}], quiet=True)
    by = {i["id"]: i for i in items}
    assert by["T1-01"]["status"] == "grader_error"
    hg.summarize(items)


# ------------------------------------------------- 4-9. number extraction

@pytest.mark.parametrize("ans,key,unit", [
    ("Answer: V = I*R = 2*3 = 6 V", 6, "V"),
    ("Final answer: 42. Note: this answer assumes 3 shifts", 42, "dimensionless"),
    ("The answer is 42. Some answers differ; 3 were unanswered.", 42, "dimensionless"),
    ("Final answer: 42\nAs a result, 3 rows remain unused", 42, "dimensionless"),
    ("The current is 20 mA (12 V ÷ 600 Ω).", 20, "mA"),
    ("The flow time is 3 hours (step 3), given to 2 significant figures", 3, "hours"),
    ("Answer: 1/3", 1 / 3, "probability"),
    ("Answer: \\frac{1}{3}", 1 / 3, "probability"),
    ("Answer: \\dfrac{1}{3}", 1 / 3, "probability"),
    ("The probability is 1/3 ≈ 0.333, answer = 1/3", 1 / 3, "probability"),
    ("Answer: 1/3 ≈ 0.3333", 1 / 3, "probability"),
    ("Answer: P = 0.25 × 0.5 ≈ 0.125", 0.125, "probability"),
    ("Answer: 1.2 \\times 10^{-4}", 1.2e-4, "dimensionless"),
    ("$\\boxed{\\frac{1}{3}}$", 1 / 3, "probability"),
    ("\\boxed{\\text{20 mA}}", 20, "mA"),
    ("$1.2\\,\\text{kW}$", 1200, "W"),
    ("The concentration is 0.333 mol·L^-1", 0.333, "mol/L"),
    ("The stress is 20 N/mm^2", 20, "MPa"),
    ("The stress is 20 N/mm²", 20, "MPa"),
    ("Answer:\n\n**6 V**", 6, "V"),
    ("Result = 3.0e2", 300, "dimensionless"),
    ("```json\n{\"id\": \"T1-01\", \"answer\": 7}\n```", 7, "dimensionless"),
    ("I computed it. {\"answer\": 7} is the final.", 7, "dimensionless"),
    ("At 12 V the answer is 20 mA", 20, "mA"),
    ("1,250", 1250, "dimensionless"),
    ("−3", -3, "dimensionless"),
])
def test_prose_numbers(ans, key, unit):
    d = grade_detail(num_item(key, unit, 1e-9), ans)
    assert d["status"] == "correct", d


@pytest.mark.parametrize("ans,key,unit", [
    ("Final answer: 20 mA. Result is verified with SPICE (3 iterations).", 20, "mA"),
    ("Final answer: 20 mA. The result is consistent with Kirchhoff's 2nd law.", 20, "mA"),
    ("Answer: 7 (naive). Correcting: Final answer: 42", 42, "dimensionless"),
    ("Answer: 6 V = 2 A * 3 ohm", 6, "V"),
    ("Answer: 107.08203932 MPa [1 MPa = 10^6 Pa]", 107.08203932, "MPa"),
    ("Answer: 20 ± 1 mA", 20, "mA"),
    ("Answer: 20 +/- 1 mA", 20, "mA"),
    ("Answer: 0.462475 or 46.2475%", 0.462475, "portfolio weight"),
    ("Answer: 3 h or 180 min", 3, "hours"),
    ("Answer: 20 000 µA", 20, "mA"),
    ("Answer: 1 000 cells", 1000, "cells"),
    ("Answer: 10^3", 1000, "cells"),
    ("Answer: 10**3", 1000, "cells"),
    ("Answer: 3.96 x 10**-4 J", 3.96e-4, "J"),
    ("Answer: 3.96×10−4 J", 3.96e-4, "J"),
    ("Answer: 3.96 x 10^(-4) J", 3.96e-4, "J"),
    ("Answer: 396 microjoules", 3.96e-4, "J"),
    ("\\boxed{\\frac13}", 1 / 3, "probability"),
    ("\\boxed{\\sqrt{2.25}}", 1.5, "dimensionless"),
    ("Answer: _10_", 10, "dimensionless"),
    ("Answer: 7.5 Hz", 450, "rpm"),
    ("Answer: 450 min^-1", 450, "rpm"),
    ("Answer: 6 Vrms", 6, "V"),
    ("Answer: 1.2 kVA", 1200, "W"),
    ("Answer: 0.3333333 Mol/L", 0.3333333, "mol/L"),
    ("Answer: 0.3333333 moles per litre", 0.3333333, "mol/L"),
    ("Answer: 6 µmol L-1", 6, "micromolar"),
    ("Answer: 6 μmol·L⁻¹", 6, "micromolar"),
    ("Answer: 0.5 mg·mL⁻¹", 0.5, "mg/mL"),
    ("Answer: 0.5 mg / mL", 0.5, "mg/mL"),
    ("Answer: 0.5 kg/m3", 0.5, "mg/mL"),
    ("Answer: 0.5 milligrams per milliliter", 0.5, "mg/mL"),
    ("Answer: 15 bits per symbol", 15, "bits"),
    ('{"answer": {"value": 20, "unit": "mA"}}', 20, "mA"),
    ('{"final_answer": 20}', 20, "mA"),
    ('[20, "mA"]', 20, "mA"),
    ('Result: {"answer": 20, "unit": "mA", "notes": {"a": 1}}', 20, "mA"),
])
def test_prose_numbers_second_pass(ans, key, unit):
    d = grade_detail(num_item(key, unit, 1e-6), ans)
    assert d["status"] == "correct", d


@pytest.mark.parametrize("ans,unit,expect", [
    ("Answer: 15-20 mA", "mA", "invalid_conflicting_answers"),
    ("Answer: between 15 and 20 mA", "mA", "invalid_conflicting_answers"),
    ("Answer: 20 MA", "mA", "unknown_unit"),
    ("Answer: 30 lb", "kg", "incorrect"),
    ("Answer: 15 ‰", "percent", "unit_mismatch"),
    ("The current is 20 mA and the voltage 12 V", "mA", "correct"),
])
def test_second_pass_rejections(ans, unit, expect):
    assert status(num_item(20 if unit in ("mA", "kg") else 15, unit), ans) == expect


@pytest.mark.parametrize("ans,key", [
    ("The answer is not A, it is C", "C"),
    ("Answer: A and B are wrong; C", "C"),
    ("A is wrong, B is wrong, C is right", "C"),
    ("Option A is wrong. Option C is right.", "C"),
    ("(A) is wrong but (C) is correct", "C"),
    ("Not A. Not B. C.", "C"),
    ("Answer: 'C'", "C"),
    ("{'answer': 'C'}", "C"),
    ("Answer: a) foo", "A"),
])
def test_choice_second_pass(ans, key):
    assert grade_detail(choice_item(key), ans)["status"] == "correct"


@pytest.mark.parametrize("ans", ["It is not A.", "Neither A nor B.", "A, C, B, D", "a) foo b) bar"])
def test_choice_no_answer(ans):
    assert grade_detail(choice_item("A"), ans)["score"] == 0.0


def test_mu_is_not_dropped():
    assert grade(MA, "20 \\mu A") == 0.0
    assert grade(MA, "20000 \\mu A") == 1.0


@pytest.mark.parametrize("ans", ["answer: 7 or 8", "Answer: 7 or 8 depending on rounding",
                                 "It is either 7 or 8"])
def test_conflicting_numbers_score_zero(ans):
    d = grade_detail(num_item(7, "dimensionless"), ans)
    assert d["score"] == 0.0 and d["status"] == "invalid_conflicting_answers"


# ------------------------------------------------- 10-14. choice extraction

@pytest.mark.parametrize("ans", [
    "The answer is a bit tricky, so B",
    "Answer: B\nI double-checked the result.",
    "Answer: B. As a result, A and C are ruled out.",
    "{\"answer\": \"B\"}",
    "Here it is:\n```json\n{\"id\": \"X-CH\", \"answer\": \"B\"}\n```",
    "I choose B",
    "B, because A is false.",
    "**B**",
    "(B)",
    "b",
    "Answer: (b)",
    "Option B",
    "The correct option is B.",
    "\\boxed{B}",
    "A is wrong. The answer is B.",
])
def test_choice_prose(ans):
    d = grade_detail(choice_item("B"), ans)
    assert d["status"] == "correct", d


def test_choice_label_a():
    assert grade(choice_item("A"), "Answer: A is correct") == 1.0
    assert grade(choice_item("A"), "A") == 1.0


@pytest.mark.parametrize("ans", ["Answer: B or C", "Answer: B/C", "B and C"])
def test_conflicting_labels_score_zero(ans):
    d = grade_detail(choice_item("B"), ans)
    assert d["score"] == 0.0 and d["status"].startswith("invalid_"), d


def test_choice_by_option_text():
    item = P("Z0-05")
    text = item["choices"][item["answer"]]
    assert grade(item, f"I'd say: {text}") == 1.0


# ------------------------------------------------- 15. --strict == scripts/grade.py

def _official(problems, responses):
    sys.path.insert(0, str(ROOT / "reasoning_benchmark" / "scripts"))
    try:
        import grade as official
    finally:
        sys.path.pop(0)
    return official.evaluate(problems, responses)


def test_strict_rejects_nested_and_units():
    t101 = P("T1-01")
    assert grade(t101, {"answer": t101["answer"]}, strict=True) == 0.0
    assert grade(t101, f"{t101['answer']}", strict=True) == 0.0
    assert grade(t101, t101["answer"], strict=True) == 1.0
    assert grade_detail(MA, 0.02, strict=True, unit="A")["score"] == 0.0


def test_strict_matches_official_scorer_on_main():
    main = [p for p in PROBLEMS.values() if p["split"] == "main"]
    responses = [{"id": p["id"], "answer": p["answer"]} for p in main if p["answer_type"] != "rubric"]
    responses[0]["answer"] = "garbage"
    responses[1]["answer"] = True
    del responses[2]
    off = _official(main, responses)
    items = hg.grade_all(PROBLEMS, responses, strict=True, quiet=True)
    mine = hg.summarize(items, strict=True)["main"]
    assert mine["total_pct"] == pytest.approx(off["score_percent"])
    assert mine["provisional"] == off["provisional"] is True
    for t, row in off["tier_scores"].items():
        assert mine["tiers"][int(t)]["weighted_pct"] == pytest.approx(row["score_percent"])
    off_status = {r["id"]: r["status"] for r in off["items"]}
    assert {i["id"]: i["status"] for i in items if i["split"] == "main"} == off_status
    assert sorted(mine["invalid_answers"]) == sorted(off["invalid_answers"])
    assert sorted(mine["missing_or_ungraded"]) == sorted(off["missing_or_ungraded"])


def test_all_correct_tiers_1_4_is_provisional_not_official():
    main = [p for p in PROBLEMS.values() if p["split"] == "main"]
    responses = [{"id": p["id"], "answer": p["answer"]} for p in main if p["answer_type"] != "rubric"]
    rep = hg.summarize(hg.grade_all(PROBLEMS, responses, quiet=True))["main"]
    assert rep["total_pct"] == pytest.approx(100.0)
    assert rep["provisional"] and rep["ungraded_tiers"] == [5]
    assert "NOT the official score" in hg.render_table({"main": rep})
    strict = hg.summarize(hg.grade_all(PROBLEMS, responses, strict=True, quiet=True), strict=True)["main"]
    assert strict["total_pct"] == pytest.approx(80.0) and strict["provisional"]


# ------------------------------------------------- 16. provisional flag

def test_invalid_answer_sets_provisional():
    t0 = [{"id": p["id"], "answer": p["answer"]} for p in PROBLEMS.values() if p["split"] == "tier0"]
    rep = hg.summarize(hg.grade_all(PROBLEMS, t0, quiet=True))["tier0"]
    assert not rep["provisional"]
    t0[0]["answer"] = "no idea"
    rep = hg.summarize(hg.grade_all(PROBLEMS, t0, quiet=True))["tier0"]
    assert rep["provisional"] and rep["invalid_answers"] == [t0[0]["id"]]


def test_ungraded_tier5_listed():
    rep = hg.summarize(hg.grade_all(PROBLEMS, [{"id": "T1-01", "answer": 7}], quiet=True))["main"]
    assert any(i.startswith("T5-") for i in rep["missing_or_ungraded"])


# ------------------------------------------------- 17-18. outputs, rubric scores

def test_reports_never_contain_keys(tmp_path):
    hold = [p for p in PROBLEMS.values() if p["split"] == "holdout" and p["answer_type"] == "numeric"][0]
    ans = tmp_path / "a.jsonl"
    ans.write_text(json.dumps({"id": hold["id"], "answer": "0"}) + "\n", encoding="utf-8")
    hg.main(["--answers", str(ans), "--json", str(tmp_path / "r.json"),
             "--csv", str(tmp_path / "r.csv"), "-q"])
    rep = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    for it in rep["items"]:
        assert "key" not in it and "explanation" not in it
    assert "key" not in (tmp_path / "r.csv").read_text(encoding="utf-8").splitlines()[0].split(",")


def test_missing_and_invalid_kept_apart():
    items = hg.grade_all(PROBLEMS, [{"id": "Z0-01", "answer": "nothing"}], quiet=True)
    rep = hg.summarize(items)["tier0"]
    assert "Z0-01" in rep["invalid_answers"] and "Z0-01" not in rep["missing_or_ungraded"]
    assert "Z0-02" in rep["missing_or_ungraded"]


def test_rubric_scores_validated_and_used():
    t5 = P("T5-01")
    ids = [c["id"] for c in t5["rubric"]]
    good = {i: 2 for i in ids}
    assert grade_detail(t5, "x", rubric_scores=good)["score"] == 1.0
    with pytest.raises(ValueError):
        grade_detail(t5, "x", rubric_scores={ids[0]: 2})
    with pytest.raises(ValueError):
        hg.grade_all(PROBLEMS, [{"id": "T5-01", "answer": "x", "rubric_scores": {i: 3 for i in ids}}], quiet=True)


# ------------------------------------------------- I/O hardening

def _run(tmp_path, content, *extra):
    ans = tmp_path / "a.jsonl"
    ans.write_bytes(content)
    return subprocess.run([sys.executable, str(ROOT / "hive_grader.py"), "--answers", str(ans), "-q", *extra],
                          capture_output=True, text=True, encoding="utf-8")


def test_bom_is_accepted(tmp_path):
    r = _run(tmp_path, "﻿".encode() + b'{"id": "Z0-01", "answer": "10"}\n')
    assert r.returncode == 0, r.stderr


def test_bad_line_reports_line_number(tmp_path):
    r = _run(tmp_path, b'{"id": "Z0-01", "answer": 10}\n{oops\n')
    assert r.returncode == 2 and ":2:" in r.stderr


def test_row_without_id(tmp_path):
    r = _run(tmp_path, b'{"answer": 10}\n')
    assert r.returncode == 2 and "id" in r.stderr


def test_csv_with_no_items(tmp_path):
    r = _run(tmp_path, b"", "--csv", str(tmp_path / "o.csv"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o.csv").read_text(encoding="utf-8").startswith("id,")


def test_exit_code_on_invalid(tmp_path):
    r = _run(tmp_path, b'{"id": "Z0-01", "answer": "no idea"}\n')
    assert r.returncode == 1
