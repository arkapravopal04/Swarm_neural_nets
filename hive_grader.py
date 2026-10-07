"""
hive_grader.py -- automatic grader for Hive answers on Reasoning Benchmark v2.

    grade(problem, answer) -> float in [0, 1]

Item kinds (from reasoning_benchmark/schema/problem.schema.json):

  numeric  (tiers 0-4)  answer is a number with `unit`, `tolerance`
                        (absolute, inclusive). We parse value + unit out of
                        the Hive answer, convert same-dimension units
                        (mA -> A, min -> hours, uM -> mol/L ...) and test
                        abs(value - key) <= tolerance. Percent vs fraction is
                        NOT converted (SCORING.md treats them as distinct);
                        a dimension mismatch scores 0.
  choice   (tiers 0-4)  label A/B/C/D, exact match. Conflicting labels -> 0.
  rubric   (tier 5)     NOT graded. No grading path exists yet: every tier-5
                        item is "ungraded" (score None), shown as "ungraded"
                        in the table and left out of every total.

Scoring/aggregation follows reasoning_benchmark/SCORING.md:
  tier %   = 100 * sum(w_i * s_i) / sum(w_i)      (w = domain multiplier)
  total %  = mean of the graded tiers per split (currently tiers 1-4;
             main / holdout separately). Tier 5 joins once it is graded.
  tier 0   = unweighted percent correct, reported on its own.

--strict reproduces the official scorer exactly (bare JSON number / label
only, no prose extraction, no unit conversion).

Answers file: JSONL, one object per line:
    {"id": "T1-03", "answer": "20 mA"}
    {"id": "T1-10", "answer": "The answer is (B)"}
optional "unit" field is honoured when `answer` is a bare number.

Usage:
    python hive_grader.py --answers hive_answers.jsonl
    python hive_grader.py --answers a.jsonl --json report.json --csv items.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

DEFAULT_BENCH = Path(__file__).resolve().parent / "reasoning_benchmark"
KEY_FILES = ("tests/problems.jsonl", "tests/holdout.jsonl", "tests/tier0.jsonl")
SPLIT_ORDER = ("main", "holdout", "tier0")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_problems(bench_dir=DEFAULT_BENCH):
    problems = {}
    for rel in KEY_FILES:
        for p in read_jsonl(Path(bench_dir) / rel):
            if p["id"] in problems:
                raise ValueError(f"duplicate problem id {p['id']}")
            problems[p["id"]] = p
    return problems


# ---------------------------------------------------------------------------
# Units: alias -> (dimension, factor to that dimension's base unit)
# Count-like units in the bench ("rows", "socks", "currency units", ...) are
# not listed; any unknown word after a number is ignored and the stated unit
# is assumed.
# ---------------------------------------------------------------------------

_UNITS = {}


def _add(dim, factor, *aliases):
    for a in aliases:
        _UNITS[a] = (dim, factor)


_add("time", 1, "s", "sec", "secs", "second", "seconds")
_add("time", 1e-3, "ms", "millisecond", "milliseconds")
_add("time", 60, "min", "mins", "minute", "minutes")
_add("time", 3600, "h", "hr", "hrs", "hour", "hours")
_add("current", 1, "A", "amp", "amps", "ampere", "amperes")
_add("current", 1e-3, "mA", "milliamp", "milliamps", "milliampere", "milliamperes")
_add("current", 1e-6, "uA", "µA", "μA", "microamp", "microamps", "microampere", "microamperes")
_add("voltage", 1, "V", "volt", "volts")
_add("voltage", 1e-3, "mV", "millivolt", "millivolts")
_add("voltage", 1e-6, "uV", "µV", "μV", "microvolt", "microvolts")
_add("voltage", 1e3, "kV", "kilovolt", "kilovolts")
_add("power", 1, "W", "watt", "watts")
_add("power", 1e-3, "mW", "milliwatt", "milliwatts")
_add("power", 1e3, "kW", "kilowatt", "kilowatts")
_add("power", 1e6, "MW", "megawatt", "megawatts")
_add("energy", 1, "J", "joule", "joules")
_add("energy", 1e-3, "mJ", "millijoule", "millijoules")
_add("energy", 1e3, "kJ", "kilojoule", "kilojoules")
_add("energy", 1e6, "MJ", "megajoule", "megajoules")
_add("pressure", 1, "Pa", "pascal", "pascals")
_add("pressure", 1e3, "kPa")
_add("pressure", 1e6, "MPa", "N/mm^2", "N/mm2")
_add("pressure", 1e9, "GPa")
_add("pressure", 1e5, "bar")
_add("mass", 1, "kg", "kilogram", "kilograms")
_add("mass", 1e-3, "g", "gram", "grams")
_add("mass", 1e-6, "mg", "milligram", "milligrams")
_add("mass", 1e3, "t", "tonne", "tonnes")
_add("velocity", 1, "m/s", "m s^-1", "m·s^-1", "m s-1")
_add("velocity", 1 / 3.6, "km/h", "kph", "km/hr")
_add("torque", 1, "N m", "N·m", "N*m", "Nm", "N-m")
_add("torque", 1e3, "kN m", "kN·m", "kNm")
_add("stiffness", 1, "N/m")
_add("stiffness", 1e3, "kN/m", "N/mm")
_add("molar_conc", 1, "mol/L", "M", "molar", "mol L^-1", "mol/l")
_add("molar_conc", 1e-3, "mM", "mmol/L", "millimolar")
_add("molar_conc", 1e-6, "uM", "µM", "μM", "umol/L", "µmol/L", "micromolar")
_add("molar_conc", 1e-9, "nM", "nanomolar")
_add("mass_conc", 1, "mg/mL", "g/L", "mg/ml", "g/l")
_add("mass_conc", 1e-3, "ug/mL", "µg/mL", "μg/mL", "mg/L", "mg/l")
_add("molar_flow", 1, "mol/min")
_add("molar_flow", 1 / 60, "mol/s")
_add("molar_flow", 60, "mol/h", "mol/hr")
_add("angular_speed", 1, "rpm", "RPM", "rev/min")
_add("information", 1, "bit", "bits")
_add("information", 8, "byte", "bytes", "B")
_add("percent", 1, "%", "percent", "per cent")
_add("bp", 1, "bp", "basis point", "basis points")

# Longest alias first so "mol/min" beats "mol", "MPa" beats "M".
_ALIASES = sorted(_UNITS, key=len, reverse=True)
# Words that mean "this is a pure number" when stated as the expected unit.
_DIMENSIONLESS = {"dimensionless", "probability", "portfolio weight", "purge fraction", "fraction"}


def unit_info(text):
    """Return (dim, factor) for a unit string, or None if not a physical unit."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"\s+(RMS|rms)$", "", t)          # "microvolts RMS"
    if t.lower() in _DIMENSIONLESS:
        return ("fraction", 1)
    return _UNITS.get(t) or _UNITS.get(t.lower())


def leading_unit(tail):
    """Longest known unit alias at the start of `tail` (word-bounded)."""
    tail = tail.lstrip()
    for a in _ALIASES:
        if tail.startswith(a):
            nxt = tail[len(a):len(a) + 1]
            if not nxt or not (nxt.isalnum() or nxt in "/^_"):
                return a
    return None


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

_NUM = re.compile(
    r"(?<![\w.])([-+−–]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+−–]?\.\d+)"
    r"(?:\s*[eE]\s*([-+−]?\d+))?"
    r"(?:\s*(?:x|×|\*|·)\s*10\s*\^?\s*\{?\s*([-+−]?\d+)\}?)?"
)
_ID = re.compile(r"\b[THZ][0-5]-\d{2}\b")
_ANSWER_MARK = re.compile(
    r"(?:final\s+answer|answer|result)\s*(?:is|=|:|≈|is\s+approximately)?\s*[:=]?",
    re.IGNORECASE)


def _to_float(m):
    s = m.group(1).replace(",", "").replace("−", "-").replace("–", "-")
    v = float(s)
    exp = m.group(2) or m.group(3)
    if exp:
        v *= 10 ** int(exp.replace("−", "-"))
    return v


def _clean(text):
    text = _ID.sub(" ", text)
    text = text.replace("**", "").replace("`", "").replace("$", "")
    text = re.sub(r"\\boxed\{([^}]*)\}", r"ANSWER: \1", text)
    text = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", text)
    return text


def _first_number_with_unit(text):
    m = _NUM.search(text)
    if not m:
        return None
    return _to_float(m), leading_unit(text[m.end():m.end() + 25])


def extract_number(answer, strict=False):
    """-> (value, unit_alias_or_None, how) or (None, None, reason)."""
    if isinstance(answer, bool):
        return None, None, "bool_not_number"
    if isinstance(answer, (int, float)):
        return (float(answer), None, "number") if math.isfinite(answer) else (None, None, "non_finite")
    if isinstance(answer, dict):
        if "answer" not in answer:
            return None, None, "no_answer_field"
        v, u, how = extract_number(answer["answer"], strict)
        return v, (u or answer.get("unit")), how
    if strict or not isinstance(answer, str):
        return None, None, "not_a_number"

    s = answer.strip()
    try:  # Hive may emit the JSON object as text
        return extract_number(json.loads(s), strict)
    except (ValueError, TypeError):
        pass
    for blob in re.findall(r"\{[^{}]*\"answer\"[^{}]*\}", s):
        try:
            return extract_number(json.loads(blob), strict)
        except ValueError:
            pass

    text = _clean(s)
    frac = re.fullmatch(r"\s*([-+]?\d+)\s*/\s*(\d+)\s*", text)
    if frac and int(frac.group(2)):
        return int(frac.group(1)) / int(frac.group(2)), None, "fraction"
    marks = list(_ANSWER_MARK.finditer(text))
    if marks:
        hit = _first_number_with_unit(text[marks[-1].end():])
        if hit:
            return hit[0], hit[1], "after_answer_marker"
    nums = list(_NUM.finditer(text))
    if not nums:
        return None, None, "no_number_found"
    m = nums[-1]
    return _to_float(m), leading_unit(text[m.end():m.end() + 25]), "last_number"


def extract_choice(answer, strict=False):
    """-> (label or None, how)."""
    if isinstance(answer, dict):
        answer = answer.get("answer")
    if not isinstance(answer, str):
        return None, "not_a_string"
    s = answer.strip().upper()
    if s in ("A", "B", "C", "D"):
        return s, "label"
    if strict:
        return None, "invalid_choice"
    try:
        return extract_choice(json.loads(answer), strict)
    except (ValueError, TypeError):
        pass
    text = _clean(answer)
    marks = list(_ANSWER_MARK.finditer(text))
    if marks:
        m = re.match(r"\s*(?:option|choice)?\s*[\(\[]?\s*([A-D])\b(?![\w'])", text[marks[-1].end():], re.I)
        if m:
            return m.group(1).upper(), "after_answer_marker"
    labels = set(re.findall(r"(?:^|[\s(\[])\(?([A-D])[\)\].:](?=\s|$)", text, re.M))
    labels |= set(re.findall(r"\b(?:option|choice)\s+([A-D])\b", text, re.I))
    labels = {l.upper() for l in labels}
    if len(labels) == 1:
        return labels.pop(), "single_label"
    return None, "ambiguous_choice" if labels else "no_label_found"


# ---------------------------------------------------------------------------
# grade()
# ---------------------------------------------------------------------------

def grade_detail(problem, answer, strict=False):
    """-> dict(score in [0,1] or None, status, parsed, detail)."""
    if problem["answer_type"] == "rubric":
        # Tier 5: no grading method yet. Not scored, not counted.
        return dict(score=None, status="ungraded", parsed="", detail="tier 5 not graded yet")
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return dict(score=0.0, status="missing", parsed="", detail="")
    kind = problem["answer_type"]

    if kind == "numeric":
        value, unit, how = extract_number(answer, strict)
        if value is None or not math.isfinite(value):
            return dict(score=0.0, status="invalid_numeric", parsed="", detail=how)
        expected = unit_info(problem["unit"])
        given = unit_info(unit) if unit else None
        if given and expected and given[0] != expected[0]:
            return dict(score=0.0, status="unit_mismatch", parsed=f"{value:g} {unit}",
                        detail=f"expected {problem['unit']}")
        if given and expected:
            value = value * given[1] / expected[1]
        ok = abs(value - problem["answer"]) <= problem["tolerance"]
        # float noise guard: conversions like 20 mA -> 0.02 A
        if not ok and given and expected and given[1] != expected[1]:
            ok = abs(value - problem["answer"]) <= problem["tolerance"] + 1e-12 * max(1.0, abs(value))
        parsed = f"{value:.10g} {problem['unit']}" + (f" (from {unit})" if unit else "")
        return dict(score=float(ok), status="correct" if ok else "incorrect",
                    parsed=parsed, detail=how)

    if kind == "choice":
        label, how = extract_choice(answer, strict)
        if label is None:
            return dict(score=0.0, status="invalid_choice", parsed="", detail=how)
        ok = label == problem["answer"]
        return dict(score=float(ok), status="correct" if ok else "incorrect", parsed=label, detail=how)

    raise ValueError(f"{problem['id']}: unknown answer_type {kind!r}")


def grade(problem, answer, strict=False):
    """Score one Hive answer against one benchmark problem.
    Returns 0..1, or None for tier 5 (ungraded)."""
    return grade_detail(problem, answer, strict)["score"]


# ---------------------------------------------------------------------------
# Batch + report
# ---------------------------------------------------------------------------

PROVISIONAL = {"missing"}


def grade_all(problems, responses, strict=False, only_answered_splits=True):
    by_id = {}
    for r in responses:
        if r["id"] in by_id:
            raise ValueError(f"duplicate response id {r['id']}")
        if r["id"] not in problems:
            raise ValueError(f"unknown response id {r['id']}")
        by_id[r["id"]] = r
    splits = {problems[i]["split"] for i in by_id} if only_answered_splits else set(SPLIT_ORDER)
    items = []
    for pid, p in problems.items():
        if p["split"] not in splits:
            continue
        r = by_id.get(pid)
        ans = None
        if r is not None:
            ans = r.get("answer")
            if p["answer_type"] == "numeric" and isinstance(ans, (int, float)) and r.get("unit") and not strict:
                ans = {"answer": ans, "unit": r["unit"]}
        d = grade_detail(p, ans, strict)
        items.append(dict(id=pid, split=p["split"], tier=p["tier"], domain=p["domain"],
                          weight=1.0 if p["tier"] == 0 else p["domain_multiplier"],
                          answer_type=p["answer_type"], key=p["answer"], **d))
        shown = "  -  " if d["score"] is None else f"{d['score']:.2f}"
        print(f"  {pid:6s} {d['status']:16s} {shown}", file=sys.stderr)
    return items


def summarize(items):
    report = {}
    for split in SPLIT_ORDER:
        group = [i for i in items if i["split"] == split]
        if not group:
            continue
        tiers = {}
        for t in sorted({i["tier"] for i in group}):
            g = [i for i in group if i["tier"] == t]
            if any(i["status"] == "ungraded" for i in g):
                tiers[t] = dict(n=len(g), ungraded=True)
                continue
            w = sum(i["weight"] for i in g)
            tiers[t] = dict(ungraded=False, 
                n=len(g),
                answered=sum(i["status"] != "missing" for i in g),
                correct_or_points=sum(i["score"] for i in g),
                weighted_pct=100 * sum(i["weight"] * i["score"] for i in g) / w,
                unweighted_pct=100 * sum(i["score"] for i in g) / len(g))
        expected = {0} if split == "tier0" else {1, 2, 3, 4, 5}
        graded = {t: v for t, v in tiers.items() if not v["ungraded"]}
        flagged = [i["id"] for i in group if i["status"] in PROVISIONAL or i["status"].startswith("invalid") or i["status"] == "unit_mismatch"]
        report[split] = dict(
            tiers=tiers,
            graded_tiers=sorted(graded),
            total_pct=sum(t["weighted_pct"] for t in graded.values()) / len(graded),
            total_unweighted_pct=sum(t["unweighted_pct"] for t in graded.values()) / len(graded),
            provisional=bool(set(tiers) != expected or any(i["status"] in PROVISIONAL for i in group)),
            flagged=flagged)
    return report


def render_table(report):
    lines = ["| Split | Tier | Items | Answered | Points | Weighted % | Unweighted % |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for split, s in report.items():
        for t, row in s["tiers"].items():
            if row["ungraded"]:
                lines.append(f"| {split} | {t} | {row['n']} | ungraded | ungraded | ungraded | ungraded |")
                continue
            lines.append(f"| {split} | {t} | {row['n']} | {row['answered']} | "
                         f"{row['correct_or_points']:.1f} | {row['weighted_pct']:.1f} | {row['unweighted_pct']:.1f} |")
        n = sum(s["tiers"][t]["n"] for t in s["graded_tiers"])
        which = "tiers " + ",".join(map(str, s["graded_tiers"]))
        tag = ", provisional" if s["provisional"] else ""
        lines.append(f"| **{split}** | **total ({which}{tag})** | **{n}** | | | "
                     f"**{s['total_pct']:.1f}** | **{s['total_unweighted_pct']:.1f}** |")
    out = "\n".join(lines)
    out += ("\n\nWeighted % = domain-multiplier weighted (SCORING.md). Split total = mean of its graded tiers."
            "\nTier 5 is ungraded and excluded from totals."
            "\nmain / holdout / tier0 are separate scores - never merge them.")
    for split, s in report.items():
        if s["flagged"]:
            out += f"\n{split}: missing/invalid -> {', '.join(s['flagged'])}"
    return out


def main():
    ap = argparse.ArgumentParser(description="Grade Hive answers on Reasoning Benchmark v2")
    ap.add_argument("--answers", type=Path, required=True, help="JSONL of {id, answer[, unit]}")
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help="reasoning_benchmark dir")
    ap.add_argument("--strict", action="store_true",
                    help="official SCORING.md rules: bare number/label only, no unit conversion")
    ap.add_argument("--all-splits", action="store_true",
                    help="also score splits with no answers (all zero) instead of skipping them")
    ap.add_argument("--json", type=Path, help="write full report + per-item results")
    ap.add_argument("--csv", type=Path, help="write per-item results")
    args = ap.parse_args()

    problems = load_problems(args.bench)
    responses = read_jsonl(args.answers)
    items = grade_all(problems, responses, args.strict, not args.all_splits)
    report = summarize(items)
    print(render_table(report))

    if args.json:
        args.json.write_text(json.dumps(dict(report=report, items=items), indent=2,
                                        ensure_ascii=False, default=str), encoding="utf-8")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0]))
            w.writeheader()
            w.writerows(items)


if __name__ == "__main__":
    main()
