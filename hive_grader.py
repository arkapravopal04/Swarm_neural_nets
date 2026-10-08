"""
hive_grader.py -- automatic grader for Hive answers on Reasoning Benchmark v2.

    grade(problem, answer) -> float in [0, 1], or None (tier 5, ungraded)

Item kinds (from reasoning_benchmark/schema/problem.schema.json):

  numeric  (tiers 0-4)  answer is a number with `unit`, `tolerance`
                        (absolute, inclusive). We parse value + unit out of
                        the Hive answer (bare number, JSON, or prose), convert
                        same-dimension units (mA -> A, min -> hours,
                        uM -> mol/L ...) and test abs(value - key) <= tolerance.
                        Percent vs fraction is NOT converted (SCORING.md treats
                        them as distinct). A unit of another dimension scores
                        0 (unit_mismatch); a unit-like token the grader does
                        not know scores 0 and is flagged (unknown_unit).
  choice   (tiers 0-4)  label A/B/C/D, exact match. Conflicting answers
                        ("B or C", "7 or 8") score 0 (invalid_conflicting_answers).
  rubric   (tier 5)     scored only from evaluator-assigned `rubric_scores`
                        in the answers file (validated as scripts/grade.py
                        does). Without them the item is "ungraded_rubric":
                        score None, left out of every total, run provisional.

Prose extraction, in order: the JSON `answer` field (whole text or a JSON
object embedded in prose / a code fence); the text after the last "Answer:" /
"The answer is" / "Result =" / \\boxed{} marker (the final value of an `=`
chain, else the first number with the expected dimension); else the last
number with the expected dimension, ignoring parentheticals.

Scoring/aggregation follows reasoning_benchmark/SCORING.md:
  tier %   = 100 * sum(w_i * s_i) / sum(w_i)      (w = domain multiplier)
  total %  = mean of the graded tiers per split (main / holdout separately).
             While tier 5 is ungraded this is a tiers-1-4 figure, labelled as
             such and always provisional -- NOT the official 5-tier score.
  tier 0   = unweighted percent correct, reported on its own.

--strict reproduces scripts/grade.py: the `answer` field must be a finite JSON
number / a label string, no prose, no units, no nesting; ungraded tier-5 items
count 0 and every tier joins the total.

Answers file: JSONL (UTF-8, BOM ok), one object per line:
    {"id": "T1-03", "answer": "20 mA"}
    {"id": "T1-10", "answer": "The answer is (B)"}
    {"id": "T5-01", "answer": "...", "rubric_scores": {"C1": 2, ...}}
optional "unit" is used when `answer` carries no unit of its own.

Report files never contain answer keys.

Exit codes: 0 ok, 1 some answers were invalid / conflicting / had an unknown
unit / crashed the grader (see "needs review"), 2 bad input.

Usage:
    python hive_grader.py --answers hive_answers.jsonl
    python hive_grader.py --answers a.jsonl --json report.json --csv items.csv
"""

import argparse
import ast
import csv
import json
import math
import re
import sys
from collections import namedtuple
from pathlib import Path

DEFAULT_BENCH = Path(__file__).resolve().parent / "reasoning_benchmark"
KEY_FILES = ("tests/problems.jsonl", "tests/holdout.jsonl", "tests/tier0.jsonl")
SPLIT_ORDER = ("main", "holdout", "tier0")
EXPECTED_TIERS = {"main": {1, 2, 3, 4, 5}, "holdout": {1, 2, 3, 4, 5}, "tier0": {0}}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class InputError(ValueError):
    pass


def read_jsonl(path):
    rows = []
    # utf-8-sig: answers files saved on Windows often start with a BOM
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise InputError(f"{path}:{n}: bad JSON ({e})") from None
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise InputError(f"{path}:{n}: each line must be an object with a string \"id\"")
            rows.append(row)
    return rows


def load_problems(bench_dir=DEFAULT_BENCH):
    problems = {}
    for rel in KEY_FILES:
        for p in read_jsonl(Path(bench_dir) / rel):
            if p["id"] in problems:
                raise InputError(f"duplicate problem id {p['id']}")
            problems[p["id"]] = p
    return problems


# ---------------------------------------------------------------------------
# Units: alias -> (dimension, factor to that dimension's base unit)
# Count-like units in the bench ("rows", "socks", "currency units", ...) are
# not listed: when the expected unit is one of those, a plain word after the
# number is ignored.
# ---------------------------------------------------------------------------

_UNITS = {}


def _add(dim, factor, *aliases):
    for a in aliases:
        _UNITS[a] = (dim, factor)


_PREFIX = {"p": 1e-12, "n": 1e-9, "µ": 1e-6, "u": 1e-6, "m": 1e-3, "c": 1e-2,
           "d": 1e-1, "h": 1e2, "k": 1e3, "M": 1e6, "G": 1e9}


def _add_prefixed(dim, base_factor, symbol, prefixes):
    _add(dim, base_factor, symbol)
    for p in prefixes:
        _add(dim, base_factor * _PREFIX[p], p + symbol)


# SI symbols with the prefixes that occur in practice (a short list per base
# keeps "mPa"/"MPa", "mW"/"MW" ... from colliding in the lowercase fallback).
_add_prefixed("current", 1, "A", "nµumk")
_add_prefixed("voltage", 1, "V", "µumkM")
_add_prefixed("power", 1, "W", "µumkMG")
_add_prefixed("energy", 1, "J", "µumkMG")
_add_prefixed("energy", 1.602176634e-19, "eV", "kMG")
_add_prefixed("pressure", 1, "Pa", "hkMG")
_add_prefixed("frequency", 1, "Hz", "kMG")
_add_prefixed("force", 1, "N", "mkM")
_add_prefixed("resistance", 1, "Ω", "mkM")
_add_prefixed("mass", 1e-3, "g", "nµumkM")
_add_prefixed("time", 1, "s", "nµm")
_add_prefixed("length", 1, "m", "nµmck")
_add_prefixed("volume", 1e-3, "L", "µumcd")
_add_prefixed("amount", 1, "mol", "nµumk")
_add_prefixed("molar_conc", 1, "M", "pnµum")

_add("current", 1, "amp", "amps", "ampere", "amperes")
_add("current", 1e-3, "milliamp", "milliamps", "milliampere", "milliamperes")
_add("current", 1e-6, "microamp", "microamps", "microampere", "microamperes")
_add("voltage", 1, "volt", "volts")
_add("voltage", 1e-3, "millivolt", "millivolts")
_add("voltage", 1e-6, "microvolt", "microvolts")
_add("voltage", 1e3, "kilovolt", "kilovolts")
_add("power", 1, "watt", "watts")
_add("power", 1e-3, "milliwatt", "milliwatts")
_add("power", 1e3, "kilowatt", "kilowatts")
_add("power", 1e6, "megawatt", "megawatts")
_add("energy", 1, "joule", "joules")
_add("energy", 1e-3, "millijoule", "millijoules")
_add("energy", 1e3, "kilojoule", "kilojoules")
_add("energy", 1e6, "megajoule", "megajoules")
_add("energy", 3600, "Wh", "W·h", "W h", "watt-hour", "watt-hours")
_add("energy", 3.6e6, "kWh", "kW·h", "kW h", "kilowatt-hour", "kilowatt-hours")
_add("power", 1, "VA", "J/s")
_add("power", 1e3, "kVA", "kJ/s")
_add("power", 1e6, "MVA")
_add("energy", 4.184, "cal")
_add("energy", 4184, "kcal")
_add("pressure", 1, "pascal", "pascals")
_add("pressure", 1e6, "N/mm^2", "N/mm2", "N mm^-2", "megapascal", "megapascals")
_add("pressure", 1, "N/m^2", "N/m2", "N m^-2")
_add("pressure", 1e3, "kN/m^2", "kN/m2")
_add("pressure", 1e6, "MN/m^2", "MN/m2")
_add("pressure", 6894.757293168, "psi")
_add("pressure", 1e5, "bar")
_add("pressure", 101325, "atm")
_add("frequency", 1, "hertz")
_add("force", 1, "newton", "newtons")
_add("force", 1e3, "kilonewton", "kilonewtons")
_add("resistance", 1, "ohm", "ohms", "Ohm", "Ohms")
_add("resistance", 1e3, "kohm", "kohms", "kiloohm", "kiloohms")
_add("mass", 1, "kilogram", "kilograms")
_add("mass", 1e-3, "gram", "grams")
_add("mass", 1e-6, "milligram", "milligrams")
_add("mass", 1e3, "t", "tonne", "tonnes")
_add("mass", 0.45359237, "lb", "lbs", "pound", "pounds")
_add("mass", 0.028349523125, "oz", "ounce", "ounces")
_add("time", 1, "sec", "secs", "second", "seconds")
_add("time", 1e-3, "msec", "millisecond", "milliseconds")
_add("time", 60, "min", "mins", "minute", "minutes")
_add("time", 3600, "h", "hr", "hrs", "hour", "hours")
_add("time", 86400, "day", "days")
_add("time", 604800, "week", "weeks")
_add("length", 1, "metre", "metres", "meter", "meters")
_add("length", 1e3, "kilometre", "kilometres", "kilometer", "kilometers")
_add("length", 0.3048, "ft", "foot", "feet")
_add("length", 1609.344, "mi", "mile", "miles")
_add("volume", 1e-3, "l", "litre", "litres", "liter", "liters")
_add("volume", 1e-6, "ml", "millilitre", "millilitres", "milliliter", "milliliters")
_add("velocity", 1, "m/s", "m s^-1", "m·s^-1", "m s-1", "m/sec", "m/second",
     "metres/second", "meters/second")
_add("velocity", 1e3, "km/s", "km s^-1", "km·s^-1")
_add("velocity", 1e-2, "cm/s")
_add("velocity", 1 / 3.6, "km/h", "kph", "km/hr", "km/hour", "km h^-1")
_add("velocity", 0.44704, "mph")
_add("torque", 1, "N m", "N·m", "N*m", "Nm", "N-m", "newton-metre", "newton-metres",
     "newton-meter", "newton-meters", "newton metres", "newton meters")
_add("torque", 1e3, "kN m", "kN·m", "kNm", "kN-m")
_add("stiffness", 1, "N/m", "N m^-1", "N·m^-1")
_add("stiffness", 1e3, "kN/m", "N/mm")
_add("molar_conc", 1, "molar")
_add("molar_conc", 1e-3, "millimolar")
_add("molar_conc", 1e-6, "micromolar")
_add("molar_conc", 1e-9, "nanomolar")
# amount / volume in every spelling: "µmol/L", "mol·L^-1", "moles/litre", "mmol dm-3" ...
for _amt, _f in (("", 1), ("m", 1e-3), ("µ", 1e-6), ("u", 1e-6), ("n", 1e-9)):
    _words = [_amt + "mol"] + {"": ["mole", "moles"], "m": ["millimole", "millimoles"],
                               "µ": ["micromole", "micromoles"], "n": ["nanomole", "nanomoles"]}.get(_amt, [])
    for _w in _words:
        for _vol in ("L", "l", "litre", "liter", "dm^3", "dm3"):
            _add("molar_conc", _f, f"{_w}/{_vol}")
        for _vol in ("L", "l", "dm"):
            _e = "3" if _vol == "dm" else ""
            for _sep in (" ", "·"):
                _add("molar_conc", _f, f"{_w}{_sep}{_vol}^-{_e or 1}", f"{_w}{_sep}{_vol}-{_e or 1}")
# mass / volume: "mg/mL", "mg·mL^-1", "g/L", "kg/m3", "milligrams/milliliter" ...
for _m, _mf in (("kg", 1e3), ("g", 1), ("mg", 1e-3), ("µg", 1e-6), ("ug", 1e-6),
                ("gram", 1), ("grams", 1), ("milligram", 1e-3), ("milligrams", 1e-3),
                ("microgram", 1e-6), ("micrograms", 1e-6)):
    for _v, _vf in (("L", 1), ("l", 1), ("litre", 1), ("liter", 1), ("dL", 0.1),
                    ("mL", 1e-3), ("ml", 1e-3), ("millilitre", 1e-3), ("milliliter", 1e-3),
                    ("cm^3", 1e-3), ("cm3", 1e-3), ("cc", 1e-3), ("m^3", 1e3), ("m3", 1e3)):
        _f = _mf / _vf          # base: g/L == mg/mL
        _add("mass_conc", _f, f"{_m}/{_v}")
        if _v in ("L", "l", "mL", "ml", "dL"):
            for _sep in (" ", "·"):
                _add("mass_conc", _f, f"{_m}{_sep}{_v}^-1", f"{_m}{_sep}{_v}-1")
# base: mol/min (the bench's unit). 1 mol/s = 60 mol/min; 1 mol/h = 1/60 mol/min.
_add("molar_flow", 1, "mol/min", "mol/minute", "mol min^-1", "mol·min^-1")
_add("molar_flow", 60, "mol/s", "mol/sec", "mol/second", "mol s^-1", "mol·s^-1")
_add("molar_flow", 1 / 60, "mol/h", "mol/hr", "mol/hour", "mol h^-1", "mol·h^-1")
_add("molar_flow", 60e-3, "mmol/s")
_add("molar_flow", 1e3, "kmol/min")
_add("molar_flow", 1e3 / 60, "kmol/h", "kmol/hr")
# rotation rates share the frequency dimension: 7.5 Hz == 450 rpm
_add("frequency", 1 / 60, "rpm", "RPM", "rev/min", "revolutions/minute", "r/min", "min^-1")
_add("frequency", 1, "rev/s", "rps", "s^-1", "1/s")
_add("frequency", 1 / (2 * math.pi), "rad/s", "rad s^-1")
_add("angle", 1, "deg", "degree", "degrees", "°")
_add("angle", 180 / math.pi, "rad", "radian", "radians")
_add("temperature", 1, "K", "kelvin", "°C", "degC", "°F", "degF")  # dimension only; no offsets
_add("area", 1, "m^2", "m2")
_add("area", 1e-6, "mm^2", "mm2")
_add("information", 1, "bit", "bits")
_add("information", 8, "byte", "bytes")       # "B" is not bytes: too often a label
_add("information", 8e3, "kB", "kilobyte", "kilobytes")
_add("percent", 1, "%", "percent", "pct")
_add("permille", 1, "‰", "per mille", "permille")
_add("bp", 1, "bp", "bps", "basis point", "basis points")

# Spelled-out prefixed units: "microjoules", "kilopascal", "nanoamps" ...
for _word, _dim_, _bf in (("amp", "current", 1), ("ampere", "current", 1), ("volt", "voltage", 1),
                          ("watt", "power", 1), ("joule", "energy", 1), ("pascal", "pressure", 1),
                          ("newton", "force", 1), ("gram", "mass", 1e-3), ("second", "time", 1),
                          ("metre", "length", 1), ("meter", "length", 1), ("litre", "volume", 1e-3),
                          ("liter", "volume", 1e-3), ("hertz", "frequency", 1), ("ohm", "resistance", 1),
                          ("mole", "amount", 1)):
    for _pw, _pf in (("nano", 1e-9), ("micro", 1e-6), ("milli", 1e-3), ("kilo", 1e3),
                     ("mega", 1e6), ("giga", 1e9)):
        for _plural in ("", "s"):
            if _word == "hertz" and _plural:
                continue
            _UNITS.setdefault(_pw + _word + _plural, (_dim_, _bf * _pf))

# Lowercase fallback ("KW", "MPA", "Hours", "hz") only where the lowercase form
# is unambiguous and longer than one character, and never across the m/M
# prefix ("MA" is not mA, "Mg" is not mg).
_lower = {}
for _a, _v in _UNITS.items():
    _lower.setdefault(_a.lower(), {})[_v] = _a
_UNITS_CI = {k: next(iter(v.items())) for k, v in _lower.items() if len(v) == 1 and len(k) > 1}


def _ci_lookup(t):
    hit = _UNITS_CI.get(t.lower())
    if not hit:
        return None
    info, alias = hit
    # m/M is a prefix only when the rest is itself a unit: "MA" is not mA,
    # but "Mol/L" is mol/L
    if t[0] != alias[0] and t[0] in "mM" and alias[1:] in _UNITS:
        return None
    return info

# Longest alias first so "mol/min" beats "mol", "MPa" beats "M".
_ALIASES = sorted(_UNITS, key=len, reverse=True)
# Words that mean "this is a pure number" when stated as the expected unit.
_DIMENSIONLESS = {"dimensionless", "probability", "portfolio weight", "purge fraction", "fraction"}
_RATIO_DIMS = {"fraction", "percent", "permille", "bp"}


def unit_info(text):
    """Return (dim, factor) for a unit string, or None if not a known physical unit."""
    if not text:
        return None
    t = text.strip().replace("μ", "µ").replace("⋅", "·")
    t = re.sub(r"\s*(RMS|rms)$", "", t) or t     # "microvolts RMS", "Vrms"
    if t.lower() in _DIMENSIONLESS:
        return ("fraction", 1)
    return _UNITS.get(t) or _ci_lookup(t)


_UNIT_TOKEN = re.compile(r"[A-Za-zµΩ°%][A-Za-z0-9µΩ°]*(?:[/·*^][A-Za-z0-9µΩ°^\-]+)*(?![\w\-])")
_SENTENCE_WORDS = {"The", "This", "That", "It", "In", "So", "And", "If", "For", "At", "As",
                   "To", "We", "I", "Then", "Thus", "Hence", "Note", "But", "Or", "On", "By"}


def _looks_like_unit(tok):
    """Unit-shaped token the grader doesn't know ("kΩ", "USD", "mm^3")?
    Lowercase words ("through", "via") are prose, not units."""
    if any(c in tok for c in "µΩ°/·^*%"):
        return True
    return any(c.isupper() for c in tok) and len(tok) <= 4 and tok not in _SENTENCE_WORDS


def leading_unit(tail):
    """Unit at the start of `tail`: a known alias (word-bounded), else an
    unknown unit-shaped token, else None."""
    tail = tail.lstrip()
    # "mol per minute", "mg / mL" -- only when that makes a known unit
    joined = re.sub(r"^([^\s/]+)(?:\s+per\s+|\s*/\s*)(?=\w)", r"\1/", tail)
    if joined != tail:
        m = _UNIT_TOKEN.match(joined)
        if m and unit_info(m.group(0)):
            tail = joined
    for a in _ALIASES:
        if tail.startswith(a):
            nxt = tail[len(a):len(a) + 1]
            # "kW·h" must not read as kW, nor "mol/x" as mol
            if not nxt or not (nxt.isalnum() or nxt in "/^_·*-µΩ°"):
                return a
    m = _UNIT_TOKEN.match(tail)
    if not m:
        return None
    tok = m.group(0)
    if unit_info(tok):
        return tok
    return tok if _looks_like_unit(tok) else None


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_ID = re.compile(r"\b[THZ][0-5]-\d{2}\b")
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_LATEX_SYMBOLS = {
    r"\times": " × ", r"\cdot": "·", r"\approx": " ≈ ", r"\simeq": " ≈ ", r"\mu": "µ",
    r"\Omega": "Ω", r"\%": "%", r"\pm": " ± ", r"\div": " ÷ ", r"\le": " ≤ ", r"\ge": " ≥ ",
    r"\left": "", r"\right": "", r"\displaystyle": "", r"\$": "",
}


def _unbrace(text, start):
    """text[start] == "{" -> (content, index after the matching "}")."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return text[start + 1:], len(text)


def _replace_command(text, names, fmt, nargs=1):
    pat = re.compile(r"\\(?:%s)\s*(?=\{)" % "|".join(names))
    while True:
        m = pat.search(text)
        if not m:
            return text
        args, end = [], m.end()
        for _ in range(nargs):
            while end < len(text) and text[end] == " ":
                end += 1
            if end >= len(text) or text[end] != "{":
                break
            arg, end = _unbrace(text, end)
            args.append(arg)
        if len(args) < nargs:   # malformed: drop the command name, keep the rest
            text = text[:m.start()] + text[m.end():]
            continue
        text = text[:m.start()] + fmt(*args) + text[end:]


def _frac(num, den):
    num, den = num.strip(), den.strip()
    simple = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", num) and re.fullmatch(r"\d+(?:\.\d+)?", den)
    return f" {num}/{den} " if simple else f" ({num})/({den}) "


def _sqrt(arg):
    """\\sqrt{140.5} -> its value; anything symbolic is left as text."""
    a = arg.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", a):
        return f" {math.sqrt(float(a))!r} "
    return f" sqrt({a}) "


def _clean(text):
    text = text.replace("μ", "µ").replace("⋅", "·")
    text = re.sub(r"[\u00a0\u2009\u202f\u2002\u2003]", " ", text)
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^" + m.group(0).translate(_SUPER), text)
    text = _ID.sub(" ", text)
    text = re.sub(r"(?<=\d) *\*\* *(?=[-+−(]*\d)", "^", text)                       # 10**-4
    text = re.sub(r"\^\s*\(\s*([-+−]?\d+)\s*\)", r"^\1", text)                         # 10^(-4)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    # LaTeX
    text = _replace_command(text, ["boxed", "fbox"], lambda a: f"\nANSWER: {a}\n")
    text = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", r" \1/\2 ", text)          # \frac13
    text = _replace_command(text, ["frac", "dfrac", "tfrac"], _frac, nargs=2)
    text = _replace_command(text, ["sqrt"], _sqrt)
    for _ in range(3):  # nested \text{\mu A}
        text = _replace_command(text, ["text", "mathrm", "textrm", "textbf", "mathbf",
                                       "operatorname", "mbox", "unit", "si"], lambda a: a)
    text = re.sub(r"(?<=\d)\\[,;:!]\s*(?=\d{3}\b)", "", text)      # 1\,000
    text = re.sub(r"\\[,;:! ]|~", " ", text)                     # thin spaces
    for k, v in _LATEX_SYMBOLS.items():
        text = re.sub(re.escape(k) + r"(?![A-Za-z])", lambda _m, v=v: v, text)
    text = text.replace("^\\circ", "°").replace("^{\\circ}", "°")
    text = re.sub(r"\\[()\[\]]", " ", text)
    text = text.replace("$", "")
    text = re.sub(r"µ\s+(?=[A-Za-z])", "µ", text)                 # "µ A" -> "µA"
    text = re.sub(r"\bper\s*cent\b", "%", text, flags=re.I)
    text = re.sub(r"\s*(?:±|\+/-|\+-)\s*\d+(?:\.\d+)?", " ", text)     # 20 ± 1 mA -> 20 mA
    text = re.sub(r"(?<![\w*])_(\S(?:[^_\n]*\S)?)_(?!\w)", r"\1", text)  # _10_
    return text


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

_NUM = re.compile(
    r"(?<![\w.^{/])(?<![\^{/][-+−–])"            # not inside a word / decimal / unit power
    r"(?P<sign>[-+−–]?)"
    r"(?:"
    r"(?P<fn>\d+(?:\.\d+)?)\s*/\s*(?P<fd>\d+(?:\.\d+)?)(?![\d.]|\s*/\s*\d)"   # 1/3
    r"|10\s*\^\s*\{?\s*(?P<tp>[-+−]?\d+)\s*\}?"                             # 10^3
    r"|(?P<m>(?:\d{1,3}(?:,\d{3})+|\d{1,3}(?: \d{3})+(?![\d.,])|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE](?P<e>[-+−]?\d+)"
    r"|\s*(?:x|×|\*|·)\s*10\s*(?:\^\s*[{(]?\s*(?P<p>[-+−]?\d+)\s*[})]?|(?P<p2>[-−]\d+)))?"  # ×10−4
    r")(?![\w.]*\d)(?!(?:st|nd|rd|th)\b)"
)

Tok = namedtuple("Tok", "value unit start end")


def _to_float(m):
    sign = "-" if m.group("sign") in ("-", "−", "–") else ""
    if m.group("fn") is not None:
        den = float(m.group("fd"))
        return float("nan") if den == 0 else float(sign + m.group("fn")) / den
    if m.group("tp") is not None:
        return float(f"{sign}1e{m.group('tp').replace('−', '-')}")
    s = sign + m.group("m").replace(",", "").replace(" ", "")
    exp = m.group("e") or m.group("p") or m.group("p2")
    if exp:
        s += "e" + exp.replace("−", "-")
    return float(s)          # "7e400" -> inf, never OverflowError


def _tokens(text):
    out = []
    for m in _NUM.finditer(text):
        out.append(Tok(_to_float(m), leading_unit(text[m.end():m.end() + 30]), m.start(), m.end()))
    return out


def _dim(unit):
    info = unit_info(unit)
    return info[0] if info else None


def _matches(tok, expected_dim):
    return expected_dim is not None and _dim(tok.unit) == expected_dim


_ANSWER_MARK = re.compile(
    r"(?<!as a )\b(?:final\s+answer|final\s+result|answer|result|option|choice|correct(?=\s*:))\b"
    r"[ \t]*(?::|=|≈|\bis\b|\bwas\b|\n|—|–|-(?=\s))"
    r"(?:[ \t]*(?:[:=]|approximately|approx\.?|about|roughly|exactly|equal\s+to|that|then)(?!\w))*",
    re.IGNORECASE)
_SEG_END = re.compile(r"\n|;|\.(?=\s|$)")
_JOIN = re.compile(r"(?P<unit>\s*[^\s\d=≈,;]{1,12})?\s*,?\s*(?:\b(?P<w>or|and|to)\b|–|—)\s*|-")


def _segment(text, start):
    """The sentence/line that follows an answer marker."""
    while start < len(text) and text[start] in " \t\r\n:":
        start += 1
    m = _SEG_END.search(text, start)
    return text[start:m.start() if m else len(text)]


def _marker_value(text, mark, expected_dim, weak=False):
    """Tok the marker points at, "conflict", or None. A weak marker
    ("result was ...") must be followed straight away by the value."""
    seg = _without_asides(_segment(text, mark.end()))
    if weak and not re.match(r"\s*(?:[A-Za-z]\w*\s*[=≈]\s*)?[-+−]?\.?\d", seg):
        return None
    tok = _pick_in_segment(seg, expected_dim)
    if tok and _conflicting(seg, _tokens(seg), tok):
        return "conflict"
    return tok


def _without_asides(text):
    """Drop "(...)" and "[...]" asides, unless that removes every number."""
    t, prev = text, None
    while prev != t:
        prev = t
        t = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", t)
    return t if _NUM.search(t) else text


def _same_quantity(a, b):
    """One value restated: "0.4625 or 46.25%", "3 h or 180 min"."""
    def base(t):
        info = unit_info(t.unit)
        if info and info[0] == "percent":
            return t.value / 100, None
        return (t.value * info[1], info[0]) if info else (t.value, None)
    (va, da), (vb, db) = base(a), base(b)
    if da != db and da and db:
        return False
    return math.isclose(va, vb, rel_tol=1e-6, abs_tol=1e-12)


def _conflicting(text, toks, chosen):
    """"7 or 8", "15-20 mA", "between 15 and 20 mA": two candidate values for
    one answer. "20 mA and 12 V" is two quantities, not a conflict."""
    i = toks.index(chosen)
    for a, b in ((toks[i - 1], chosen) if i else (None, None), (chosen, toks[i + 1]) if i + 1 < len(toks) else (None, None)):
        if a is None or _same_quantity(a, b):
            continue
        m = _JOIN.fullmatch(text, a.end, b.start)
        if not m:
            continue
        if m.group("w") == "and":
            between = re.search(r"\bbetween\s*$", text[:a.start], re.I)
            if not between and a.unit != b.unit:
                continue
        elif a.unit and b.unit and _dim(a.unit) != _dim(b.unit):
            continue
        return True
    return False


def _pick_in_segment(seg, expected_dim):
    """Value an answer marker points at: the final value of an `=` chain
    (the last one with the expected dimension), else the first number with
    the expected dimension, else the first number."""
    toks = _tokens(seg)
    if not toks:
        return None
    seps = list(re.finditer(r"=|≈|≃", seg))
    if not seps:
        return next((t for t in toks if _matches(t, expected_dim)), toks[0])
    bounds = [0] + [m.end() for m in seps] + [len(seg) + 1]
    heads = []          # first number of each part of the chain
    for k in range(len(bounds) - 1):
        part = [t for t in toks if bounds[k] <= t.start < bounds[k + 1]]
        if part:
            heads.append((k, part))
    k, part = next(((k, p) for k, p in reversed(heads) if _matches(p[0], expected_dim)), heads[-1])
    chosen = part[0]
    if k >= 1 and seps[k - 1].group(0) != "=":
        # "1/3 ≈ 0.333": the exact value before the approximation wins
        prev = [p for kk, p in heads if kk == k - 1]
        lead = seg[bounds[k - 1]:prev[0][0].start] if prev else ""
        if prev and len(prev[0]) == 1 and re.fullmatch(r"\s*[^\d\s]{0,3}\s*", lead) \
                and (expected_dim is None or not chosen.unit or _matches(prev[0][0], expected_dim)
                     or not prev[0][0].unit):
            chosen = prev[0][0]
    return chosen


_NOISE = re.compile(
    r"\b(?:to|at|with)\s+\d+\s+(?:significant\s+(?:figures?|digits?)|sig\.?\s*figs?|s\.?f\.?"
    r"|decimal\s+places?|d\.?p\.?)"
    r"|\b(?:step|eq(?:uation)?\.?|line|item|case|part|figure|fig\.?|table|section)\s*\d+"
    r"|^\s*\d+[.)][ \t]+(?=\S)",
    re.IGNORECASE | re.MULTILINE)


def _strip_asides(text):
    """For the last-number fallback: drop parentheticals, "to 3 significant
    figures", "step 2", list numbering."""
    return _without_asides(_NOISE.sub(" ", text))


def _json_answer(s):
    """`answer` value from the whole text as JSON, or the last JSON object
    with an "answer" key embedded in prose / a code fence. _MISS if none."""
    try:
        return json.loads(s)
    except (ValueError, TypeError, RecursionError):
        pass
    found = _MISS
    if not re.search(r"[\"']answer[\"']", s):
        return found
    dec = json.JSONDecoder()
    for i in [m.start() for m in re.finditer(r"\{", s)][:200]:
        try:
            v, _ = dec.raw_decode(s, i)    # handles nested {"notes": {...}}
        except (ValueError, RecursionError):
            continue
        if isinstance(v, dict) and "answer" in v:
            found = v
    if found is _MISS:
        for blob in re.findall(r"\{[^{}]*'answer'[^{}]*\}", s):
            try:   # {'answer': 'B'} -- Python-dict style
                v = ast.literal_eval(blob)
                found = v if isinstance(v, dict) else found
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                pass
    return found


_MISS = object()
# keys accepted for the value in non-strict mode, in order of preference
_VALUE_KEYS = ("answer", "final_answer", "result", "value")


def _finite(v):
    try:
        return math.isfinite(float(v))
    except (OverflowError, ValueError, TypeError):
        return False


def extract_number(answer, strict=False, expected_unit=None):
    """-> (value, unit_or_None, how) or (None, None, reason)."""
    if isinstance(answer, bool):
        return None, None, "bool_not_number"
    if isinstance(answer, (int, float)):
        return (float(answer), None, "number") if _finite(answer) else (None, None, "non_finite")
    if strict:
        # scripts/grade.py: the answer field itself must be a finite JSON number
        return None, None, "not_a_number"
    if isinstance(answer, dict):
        key = next((k for k in _VALUE_KEYS if k in answer), None)
        if key is None:
            return None, None, "no_answer_field"
        v, u, how = extract_number(answer[key], strict, expected_unit)
        hint = next((answer[k] for k in ("unit", "units") if isinstance(answer.get(k), str)), None)
        return v, (u or hint), how
    if isinstance(answer, list):
        # [20] or [20, "mA"]
        if len(answer) == 1 or (len(answer) == 2 and isinstance(answer[1], str)):
            v, u, how = extract_number(answer[0], strict, expected_unit)
            return v, u or (answer[1] if len(answer) == 2 else None), how
        return None, None, "not_a_number"
    if not isinstance(answer, str):
        return None, None, "not_a_number"

    s = answer.strip()
    j = _json_answer(s)
    if j is not _MISS and not isinstance(j, str):
        return extract_number(j, strict, expected_unit)

    text = _clean(s)
    exp = unit_info(expected_unit)
    expected_dim = exp[0] if exp else None

    # "Final answer" / "Answer" / \boxed markers outrank "result" / "option"
    # ones, so "Final answer: 20 mA. Result is verified (3 runs)." reads 20.
    marks = list(_ANSWER_MARK.finditer(text))
    strong = [m for m in marks if re.match(r"final|answer\b", m.group(0), re.I)]
    weak = [m for m in marks if m not in strong]
    for group, is_weak in ((strong, False), (weak, True)):
        for mark in reversed(group):
            tok = _marker_value(text, mark, expected_dim, is_weak)
            if tok == "conflict":
                return None, None, "conflicting_answers"
            if tok:
                return tok.value, tok.unit, "after_answer_marker"

    body = _strip_asides(text)
    toks = _tokens(body)
    if not toks:
        return None, None, "no_number_found"
    dim_toks = [t for t in toks if _matches(t, expected_dim)]
    tok = (dim_toks or toks)[-1]
    if _conflicting(body, toks, tok):
        return None, None, "conflicting_answers"
    return tok.value, tok.unit, "last_number"


# ---------------------------------------------------------------------------
# Choice extraction
# ---------------------------------------------------------------------------

# "A" followed by one of these is the label, not the article ("A is correct").
_A_VERBS = {"is", "was", "would", "should", "seems", "holds", "fits", "matches", "best",
            "and", "or", "because", "since", "as", "only", "follows", "must", "can",
            "could", "remains", "satisfies", "applies", "gives", "states", "says", "alone",
            "here", "wins", "correctly", "describes", "preserves", "reflects"}


_NEG_BEFORE = re.compile(r"(?:\b(?:not|nor|neither|except|excluding|than|isn't|eliminate|"
                         r"eliminating|rule\s+out|ruling\s+out)\s+(?:option\s+|choice\s+)?[(\[]?\s*)$",
                         re.IGNORECASE)
_NEG_AFTER = re.compile(r"\s*[)\]]?(?:\s*(?:,|and|or)\s*(?:option\s+)?[(\[]?[A-D][)\]]?)*\s+"
                        r"(?:is|are|was|were|seems?|looks?)\s+(?:also\s+)?"
                        r"(?:wrong|false|incorrect|invalid|not|ruled\s+out|eliminated|excluded|a\s+trap)\b",
                        re.IGNORECASE)


def _negated(text, s, e):
    return bool(_NEG_BEFORE.search(text[max(0, s - 30):s]) or _NEG_AFTER.match(text, e))


def _labels(seg, valid, keep_negated=False, lower_ok=True):
    """Standalone choice labels in order: (label, start, end). Negated ones
    ("not A", "A and B are wrong") are dropped."""
    out = []
    for m in re.finditer(r"(?<![\w\-/])([A-Da-d])(?![\w\-/])(?!['’][a-z])", seg):
        lab, s, e = m.group(1), m.start(1), m.end(1)
        if lab.islower():
            if not lower_ok:
                continue
            # lowercase only as "(b)" / "b)" at the start / the whole segment
            if not (re.match(r"[\(\[]", seg[s - 1:s]) and re.match(r"[\)\]]", seg[e:e + 1])) \
                    and not (seg[:s].strip(" \t*") == "" and seg[e:e + 1] == ")") \
                    and seg.strip(" \t.()[]*:'\"") != lab:
                continue
        elif lab == "A":
            nxt = re.match(r"\s+([a-z]+)", seg[e:])
            if nxt and nxt.group(1) not in _A_VERBS:
                continue                       # article: "A bit tricky"
        if not keep_negated and _negated(seg, s, e):
            continue
        if lab.upper() in valid:
            out.append((lab.upper(), s, e))
    return out


def _label_in_segment(seg, valid):
    """-> (label, how) / (None, "conflicting_answers") / None."""
    labs = _labels(seg, valid)
    if not labs:
        return None
    first = labs[0]
    if re.match(r"\s*\)?\s*(?:or|and|/|&)\s*\(?\s*[A-D]\b", seg[first[2]:]):
        return None, "conflicting_answers"
    if re.fullmatch(r"[\s(\[]*(?:option|choice)?[\s(\[]*", seg[:first[1]], re.I):
        return first[0], "after_answer_marker"
    distinct = {l for l, _, _ in labs}
    if len(distinct) == 1:
        return first[0], "after_answer_marker"
    return None


def extract_choice(answer, strict=False, choices=None):
    """-> (label or None, how)."""
    valid = tuple(sorted(choices)) if choices else ("A", "B", "C", "D")
    if strict:
        # scripts/grade.py: the answer field itself must be a label string
        if isinstance(answer, str) and answer.strip().upper() in valid:
            return answer.strip().upper(), "label"
        return None, "invalid_choice"
    if isinstance(answer, dict):
        answer = next((answer[k] for k in _VALUE_KEYS + ("label", "choice") if k in answer), None)
    if not isinstance(answer, str):
        return None, "not_a_string"
    s = answer.strip()
    if s.upper() in valid:
        return s.upper(), "label"
    j = _json_answer(s)
    if j is not _MISS and (isinstance(j, dict) or (isinstance(j, str) and j != s)):
        return extract_choice(j, strict, choices)

    text = _clean(s)
    m = re.fullmatch(r"[\s(\[]*(?:(?:option|choice)\s+)?[\s(\[]*([A-Da-d])[\s)\].!]*", text, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "label"

    for mark in reversed(list(_ANSWER_MARK.finditer(text))):
        hit = _label_in_segment(_segment(text, mark.end()), valid)
        if hit:
            return hit

    # leading label: "B, because ...", "(C) The meeting ..."
    m = re.match(r"\s*[(\[]?\s*([A-D])\s*[)\]]?\s*(?:[.,:;\-—–]|$)(?!\s*\(?[A-D]\b)", text)
    if m and m.group(1) in valid and not _negated(text, m.start(1), m.end(1)):
        return m.group(1), "leading_label"

    explicit = set()
    for m in re.finditer(r"\b(?:option|choice|choose|chose|select|selected|pick|picked|go\s+with)"
                         r"\s+\(?([A-Da-d])\b|(?<![\w(])\(([A-D])\)", text, re.I):
        g = 1 if m.group(1) else 2
        if not _negated(text, m.start(g), m.end(g)) and m.group(g).upper() in valid:
            explicit.add(m.group(g).upper())
    if len(explicit) == 1:
        return explicit.pop(), "explicit_label"
    if len(explicit) > 1:
        return None, "conflicting_answers"

    labs = {l for l, _, _ in _labels(text, valid, lower_ok=False)}
    if len(labs) == 1:
        return labs.pop(), "single_label"
    if labs:
        return None, "conflicting_answers"

    # the option text quoted verbatim (long texts only; "Yes." is everywhere)
    if isinstance(choices, dict):
        low = re.sub(r"\s+", " ", text.lower())
        hits = {k for k, v in choices.items()
                if len(v) >= 12 and re.sub(r"\s+", " ", v.lower()).rstrip(".") in low}
        hits = {k for k in hits if not any(k != o and choices[k].lower().rstrip(".")
                                           in choices[o].lower() for o in hits)}
        if len(hits) == 1:
            return hits.pop(), "choice_text"
    return None, "no_label_found"


# ---------------------------------------------------------------------------
# grade()
# ---------------------------------------------------------------------------

def check_rubric_scores(problem, scores):
    """Same validation as scripts/grade.py; raises ValueError."""
    expected = {c["id"] for c in problem["rubric"]}
    if not isinstance(scores, dict) or set(scores) != expected:
        raise InputError(f"{problem['id']}: rubric_scores must contain exactly {sorted(expected)}")
    if any(type(v) is not int or v not in (0, 1, 2) for v in scores.values()):
        raise InputError(f"{problem['id']}: each evaluator-assigned criterion score must be integer 0, 1, or 2")


def grade_detail(problem, answer, strict=False, unit=None, rubric_scores=None):
    """-> dict(score in [0,1] or None, status, parsed, detail).
    Status names follow scripts/grade.py where it has one."""
    kind = problem["answer_type"]
    if kind == "rubric":
        if rubric_scores is not None:
            check_rubric_scores(problem, rubric_scores)
            return dict(score=sum(rubric_scores.values()) / problem["max_points"],
                        status="evaluator_graded", parsed=json.dumps(rubric_scores), detail="")
        # No grading method for tier 5 yet: strict counts it 0 like grade.py,
        # default mode leaves it out of every total.
        return dict(score=0.0 if strict else None,
                    status="missing" if answer is None else "ungraded_rubric", parsed="",
                    detail="needs evaluator rubric_scores")
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return dict(score=0.0, status="missing", parsed="", detail="")

    if kind == "numeric":
        value, given_unit, how = extract_number(answer, strict, problem["unit"])
        if how == "conflicting_answers":
            return dict(score=0.0, status="invalid_conflicting_answers", parsed="", detail=how)
        if value is None or not math.isfinite(value):
            return dict(score=0.0, status="invalid_numeric_answer", parsed="", detail=how)
        if strict:
            given_unit = None
        elif not given_unit and isinstance(unit, str):
            given_unit = unit
        expected = unit_info(problem["unit"])
        exp_dim = expected[0] if expected else None
        physical = exp_dim is not None and exp_dim not in _RATIO_DIMS
        given = unit_info(given_unit) if given_unit else None
        if given_unit and not given:
            if physical:
                return dict(score=0.0, status="unknown_unit", parsed=f"{value:g} {given_unit}",
                            detail=f"{how}; expected {problem['unit']}")
            given_unit = None        # a word after a count / fraction ("7 socks")
        elif given and not physical and given[0] not in _RATIO_DIMS:
            given, given_unit = None, None   # "7 s" on a count or probability item
        if given and given[0] != exp_dim:
            # incl. percent vs fraction, which SCORING.md keeps distinct
            return dict(score=0.0, status="unit_mismatch", parsed=f"{value:g} {given_unit}",
                        detail=f"{how}; expected {problem['unit']}")
        if given:
            value = value * given[1] / expected[1]
        try:
            ok = abs(value - problem["answer"]) <= problem["tolerance"]
            # float noise guard: conversions like 20 mA -> 0.02 A
            if not ok and given and expected and given[1] != expected[1]:
                ok = abs(value - problem["answer"]) <= problem["tolerance"] + 1e-12 * max(1.0, abs(value))
        except OverflowError:
            ok = False
        parsed = f"{value:.10g} {problem['unit']}" + (f" (from {given_unit})" if given_unit else "")
        return dict(score=float(ok), status="correct" if ok else "incorrect",
                    parsed=parsed, detail=how)

    if kind == "choice":
        label, how = extract_choice(answer, strict, problem.get("choices"))
        if label is None:
            status = "invalid_conflicting_answers" if how == "conflicting_answers" else "invalid_choice"
            return dict(score=0.0, status=status, parsed="", detail=how)
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

PENDING = {"missing", "ungraded_rubric"}
NEEDS_REVIEW = {"unknown_unit", "grader_error"}   # plus every invalid_*


def is_invalid(status):
    return status.startswith("invalid_") or status in NEEDS_REVIEW


ITEM_FIELDS = ("id", "split", "tier", "domain", "weight", "answer_type",
               "score", "status", "parsed", "detail", "answer")


def _raw(ans):
    if ans is None:
        return ""
    return ans if isinstance(ans, str) else json.dumps(ans, ensure_ascii=False)


def grade_all(problems, responses, strict=False, only_answered_splits=True, quiet=False):
    by_id = {}
    for r in responses:
        if r["id"] in by_id:
            raise InputError(f"duplicate response id {r['id']}")
        if r["id"] not in problems:
            raise InputError(f"unknown response id {r['id']}")
        by_id[r["id"]] = r
    for pid, r in by_id.items():   # fail fast, before any grading output
        if r.get("rubric_scores") is not None and problems[pid]["answer_type"] == "rubric":
            check_rubric_scores(problems[pid], r["rubric_scores"])
    splits = {problems[i]["split"] for i in by_id} if only_answered_splits else set(SPLIT_ORDER)
    items = []
    for pid, p in problems.items():
        if p["split"] not in splits:
            continue
        r = by_id.get(pid) or {}
        ans = r.get("answer")
        try:
            d = grade_detail(p, ans, strict, unit=r.get("unit"), rubric_scores=r.get("rubric_scores"))
        except InputError:
            raise
        except Exception as e:   # one odd answer must not sink the whole report
            d = dict(score=0.0, status="grader_error", parsed="", detail=f"{type(e).__name__}: {e}")
        items.append(dict(id=pid, split=p["split"], tier=p["tier"], domain=p["domain"],
                          weight=1.0 if p["tier"] == 0 else p["domain_multiplier"],
                          answer_type=p["answer_type"], **d, answer=_raw(ans)))
        if not quiet:
            shown = "  -  " if d["score"] is None else f"{d['score']:.2f}"
            print(f"  {pid:6s} {d['status']:28s} {shown}", file=sys.stderr)
    return items


def summarize(items, strict=False):
    report = {}
    for split in SPLIT_ORDER:
        group = [i for i in items if i["split"] == split]
        if not group:
            continue
        tiers = {}
        for t in sorted({i["tier"] for i in group}):
            g = [i for i in group if i["tier"] == t]
            if any(i["score"] is None for i in g):
                tiers[t] = dict(n=len(g), ungraded=True)
                continue
            w = sum(i["weight"] for i in g)
            tiers[t] = dict(ungraded=False,
                            n=len(g),
                            answered=sum(i["status"] not in PENDING for i in g),
                            correct_or_points=sum(i["score"] for i in g),
                            weighted_pct=100 * sum(i["weight"] * i["score"] for i in g) / w,
                            unweighted_pct=100 * sum(i["score"] for i in g) / len(g))
        graded = {t: v for t, v in tiers.items() if not v["ungraded"]}
        domains = {}
        for dom in sorted({i["domain"] for i in group}):
            g = [i for i in group if i["domain"] == dom and i["score"] is not None]
            if g:
                domains[dom] = dict(count=len(g), score_pct=100 * sum(i["score"] for i in g) / len(g))
        pending = [i["id"] for i in group if i["status"] in PENDING]
        invalid = [i["id"] for i in group if is_invalid(i["status"])]
        complete_tiers = set(tiers) == EXPECTED_TIERS[split]
        report[split] = dict(
            tiers=tiers,
            graded_tiers=sorted(graded),
            ungraded_tiers=sorted(set(tiers) - set(graded)),
            total_pct=(sum(t["weighted_pct"] for t in graded.values()) / len(graded)) if graded else None,
            total_unweighted_pct=(sum(t["unweighted_pct"] for t in graded.values()) / len(graded)) if graded else None,
            official=strict,
            provisional=bool(pending or invalid or not complete_tiers or len(graded) < len(tiers)),
            missing_or_ungraded=pending,
            invalid_answers=invalid,
            unit_mismatch=[i["id"] for i in group if i["status"] == "unit_mismatch"],
            domain_scores=domains,
            tool_policy_compliance=("Not inferred by this grader; inspect run/tool logs. "
                                    "Tier 0 run with tools is invalid for the no-tool condition."))
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
        if s["total_pct"] is None:
            lines.append(f"| **{split}** | **total: no graded tiers** | | | | | |")
            continue
        n = sum(s["tiers"][t]["n"] for t in s["graded_tiers"])
        which = "tiers " + ",".join(map(str, s["graded_tiers"]))
        if s["ungraded_tiers"]:
            which += f"; tier {','.join(map(str, s['ungraded_tiers']))} ungraded, NOT the official score"
        tag = ", provisional" if s["provisional"] else ""
        lines.append(f"| **{split}** | **total ({which}{tag})** | **{n}** | | | "
                     f"**{s['total_pct']:.1f}** | **{s['total_unweighted_pct']:.1f}** |")
    out = "\n".join(lines)

    dom_lines = ["", "| Split | Domain | Items | Mean score % |", "|---|---|---:|---:|"]
    for split, s in report.items():
        for dom, d in s["domain_scores"].items():
            dom_lines.append(f"| {split} | {dom} | {d['count']} | {d['score_pct']:.1f} |")
    out += "\n" + "\n".join(dom_lines)

    out += ("\n\nWeighted % = domain-multiplier weighted (SCORING.md). Split total = mean of its graded tiers."
            "\nDomain scores are descriptive, unweighted and unstable (few items per domain)."
            "\nmain / holdout / tier0 are separate scores - never merge them."
            "\nTool-policy compliance is not checked here; review the run logs.")
    for split, s in report.items():
        if s["missing_or_ungraded"]:
            out += f"\n{split}: missing/ungraded -> {', '.join(s['missing_or_ungraded'])}"
        if s["invalid_answers"]:
            out += f"\n{split}: invalid/needs review -> {', '.join(s['invalid_answers'])}"
        if s["unit_mismatch"]:
            out += f"\n{split}: unit mismatch (scored 0) -> {', '.join(s['unit_mismatch'])}"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grade Hive answers on Reasoning Benchmark v2",
                                 epilog="exit status: 0 ok, 1 invalid/conflicting/unknown-unit "
                                        "answers or grader errors, 2 bad input")
    ap.add_argument("--answers", type=Path, required=True, help="JSONL of {id, answer[, unit][, rubric_scores]}")
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help="reasoning_benchmark dir")
    ap.add_argument("--strict", action="store_true",
                    help="scripts/grade.py rules: bare JSON number/label only, no units, "
                         "ungraded tier 5 counts 0")
    ap.add_argument("--all-splits", action="store_true",
                    help="also score splits with no answers (all zero) instead of skipping them")
    ap.add_argument("--json", type=Path, help="write full report + per-item results (no answer keys)")
    ap.add_argument("--csv", type=Path, help="write per-item results (no answer keys)")
    ap.add_argument("-q", "--quiet", action="store_true", help="no per-item lines on stderr")
    args = ap.parse_args(argv)

    try:
        problems = load_problems(args.bench)
        responses = read_jsonl(args.answers)
        items = grade_all(problems, responses, args.strict, not args.all_splits, args.quiet)
    except (InputError, OSError) as e:
        ap.error(str(e))
    report = summarize(items, args.strict)
    print(render_table(report))

    if args.json:
        args.json.write_text(json.dumps(dict(report=report, items=items), indent=2,
                                        ensure_ascii=False, default=str), encoding="utf-8")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ITEM_FIELDS)
            w.writeheader()
            w.writerows(items)
    return 1 if any(is_invalid(i["status"]) for i in items) else 0


if __name__ == "__main__":
    sys.exit(main())
