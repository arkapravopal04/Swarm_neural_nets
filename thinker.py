"""
thinker.py -- think first, then split (THINK_FIRST.md, P1).

Before the colony runs, the thinker solves the whole request the way the
plain call does (Qwen3 chat, thinking ON, adapter off, up to 4096 tokens),
n times in one batch. Each run ends with PLAN: and a last ANSWER: line.

If >= 2 ANSWER lines agree -- numbers with their units, or option letters,
never prose -- and at least one of the agreeing runs finished uncapped, that
line is the answer and the colony never starts. Otherwise the colony runs,
with the best PLAN as background in its agents' think() seeds, and the
thinker's own best line is kept as the fallback the gate compares against.

Agreement is a stopping signal, not verification: the runs share a prompt
and are sampled at T=0.6, so their mistakes are correlated.
"""
import math
import re
import time

import hive_grader
import serving
from text_utils import sanitize_thought_text

INSTRUCTION = (
    "\n\nSolve this completely. When you are done, write PLAN: followed by at most 6 short "
    "numbered steps that your solution followed. Then write a last line ANSWER: followed by "
    "only the final value with its unit, or only the option letter, or NONE if you could not "
    "finish.")
CHOICE_MARK = "Answer with the letter of the correct option."
PLAN_MAX_CHARS = 600
AGREE_REL_TOL = 1e-3      # a consensus bar only; the answer key still grades
CLOSE_TOKENS = 400        # a capped run gets this many greedy tokens to answer

_ANSWER_RE = re.compile(r"^[ \t>*#]*ANSWER[ \t*]*:[ \t*]*(.*?)[ \t*]*$", re.M | re.I)
_PLAN_RE = re.compile(r"^[ \t>*#]*PLAN[ \t*]*:(.*?)(?=^[ \t>*#]*ANSWER[ \t*]*:|\Z)",
                      re.M | re.I | re.S)
_NUM_PREFIX_RE = re.compile(
    r"\s*(?:(?:about|approximately|approx\.?|roughly|~|≈)\s*"
    r"|[A-Za-z_][\w]{0,15}\s*(?:=|≈)\s*)?", re.I)
_NUM_SUFFIX_RE = re.compile(r"\s*(?:[A-Za-z][\w/^·\-]*)?\s*[.)]?\s*")
_CHOICE_RE = re.compile(r"\s*\(?([A-Da-d])\)?\.?\s*")   # the grader's labels are A-D too
_BOXED_MARK_RE = re.compile(r"^\s*ANSWER:\s*|\s*$")


def visible(text):
    """The part of a thinking-mode run after </think> ('' if it never closed)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return "" if "<think>" in text else text.strip()


def answer_value(text):
    """What follows the last ANSWER: in the visible text, or None."""
    found = _ANSWER_RE.findall(visible(text))
    return found[-1].strip() if found else None


def plan_text(text):
    """The last PLAN: block of the visible text, sanitised, without any
    ANSWER line, cut at a line boundary to PLAN_MAX_CHARS. None if absent."""
    found = _PLAN_RE.findall(visible(text))
    if not found:
        return None
    plan = sanitize_thought_text(found[-1]) or ""
    lines = [l.rstrip() for l in plan.strip().split("\n")
             if l.strip() and not _ANSWER_RE.match(l)]
    out = ""
    for line in lines:
        if len(out) + len(line) + 1 > PLAN_MAX_CHARS:
            break
        out += line + "\n"
    if not out and lines:
        out = lines[0][:PLAN_MAX_CHARS]
    return out.strip() or None


def parse_vote(value, choice_item):
    """("choice", letter) | ("num", value, unit) | None. Only a bare letter
    (option items) or one number with an optional unit votes; prose, plans
    and NONE never do."""
    if not value:
        return None
    v = value.strip().strip("*`$").rstrip(".").strip()
    if not v or v.upper() == "NONE":
        return None
    # _clean rewrites \boxed{x} to "\nANSWER: x\n"; take the x back out.
    text = _BOXED_MARK_RE.sub("", hive_grader._clean(v)).strip()
    if choice_item:
        m = _CHOICE_RE.fullmatch(text)
        return ("choice", m.group(1).upper()) if m else None
    toks = hive_grader._tokens(text)
    if len(toks) != 1 or not math.isfinite(toks[0].value):
        return None
    tok = toks[0]
    if not _NUM_PREFIX_RE.fullmatch(text[:tok.start]):
        return None
    rest = text[tok.end:]
    if tok.unit and rest.lstrip().startswith(tok.unit):
        rest = rest.lstrip()[len(tok.unit):]
        if not re.fullmatch(r"\s*[.)]?\s*", rest):
            return None
    elif not _NUM_SUFFIX_RE.fullmatch(rest):
        return None
    return ("num", tok.value, tok.unit)


def agree(a, b):
    if a is None or b is None or a[0] != b[0]:
        return False
    if a[0] == "choice":
        return a[1] == b[1]
    (_, va, ua), (_, vb, ub) = a, b
    ia, ib = hive_grader.unit_info(ua), hive_grader.unit_info(ub)
    # abs_tol stays 0: tiny quantities (pA, pF) must not all agree.
    if ia and ib:
        return ia[0] == ib[0] and math.isclose(va * ia[1], vb * ib[1], rel_tol=AGREE_REL_TOL)
    if ua and ub and ua.strip() != ub.strip():
        # Two units, one unknown to the grader: no conversion is possible,
        # so they agree only when written the same ("5 pF" is not "5 nF").
        return False
    # One side has no unit: raw values ("20" matches "20 mA", not "0.02 A").
    return math.isclose(va, vb, rel_tol=AGREE_REL_TOL)


def decide(runs, choice_item):
    """Vote over runs ([{text, capped, ...}]). Returns the agreed line (or
    None), the fallback line and its run index, the plan and its source."""
    lines = [answer_value(r["text"]) for r in runs]
    votes = [parse_vote(l, choice_item) for l in lines]

    groups = []
    for i, v in enumerate(votes):
        if v is None:
            continue
        for g in groups:
            if agree(votes[g[0]], v):
                g.append(i)
                break
        else:
            groups.append([i])
    # Biggest group; ties go to the one with an uncapped run, then the earliest.
    best = max(groups, key=lambda g: (len(g), any(not runs[i]["capped"] for i in g)),
               default=[])
    best_uncapped = [i for i in best if not runs[i]["capped"]]

    agreed_run = best_uncapped[0] if len(best) >= 2 and best_uncapped else None

    # Fallback (the gate's thinker-only answer): plurality, else the first
    # uncapped run with an ANSWER line, else the first ANSWER line at all.
    with_line = [i for i, l in enumerate(lines) if l]
    if len(best) >= 2:
        fallback_run = (best_uncapped or best)[0]
    else:
        fallback_run = next((i for i in with_line if not runs[i]["capped"]),
                            with_line[0] if with_line else None)

    # Plan: the plurality group (uncapped first), else any uncapped run,
    # else any capped one.
    plurality = sorted(best, key=lambda i: runs[i]["capped"]) if len(best) >= 2 else []
    order = ([("plurality", i) for i in plurality]
             + [("uncapped", i) for i in range(len(runs)) if not runs[i]["capped"]]
             + [("capped", i) for i in range(len(runs)) if runs[i]["capped"]])
    plan, plan_source = None, "none"
    for source, i in order:
        plan = plan_text(runs[i]["text"])
        if plan:
            plan_source = source
            break

    return {
        "answer_line": f"ANSWER: {lines[agreed_run]}" if agreed_run is not None else None,
        "fallback_line": f"ANSWER: {lines[fallback_run]}" if fallback_run is not None else None,
        "fallback_run": fallback_run,
        "plan": plan, "plan_source": plan_source,
        "answers": lines,
    }


class Thinker:
    """`generate_fn(prompt, n) -> [{"text", "capped", "tokens"}]`; the
    default runs the model. Tests pass a fake."""

    def __init__(self, model=None, tokeniser=None, n=3, max_new_tokens=4096, generate_fn=None,
                 settle=True):
        self.model, self.tokeniser = model, tokeniser
        self.n, self.max_new_tokens = n, max_new_tokens
        # settle=False: never end the run, only hand the colony a plan. For a
        # request whose answer is a whole plan (the G5 school canary), where
        # one agreed figure ("40,000 rupees") is not an answer.
        self.settle = settle
        self.generate_fn = generate_fn or self._generate

    def solve(self, request_text):
        t0 = time.time()
        runs = self.generate_fn(request_text.rstrip() + INSTRUCTION, self.n)
        out = decide(runs, CHOICE_MARK in request_text)
        if not self.settle:
            out["answer_line"] = None
        out["runs"] = [dict(r, answer=a) for r, a in zip(runs, out.pop("answers"))]
        out["seconds"] = round(time.time() - t0, 1)
        out["n"] = self.n
        return out

    def _generate(self, prompt, n):
        import torch
        model, tok = self.model, self.tokeniser
        stops = serving.stop_ids(tok)
        pad = tok.pad_token_id if tok.pad_token_id is not None else stops[-1]
        ids = torch.tensor([serving.render(tok, prompt, thinking=True)], device=model.device)
        runs = []
        with torch.no_grad(), serving.adapter_off(model):
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=self.max_new_tokens, do_sample=True,
                                 temperature=0.6, top_p=0.95, top_k=20, num_return_sequences=n,
                                 eos_token_id=stops, pad_token_id=pad)
            for row in out:
                new = row[ids.shape[1]:].tolist()
                end = next((k for k, t in enumerate(new) if t in stops), None)
                capped = end is None
                used = new if capped else new[:end]
                text = tok.decode(used, skip_special_tokens=False)
                if capped and "</think>" not in text:
                    # Like probe 4's chat_once: close the thinking and let it answer.
                    tail = tok("\n</think>\n\n", return_tensors="pt", add_special_tokens=False
                               )["input_ids"].to(row.device)
                    ids2 = torch.cat([row.unsqueeze(0), tail], dim=1)
                    out2 = model.generate(input_ids=ids2, attention_mask=torch.ones_like(ids2),
                                          max_new_tokens=CLOSE_TOKENS, do_sample=False,
                                          eos_token_id=stops, pad_token_id=pad)
                    text += "\n</think>\n\n" + tok.decode(out2[0, ids2.shape[1]:], skip_special_tokens=False)
                for sp in ("<|im_end|>", "<|endoftext|>"):
                    text = text.replace(sp, "")
                runs.append({"text": text, "capped": capped, "tokens": len(used)})
        return runs
