"""
probe4_serving.py -- is the hive's trouble in how the model is served?

Measurement only. It changes no colony code and is not part of a colony run.
The phaser, the agent's decide() and the judge run through their own code;
only the model they are handed is swapped for `Serving`, which changes how
each generate() call reaches the model:

    template   "raw"  -- the prompt text as the colony sends it today
               "chat" -- the same text wrapped in the tokenizer's chat
                         template as one user turn (thinking off)
    adapter    True -- the LoRA in adapter/;  False -- base Qwen3-4B
    penalties  True -- the call's own repetition_penalty / no_repeat_ngram
               False -- both removed
    greedy     True -- do_sample=False (judge consistency only)

The production setting is raw + adapter + penalties, sampled.

Sections (each writes one JSONL file under the output dir, row by row, so a
session that dies keeps what it finished):

    phaser    Problem_Phaser.parse_problem on the school prompt
    decide    Agent.decide() for three agents of the school run
    judge     Judge.deep_critique on four fixed REPORTs, right and wrong
    single    the school prompt as one chat call, base model, thinking on
    bench     benchmark dev items (tiers 1-4) as one chat call, base model,
              thinking on and off -- answers.jsonl files for hive_grader.py,
              graded locally (the keys never reach Kaggle)

The figure metrics are rough on purpose: "kept" is how many of the request's
figures appear in the text; "unexplained" counts numbers that are not a
request figure, not one +-x/ step from two of them, and not a small integer.
They are for comparing conditions, not for grading an answer.
"""
import contextlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, asdict
from itertools import combinations

import torch

SCHOOL = ("A school is taking 45 students and 5 teachers on a 3-day trip with a budget of "
          "180,000 rupees. Choose between a bus that costs 40,000 rupees in total and a train "
          "at 900 rupees per person, plan meals so that each day stays under 15,000 rupees, "
          "split the students into groups with one teacher each, and decide what to do if a "
          "student falls ill during the trip.")


@dataclass(frozen=True)
class Cond:
    template: str = "raw"
    adapter: bool = True
    penalties: bool = True
    greedy: bool = False
    thinking: bool = False

    @property
    def name(self):
        return "+".join([self.template, "lora" if self.adapter else "base",
                         "pen" if self.penalties else "nopen"]
                        + (["greedy"] if self.greedy else [])
                        + (["think"] if self.thinking else []))


PRODUCTION = Cond()
SERVING_GRID = [Cond(t, a, p) for t in ("raw", "chat") for a in (True, False)
                for p in (True, False)]
JUDGE_CONDS = [  # (condition, repeats)
    (PRODUCTION, 10),
    (Cond("chat", False, False), 10),
    (Cond("raw", True, True, greedy=True), 1),
    (Cond("chat", True, False, greedy=True), 1),
    (Cond("chat", False, False, greedy=True), 1),
]


# ---------------------------------------------------------------------------
# Serving wrapper
# ---------------------------------------------------------------------------

class Serving:
    """Stands in for the model inside the colony's own code. generate()
    re-wraps the prompt / strips penalties / toggles the adapter per `cond`,
    and returns the caller's original prompt ids followed by the new tokens,
    so `outputs[0][prompt_len:]` in the caller still slices correctly."""

    def __init__(self, model, tok, cond=PRODUCTION):
        self.model, self.tok, self.cond = model, tok, cond
        self.calls = []
        self.stop_ids = {i for i in (tok.eos_token_id,
                                     tok.convert_tokens_to_ids("<|im_end|>"),
                                     tok.convert_tokens_to_ids("<|endoftext|>"))
                         if isinstance(i, int) and i >= 0}

    def __getattr__(self, name):  # device, config, ... come from the model
        return getattr(self.model, name)

    def __call__(self, *a, **kw):  # think()'s manual forward pass, unchanged
        return self.model(*a, **kw)

    def _adapter_ctx(self):
        if self.cond.adapter or not hasattr(self.model, "disable_adapter"):
            return contextlib.nullcontext()
        return self.model.disable_adapter()

    def chat_ids(self, text, thinking=False):
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            enable_thinking=thinking, return_tensors="pt")
        if not torch.is_tensor(ids):  # BatchEncoding on some versions
            ids = ids["input_ids"]
        return ids

    def generate(self, input_ids=None, attention_mask=None, **kw):
        c = self.cond
        orig = input_ids
        ids = orig
        if c.template == "chat":
            text = self.tok.decode(orig[0], skip_special_tokens=False)
            ids = self.chat_ids(text, c.thinking).to(orig.device)
        mask = torch.ones_like(ids)
        if not c.penalties:
            kw.pop("repetition_penalty", None)
            kw.pop("no_repeat_ngram_size", None)
        if c.greedy:
            kw["do_sample"] = False
            for k in ("temperature", "top_p", "top_k"):
                kw.pop(k, None)
        # stopping criteria that slice at the caller's prompt length
        shifted = []
        for crit in kw.get("stopping_criteria") or []:
            if hasattr(crit, "prompt_len"):
                shifted.append((crit, crit.prompt_len))
                crit.prompt_len = ids.shape[1]
        t0 = time.time()
        try:
            with self._adapter_ctx(), torch.no_grad():
                out = self.model.generate(input_ids=ids, attention_mask=mask, **kw)
        finally:
            for crit, n in shifted:
                crit.prompt_len = n
        new = out[:, ids.shape[1]:]
        new_list = new[0].tolist()
        self.calls.append(dict(
            cond=c.name, prompt_tokens=int(ids.shape[1]), new_tokens=len(new_list),
            max_new_tokens=kw.get("max_new_tokens"),
            stopped=any(t in self.stop_ids for t in new_list),
            seconds=round(time.time() - t0, 2),
            text=self.tok.decode(new[0], skip_special_tokens=True)))
        return torch.cat([orig, new.to(orig.device)], dim=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def numbers(text):
    out = []
    for m in _NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def figure_metrics(text, request=SCHOOL):
    req = sorted(set(numbers(request)))
    near = set(req)
    for a, b in combinations(req, 2):
        near |= {a + b, abs(a - b), a * b}
        if b:
            near.add(a / b)
        if a:
            near.add(b / a)
    got = numbers(text)
    unexplained = sorted({v for v in got
                          if not any(math.isclose(v, n, rel_tol=1e-9) for n in near)
                          and not (v.is_integer() and v <= 12)})
    return dict(kept=sum(1 for r in req if r in got), of=len(req),
                unexplained=len(unexplained), unexplained_values=unexplained[:12])


def strip_think(text):
    """The answer after a closing </think>; '' when thinking never closed."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return "" if "<think>" in text else text.strip()


class Writer:
    def __init__(self, out_dir, section):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{section}.jsonl")
        self.rows = []

    def __call__(self, row):
        self.rows.append(row)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _seed(i):
    torch.manual_seed(1000 + i)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def run_phaser(model, tok, embedder, out_dir, conds=SERVING_GRID, repeats=2):
    from problem_phaser import Problem_Phaser
    w = Writer(out_dir, "phaser")
    for c in conds:
        for i in range(repeats):
            _seed(i)
            s = Serving(model, tok, c)
            spec = Problem_Phaser(s, tok, embed_model=embedder).parse_problem(SCHOOL)
            goal = spec.get("goal", "")
            reqs = [str(r) for r in spec.get("requirement") or []]
            w(dict(cond=c.name, rep=i, goal=goal, constraints=reqs,
                   goal_figures=figure_metrics(goal), constraint_figures=figure_metrics(" ".join(reqs)),
                   goal_fidelity=spec.get("goal_fidelity"),
                   constraint_fidelity=spec.get("constraint_fidelity"),
                   calls=s.calls))
    return w.rows


DECIDE_AGENTS = {
    "root-decomposer": dict(role="decomposer", parent_id=None, task=SCHOOL),
    "transport-executor": dict(role="executor", parent_id="root", task=(
        "Choose between the bus that costs 40,000 rupees in total and the train at "
        "900 rupees per person for the 45 students and 5 teachers.")),
    "meals-executor": dict(role="executor", parent_id="root", task=(
        "Plan meals for the trip so that each day stays under 15,000 rupees.")),
}


def run_decide(model, tok, out_dir, conds=SERVING_GRID, repeats=3, agents=DECIDE_AGENTS):
    from agent_node import Agent, _last_action_match
    from colony_state import AgentNode
    from event_queue import Messenger
    from text_utils import looks_degenerate
    w = Writer(out_dir, "decide")
    for name, a in agents.items():
        for c in conds:
            for i in range(repeats):
                _seed(i)
                s = Serving(model, tok, c)
                node = AgentNode(agent_id=f"{name}-{i}", role=a["role"], status="running",
                                 parent_id=a["parent_id"], task=a["task"], task_id=f"t-{name}",
                                 generation=1, request_text=SCHOOL)
                agent = Agent(tokeniser=tok, model=s, message=Messenger(), node=node)
                error = None
                try:
                    result = agent.decide()
                except Exception as e:  # a crash is a result here
                    result, error = None, f"{type(e).__name__}: {e}"
                last = s.calls[-1] if s.calls else {}
                text = last.get("text", "")
                m = _last_action_match(text)
                w(dict(agent=name, cond=c.name, rep=i, attempts=len(s.calls),
                       action=((m.group(1) or "").strip().upper() or None) if m is not None and m.groups() else None,
                       result=repr(result)[:600], error=error,
                       new_tokens=last.get("new_tokens"), stopped=last.get("stopped"),
                       chars=len(text), degenerate=looks_degenerate(text),
                       figures=figure_metrics(text), text=text, calls=s.calls))
    return w.rows


# Four REPORTs from the school runs. expect = the verdict a correct judge gives.
JUDGE_CASES = [
    dict(id="transport-right", expect="accept", task=DECIDE_AGENTS["transport-executor"]["task"],
         output=("The train costs 900 x 50 = 45,000 rupees for 45 students and 5 teachers. "
                 "The bus costs 40,000 rupees in total. Take the bus: it is 5,000 rupees "
                 "cheaper and leaves 140,000 rupees of the budget.")),
    dict(id="transport-wrong", expect="reject", task=DECIDE_AGENTS["transport-executor"]["task"],
         output=("The train costs 900 x 45 = 40,500 rupees for the students. The bus costs "
                 "40,000 rupees. The two are almost equal, so take the train for comfort.")),
    dict(id="meals-right", expect="accept", task=DECIDE_AGENTS["meals-executor"]["task"],
         output=("There are 50 people. At about 250 rupees per person per day (estimated: "
                 "breakfast 60, lunch 100, dinner 90), each day costs 50 x 250 = 12,500 "
                 "rupees, under the 15,000 limit. Over 3 days meals cost 37,500 rupees.")),
    dict(id="meals-wrong", expect="reject", task=DECIDE_AGENTS["meals-executor"]["task"],
         output=("Plan 60 lunches at about 150 rupees and dinners at 200 rupees for the 45 "
                 "students: 45 x 350 = 15,750 rupees per day, which keeps each day within "
                 "budget.")),
]


def run_judge(model, tok, out_dir, conds=JUDGE_CONDS, cases=JUDGE_CASES):
    import main
    from judge import Judge
    w = Writer(out_dir, "judge")
    for c, repeats in conds:
        s = Serving(model, tok, c)
        judge = Judge(llm_call_fn=main.build_llm_call_fn(s, tok))
        for case in cases:
            for i in range(repeats):
                _seed(i)
                n0 = len(s.calls)
                r = judge.deep_critique(case["output"], case["task"], project_goal=SCHOOL)
                last = s.calls[-1] if len(s.calls) > n0 else {}
                w(dict(case=case["id"], expect=case["expect"], cond=c.name, rep=i,
                       verdict=r.get("verdict"), right=r.get("verdict") == case["expect"],
                       new_tokens=last.get("new_tokens"), stopped=last.get("stopped"),
                       reasoning=r.get("reasoning", "")[:1500], raw=last.get("text", "")))
    return w.rows


def chat_once(model, tok, prompt, thinking, max_new_tokens, adapter=False, seed=0):
    """One chat call the way Qwen3 is meant to be served. A thinking run that
    hits the cap is closed with </think> and given 400 tokens to answer."""
    s = Serving(model, tok, Cond("chat", adapter, False, thinking=thinking))
    ids = s.chat_ids(prompt, thinking).to(model.device)
    sampling = (dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20) if thinking
                else dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20))
    torch.manual_seed(seed)
    t0 = time.time()
    with s._adapter_ctx(), torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                             **sampling)
        new = out[0, ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=False)
        capped = not any(t in s.stop_ids for t in new.tolist())
        if thinking and capped and "</think>" not in text:
            tail = tok("\n</think>\n\n", return_tensors="pt", add_special_tokens=False
                       )["input_ids"].to(out.device)
            ids2 = torch.cat([out, tail], dim=1)
            out2 = model.generate(input_ids=ids2, attention_mask=torch.ones_like(ids2),
                                  max_new_tokens=400, pad_token_id=tok.pad_token_id,
                                  do_sample=False)
            text += "\n</think>\n\n" + tok.decode(out2[0, ids2.shape[1]:], skip_special_tokens=False)
    for sp in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(sp, "")
    return dict(text=text, answer=strip_think(text) if thinking else text.strip(),
                new_tokens=int(new.shape[0]), capped=capped,
                seconds=round(time.time() - t0, 1))


def run_single(model, tok, out_dir, repeats=5, max_new_tokens=4096):
    w = Writer(out_dir, "single")
    for i in range(repeats):
        r = chat_once(model, tok, SCHOOL, True, max_new_tokens, seed=i)
        w(dict(rep=i, figures=figure_metrics(r["answer"]), **r))
    return w.rows


def bench_prompt(item):
    """The export's text plus a choice item's options. The JSON format
    sentence stays: hive_grader reads a JSON answer first."""
    text = item["text"].rstrip()
    if item.get("choices"):
        text += "\n\nOptions: " + "; ".join(
            f"{k}) {str(v).rstrip('. ')}" for k, v in sorted(item["choices"].items()))
    return text


def run_bench(model, tok, items, out_dir, thinking, max_new_tokens):
    """answers_<mode>.jsonl ({"id", "answer"}) for hive_grader.py. Resumable."""
    mode = "think" if thinking else "nothink"
    path = os.path.join(out_dir, f"answers_{mode}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    w = Writer(out_dir, f"bench_{mode}")
    for item in items:
        if item["id"] in done:
            continue
        r = chat_once(model, tok, bench_prompt(item), thinking, max_new_tokens)
        w(dict(id=item["id"], **r))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": item["id"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
    return w.rows


# ---------------------------------------------------------------------------
# Summaries (printed in the notebook; the JSONL rows hold everything)
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def summarize(rows, key, fields):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in key), []).append(r)
    out = []
    for g, rs in groups.items():
        row = dict(zip(key, g), n=len(rs))
        for name, fn in fields.items():
            row[name] = fn(rs)
        out.append(row)
    return out


def print_table(rows):
    if not rows:
        print("(no rows)")
        return
    cols = list(rows[0])
    width = {c: max(len(str(c)), *(len(str(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(str(c).ljust(width[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c)).ljust(width[c]) for c in cols))


def _pct(rs, pred):
    return f"{100 * sum(1 for r in rs if pred(r)) // max(1, len(rs))}%"


def summary_phaser(rows):
    return summarize(rows, ["cond"], {
        "goal_kept": lambda rs: _mean([r["goal_figures"]["kept"] for r in rs]),
        "goal_unexpl": lambda rs: _mean([r["goal_figures"]["unexplained"] for r in rs]),
        "cons_kept": lambda rs: _mean([r["constraint_figures"]["kept"] for r in rs]),
        "cons_unexpl": lambda rs: _mean([r["constraint_figures"]["unexplained"] for r in rs]),
        "stopped": lambda rs: _pct([c for r in rs for c in r["calls"]], lambda c: c["stopped"]),
    })


def summary_decide(rows):
    return summarize(rows, ["agent", "cond"], {
        "actions": lambda rs: ",".join(str(r["action"]) for r in rs),
        "tokens": lambda rs: _mean([r["new_tokens"] for r in rs]),
        "stopped": lambda rs: _pct(rs, lambda r: r["stopped"]),
        "retries": lambda rs: _mean([r["attempts"] - 1 for r in rs]),
        "degen": lambda rs: _pct(rs, lambda r: r["degenerate"]),
        "kept": lambda rs: _mean([r["figures"]["kept"] for r in rs]),
        "unexpl": lambda rs: _mean([r["figures"]["unexplained"] for r in rs]),
    })


def summary_judge(rows):
    return summarize(rows, ["cond", "case"], {
        "expect": lambda rs: rs[0]["expect"],
        "accept": lambda rs: f"{sum(1 for r in rs if r['verdict'] == 'accept')}/{len(rs)}",
        "right": lambda rs: _pct(rs, lambda r: r["right"]),
        "tokens": lambda rs: _mean([r["new_tokens"] for r in rs]),
        "stopped": lambda rs: _pct(rs, lambda r: r["stopped"]),
    })
