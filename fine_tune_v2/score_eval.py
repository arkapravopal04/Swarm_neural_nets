"""
fine_tune_v2/score_eval.py -- score generated decide() outputs the way the
runtime would read them (TRAINING_SPEC.md §9), and test the §10 gates G1-G3.

Generation runs on a GPU (generate_eval.py); scoring runs here on a CPU, so
the scorer can be tested and fixed without spending GPU time.

    # score one generation file against its rows
    python fine_tune_v2/score_eval.py score --rows fine_tune_v2/data/dev.jsonl --gens gens.jsonl --out scored.jsonl
    # gates: arm files (one per seed) against base files
    python fine_tune_v2/score_eval.py gates --base base_s0.jsonl base_s1.jsonl --arm lora_s0.jsonl lora_s1.jsonl
    # self-test: the gold responses scored as if a model had written them
    python fine_tune_v2/score_eval.py selftest --rows fine_tune_v2/data/dev.jsonl

How a row is read (every check uses the runtime's own code where one exists):
- The generated text is streamed token by token through the real
  Agent.decide() (its _ActionPayloadStop cut, its parser, its degenerate
  re-sample check) -- what the colony would have acted on.
- valid: decide() returned SPAWN / TOOL / REPORT / DIE on the first draw
  (THINK is its parse-failure fallback), and a TOOL / SPAWN payload is
  well-formed (exact tool-argument keys; SPAWN graph below).
- action_ok: the action is in the class's acceptable set. E1/E2 and the
  rows built from them (E6, E7, E9) accept REPORT or TOOL when the value is
  right (spec §9); U2 accepts SPAWN or REPORT (spec §7.1).
- answer_correct (E-class answer rows): the key is the solver's answer,
  regenerated from the row's scenario id (build.py seeds every scenario by
  id): a single question's printed value or option letter, a multi-part
  leaf's printed value (else the first figure of its gold REPORT the prompt
  does not give; prose-only parts have no key). A REPORT is right when the key
  is one of its answer figures (first or last of its first sentence, or its
  last) or, for a choice, the letter it picks; a TOOL program, run in the
  runtime's sandbox, must print it last. E4 rests on an estimate: it needs a
  labelled estimate and arithmetic that holds.
- U1 roll-ups carry each child's result; a U2 REPORT names the missing part.
- stopped: <|im_end|> / <|endoftext|>, or decide()'s structural stop -- not
  its 200-token REPORT cap.
- invented: V2 (the runtime's figure classification and chain passes) on a
  REPORT against the prompt's content; E4 allows labelled estimates.
- SPAWN: labels unique, dependencies resolve and are acyclic, roles allowed at
  this depth, every gold part matched by its own child (task wording),
  decomposer children where the gold has them (nesting_correct).
- row_correct = valid and action_ok and not invented and (answer_correct for
  answer rows) and the SPAWN checks for SPAWN rows.
"""
import argparse
import collections
import contextlib
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import families_multi as FM  # noqa: E402
import families_single as FS  # noqa: E402
import text_utils as tu  # noqa: E402
from agent_node import (EMPTY_DECISION_PLACEHOLDER, Agent, _ActionPayloadStop,  # noqa: E402
                        may_spawn_decomposers)
from colony_state import AgentNode  # noqa: E402
from event_queue import Messenger  # noqa: E402
from render import PRODUCTION_ROLES, PRODUCTION_TOOLS  # noqa: E402
from validate import _GoldModel, v2_figures_text, v3_arithmetic  # noqa: E402

TOKENIZER_DIR = os.path.join(ROOT, "fine_tuned_params_v1")
ACTIONS = {"SPAWN", "TOOL", "REPORT", "DIE"}
TOOL_KEYS = {"run_code": {"code_string"}, "verify_math": {"expression", "mode"}}
# Classes whose gold is the executor's answer (REPORT) or its program (TOOL).
ANSWER_CLASSES = {"E1", "E2", "E3", "E4", "E6", "E7", "E8", "E9"}
# Either form is acceptable when the value is right (spec §9).
REPORT_OR_TOOL = {"E1", "E2", "E6", "E7", "E9"}
TOL = 1e-6


def _payload(text):
    return text.split("PAYLOAD:", 1)[1].strip() if "PAYLOAD:" in text else ""


def _gold_action(text):
    return text.split("\n", 1)[0].split(":", 1)[1].strip()


def _same(a, b):
    return abs(a - b) <= TOL * max(1.0, abs(b))


# ------------------------------------------------------------------ runtime read

def runtime_read(row, text, tok):
    """(action, payload, calls, cut) as Agent.decide() reads `text`."""
    node = AgentNode(agent_id="agent_eval", role=row["role"], status="running",
                     parent_id=None if row["generation"] == 0 else "agent_parent",
                     task="", task_id="task_eval", generation=row["generation"], request_text="")
    agent = Agent(tokeniser=tok, model=None, message=Messenger(), node=node)
    gm = _GoldModel(tok, text)
    agent.model = gm
    agent._build_prompt = lambda *a, **k: row["prompt"]
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            action, payload = agent.decide(PRODUCTION_ROLES, list(PRODUCTION_TOOLS), [])
        except Exception as e:  # a crash in the runtime's reader is an invalid output
            return "CRASH", f"{type(e).__name__}: {e}", gm.calls, False
    n_text = len(tok(text, add_special_tokens=False)["input_ids"])
    consumed = getattr(gm, "consumed", n_text)
    # decide()'s structural stop (balanced payload, blank line) counts as
    # stopping; its 200-token REPORT cap does not.
    cut = consumed < n_text and (action != "REPORT" or consumed < _ActionPayloadStop.REPORT_MAX_NEW_TOKENS)
    return action, payload, gm.calls, cut


# ------------------------------------------------------------------ checks

_SANDBOX_CACHE = {}


def run_program(code, domain):
    key = (code, domain)
    if key not in _SANDBOX_CACHE:
        from tools import ToolRegistry
        with contextlib.redirect_stdout(io.StringIO()):
            r = ToolRegistry.execute("run_code", {"code_string": code}, domain=domain)
        out = str(r.get("data") or "").replace("\r\n", "\n").strip()
        _SANDBOX_CACHE[key] = (r.get("status") == "success", out)
    return _SANDBOX_CACHE[key]


def tool_ok(payload):
    if not isinstance(payload, dict):
        return False
    name, args = payload.get("tool_name"), payload.get("args")
    if not (name in TOOL_KEYS and isinstance(args, dict) and set(args) == TOOL_KEYS[name]):
        return False
    return name != "verify_math" or args.get("mode") == "computational"


def _match_parts(gold, subs):
    """Every gold part matched to its own child by task wording (greedy,
    best overlap first, Jaccard >= 0.3), and no child left over."""
    pairs = sorted(((len(_words(g["task"]) & _words(s["task"]))
                     / max(1, len(_words(g["task"]) | _words(s["task"]))), i, j)
                    for i, g in enumerate(gold) for j, s in enumerate(subs)), reverse=True)
    used_g, used_s = set(), set()
    for score, i, j in pairs:
        if score >= 0.3 and i not in used_g and j not in used_s:
            used_g.add(i)
            used_s.add(j)
    return len(used_g) == len(gold) == len(subs)


def spawn_check(payload, gold_payload, generation):
    """(graph_valid, coverage_ok, nesting_ok, reason)."""
    subs = payload.get("subtasks") if isinstance(payload, dict) else None
    if not isinstance(subs, list) or not subs:
        return False, False, False, "no subtasks list"
    labels = []
    for s in subs:
        if not isinstance(s, dict) or not {"label", "role", "task", "dependencies"} <= set(s):
            return False, False, False, "subtask missing a field"
        if (not isinstance(s["label"], str) or not isinstance(s["dependencies"], list)
                or not all(isinstance(d, str) for d in s["dependencies"]) or not str(s["task"]).strip()):
            return False, False, False, "bad label, task or dependencies"
        labels.append(s["label"])
    if len(set(labels)) != len(labels):
        return False, False, False, "duplicate labels"
    allowed = {"executor", "decomposer"} if may_spawn_decomposers(generation) else {"executor"}
    if any(s["role"] not in allowed for s in subs):
        return False, False, False, "role not allowed at this depth"
    deps = {s["label"]: list(s["dependencies"]) for s in subs}
    if any(d not in deps for ds in deps.values() for d in ds):
        return False, False, False, "dependency names no label in the batch"
    state = {}

    def cyclic(n):
        if state.get(n) == 1:
            return True
        if state.get(n) == 2:
            return False
        state[n] = 1
        bad = any(cyclic(d) for d in deps[n])
        state[n] = 2
        return bad
    if any(cyclic(n) for n in deps):
        return False, False, False, "dependency cycle"
    gold = gold_payload["subtasks"]
    coverage = _match_parts(gold, subs)
    nesting = (sum(s["role"] == "decomposer" for s in subs)
               == sum(s["role"] == "decomposer" for s in gold))
    return True, coverage, nesting, None


_SCENARIOS = {}


def scenario(row):
    """The row's scenario, regenerated from its id exactly as build.py made it
    (build.scenarios seeds each with "<seed>-<family>-<index>")."""
    sid = row["scenario_id"]
    if sid not in _SCENARIOS:
        name, idx = sid.rsplit("-", 1)
        fn = FS.FAMILIES.get(name) or FM.FAMILIES[name]
        sc = fn(random.Random(f"v2-{name}-{int(idx)}"), int(idx))
        if sc.request not in row["prompt"]:
            raise SystemExit(f"scenario {sid} does not regenerate the row's request")
        _SCENARIOS[sid] = sc
    return _SCENARIOS[sid]


def answer_key(row):
    """("value", x) | ("values", [x, ...]) | ("label", "B") | (None, None).
    A multi-part leaf with no program is keyed on the first figure of its gold
    REPORT that the prompt does not give; a prose-only part has no key."""
    sc = scenario(row)
    if isinstance(sc, FS.Single):
        vals = list(tu.figure_values(sc.printed))
        if isinstance(sc.answer, str) and not vals:
            return "label", sc.answer.strip()
        return ("value", vals[-1]) if vals else (None, None)
    m = re.search(r"Your Task: (.*)", row["prompt"])
    task = (m.group(1) if m else "").strip().rstrip(".")
    found = []

    def walk(ps):
        for p in ps:
            if p.text.strip().rstrip(".") == task:
                found.append(p)
            walk(p.pieces)
    walk(sc.parts)
    if not found:
        return None, None
    part = found[0]
    if part.printed:
        vals = list(tu.figure_values(part.printed))
        return ("value", vals[-1]) if vals else (None, None)
    # No program: the figures in the gold REPORT's answer positions that the
    # prompt does not already give; any one of them stated as the answer counts.
    given = set(tu.figure_values(prompt_sources(row["prompt"])))
    keys = [v for v in answer_figures(part.report or "") if v not in given]
    return ("values", keys) if keys else (None, None)


_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "so", "it", "this", "that", "for", "with", "be",
         "choose", "pick", "select", "use", "go", "answer", "option", "because", "since", "as", "in", "on"}
_CHOICE_RE = re.compile(r"\b(?:answer|option|choice|choose|pick|select)\w*\s*(?:is\s*|:\s*)?\(?([A-E])\b"
                        r"|(?:^|[.!?]\s+)\(?([A-E])\)?(?=[.:,)]|\s+(?:is|-)\b|\s*$)")


def chosen_label(text):
    """The option letter a REPORT chooses: its first "the answer is B" /
    "option B" phrase or sentence-opening letter ("C is correct")."""
    text = re.sub(r"[*_`]", "", text).strip()
    for a, b in _CHOICE_RE.findall(text):
        return a or b
    return None


def answer_figures(text):
    """The figures a REPORT can be giving as its answer: the first and the
    last of its first sentence (answer-first or steps-then-answer), and its
    last figure."""
    sents = tu.split_sentences(text) or [text]
    first = list(tu.figure_values(sents[0]))
    every = list(tu.figure_values(text))
    return first[:1] + first[-1:] + every[-1:]


def answer_ok(text, kind, key):
    if kind == "label":
        return chosen_label(text) == key
    keys = key if kind == "values" else [key]
    return any(_same(v, k) for v in answer_figures(text) for k in keys)


def printed_ok(out, kind, key):
    """A program's answer is the last thing it prints."""
    if not out.strip():
        return False
    if kind == "label":
        return out.strip().splitlines()[-1].strip() == key
    vals = list(tu.figure_values(out))
    keys = key if kind == "values" else [key]
    return bool(vals) and any(_same(vals[-1], k) for k in keys)


def estimate_first_step_ok(row, action, payload, arith_error):
    """An E4 row before any tool: the estimates are the model's own, so there
    is no single right total. A program must run, print a figure, and use the
    request's own inputs (the gold program's non-estimate values, e.g. the
    head count's parts). A direct REPORT instead needs a labelled estimate,
    arithmetic that holds, and those inputs worked into it."""
    gold_code = json.loads(_payload(row["response"]))["args"]["code_string"]
    given = {n: float(v) for n, v in re.findall(r"^(\w+) = ([\d.]+)\s*$", gold_code, re.M)}
    # Either every input as written, or a group of them already combined
    # ("people = 246" for attendees + speakers).
    options = [list(given.values())]
    m = re.search(r"show\((.*)\)\s*$", gold_code, re.S)
    for grp in re.findall(r"\(([^()]*)\)", m.group(1) if m else ""):
        names = re.findall(r"[A-Za-z_]\w*", grp)
        if names and all(n in given for n in names):
            options.append([eval(grp, {"__builtins__": {}}, given)]
                           + [v for n, v in given.items() if n not in names])

    def has_inputs(found):
        return any(all(any(_same(u, v) for u in found) for v in opt) for opt in options)

    if action == "TOOL":
        code = payload["args"].get("code_string", "") if payload.get("tool_name") == "run_code" else ""
        ok, out = run_program(code, row["domain"]) if code else (False, "")
        used = [float(x.replace("_", "")) for x in re.findall(r"(?<![\w.])\d[\d_]*(?:\.\d+)?", code)]
        printed = list(tu.figure_values(out))
        # The last value printed must be a total it worked out, not an input echoed back.
        inputs = {v for opt in options for v in opt}
        return (ok and bool(printed) and not any(_same(printed[-1], v) for v in inputs)
                and has_inputs(used))
    text = str(payload)
    return bool(tu.hedged_figure_values(text)) and not arith_error and has_inputs(list(tu.figure_values(text)))


def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in _STOP and len(w) > 2}


def rollup_ok(text, gold_text):
    """U1: the roll-up carries each child's result -- every gold sentence
    that states figures is represented by at least one of them (its result or
    its working, whichever the roll-up chose to quote)."""
    have = list(tu.figure_values(text))
    for sent in tu.split_sentences(gold_text):
        vals = list(tu.figure_values(sent))
        if vals and not any(_same(h, v) for h in have for v in vals):
            return False
    return True


def names_missing(text, gold_payload):
    """U2 REPORT: it names the part still missing (a content word of the gold
    re-spawn's task)."""
    want = set()
    for sub in gold_payload.get("subtasks", []):
        want |= _words(sub["task"])
    return bool(want & _words(text))


def prompt_sources(prompt):
    """The prompt's content (request, task, constraints, ghost context,
    thoughts with tool and child results), without the action menu and format
    rules that follow it."""
    return prompt.split("\nAvailable actions:")[0]


# ------------------------------------------------------------------ one row

def score_row(row, gen, tok):
    text = gen["text"].strip()
    cls, gold = row["class"], row["response"]
    g_action = _gold_action(gold)
    action, payload, calls, cut = runtime_read(row, text, tok)
    rec = {"id": row["id"], "class": cls, "family": row["family"], "role": row["role"],
           "generation": row["generation"], "action": action, "gold_action": g_action,
           "calls": calls, "stopped_eos": bool(gen.get("eos")), "cut": cut,
           "stopped": bool(gen.get("eos")) or cut, "n_new": gen.get("n_new"),
           "answer_correct": None, "invented": False, "arith_error": False,
           "graph_valid": None, "coverage_ok": None, "nesting_ok": None, "content_ok": None, "why": None}

    valid = action in ACTIONS and calls == 1
    if action == "TOOL":
        valid = valid and tool_ok(payload)
    if action == "SPAWN":
        gold_subs = json.loads(_payload(gold)) if g_action == "SPAWN" else {"subtasks": []}
        gv, cov, nest, why = spawn_check(payload, gold_subs, row["generation"])
        if g_action != "SPAWN":
            cov = nest = None
        rec.update(graph_valid=gv, coverage_ok=cov, nesting_ok=nest)
        valid = valid and gv
        if why:
            rec["why"] = why
    if action == "REPORT" and str(payload).strip() in ("", EMPTY_DECISION_PLACEHOLDER):
        valid = False
    rec["valid"] = valid

    acceptable = {g_action}
    if (cls in REPORT_OR_TOOL and g_action in ("REPORT", "TOOL")) or (cls == "E4" and g_action == "TOOL"):
        acceptable = {"REPORT", "TOOL"}
    if cls == "U2":
        acceptable = {"SPAWN", "REPORT"}
    rec["action_ok"] = action in acceptable

    if action == "REPORT" and valid:
        why = v2_figures_text(prompt_sources(row["prompt"]), str(payload), estimate_ok=(cls == "E4"))
        rec["invented"] = why is not None
        rec["arith_error"] = v3_arithmetic(str(payload)) is not None
        if why and not rec["why"]:
            rec["why"] = why

    is_answer_row = cls in ANSWER_CLASSES and g_action in ("REPORT", "TOOL")
    if is_answer_row:
        kind, key = answer_key(row)
        if not (rec["action_ok"] and valid):
            rec["answer_correct"] = False
        elif cls == "E4" and g_action == "TOOL":
            rec["answer_correct"] = estimate_first_step_ok(row, action, payload, rec["arith_error"])
        elif cls == "E4":
            # After the program ran: report its printed total, as an estimate.
            rec["answer_correct"] = (kind is not None and answer_ok(str(payload), kind, key)
                                     and bool(tu.hedged_figure_values(str(payload))))
        elif kind is not None:
            if action == "REPORT":
                rec["answer_correct"] = answer_ok(str(payload), kind, key)
            else:
                ok, out = run_program(payload["args"].get("code_string", ""), row["domain"]) \
                    if payload.get("tool_name") == "run_code" else (False, "")
                rec["answer_correct"] = ok and printed_ok(out, kind, key)
    if cls == "U1" and action == "REPORT" and valid:
        rec["content_ok"] = rollup_ok(str(payload), _payload(gold))
    if cls == "U2" and action == "REPORT" and valid and g_action == "SPAWN":
        rec["content_ok"] = names_missing(str(payload), json.loads(_payload(gold)))

    ok = valid and rec["action_ok"] and not rec["invented"] and rec["content_ok"] is not False
    if is_answer_row and rec["answer_correct"] is not None:
        ok = ok and rec["answer_correct"]
    if cls == "E5" and action == "DIE":
        ok = ok and str(payload).strip().upper().startswith("TASK TOO LARGE")
    if action == "SPAWN" and g_action == "SPAWN":
        ok = ok and bool(rec["coverage_ok"]) and bool(rec["nesting_ok"])
    rec["row_correct"] = bool(ok)
    return rec


def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_file(rows_path, gens_path, out_path, tok=None):
    tok = tok or load_tok()
    rows = {r["id"]: r for r in read_jsonl(rows_path)}
    gens = read_jsonl(gens_path)
    missing = [g["id"] for g in gens if g["id"] not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} generations have no row (first: {missing[0]})")
    if len(gens) != len(rows):
        print(f"warning: {len(gens)} generations for {len(rows)} rows", file=sys.stderr)
    recs = [score_row(rows[g["id"]], g, tok) for g in gens]
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return recs


# ------------------------------------------------------------------ summaries

METRICS = ("row_correct", "valid", "action_ok", "answer_correct", "invented", "arith_error",
           "stopped", "stopped_eos", "coverage_ok", "nesting_ok")


def summarise(recs):
    def rate(rs, k):
        vals = [r[k] for r in rs if r[k] is not None]
        return (sum(bool(v) for v in vals) / len(vals), len(vals)) if vals else (None, 0)
    out = {"pooled": {k: rate(recs, k) for k in METRICS}, "n": len(recs)}
    by = collections.defaultdict(list)
    for r in recs:
        by[r["class"]].append(r)
    out["by_class"] = {c: {k: rate(rs, k) for k in ("row_correct", "valid", "action_ok",
                                                     "answer_correct", "invented")}
                       for c, rs in sorted(by.items())}
    ex = [r for r in recs if r["role"] == "executor"]
    out["executor_stopped"] = rate(ex, "stopped")
    # G2's rate: among the valid REPORTs, so an arm that reports more often is
    # not charged for it.
    out["invented_report"] = rate([r for r in recs if r["action"] == "REPORT" and r["valid"]], "invented")
    return out


def print_summary(name, s):
    p = s["pooled"]
    print(f"== {name}  ({s['n']} rows)")
    print("   " + "  ".join(f"{k} {v[0]:.3f}" for k, v in p.items() if v[0] is not None))
    print(f"   executor stopped {s['executor_stopped'][0]:.3f} (n={s['executor_stopped'][1]})")
    for c, m in s["by_class"].items():
        print(f"   {c:3s} " + "  ".join(f"{k} {v[0]:.2f}" for k, v in m.items() if v[0] is not None)
              + f"  (n={m['row_correct'][1]})")


def _by_seed(files):
    """{row id: [row_correct per file]} and {row id: family}."""
    vals, fam = collections.defaultdict(dict), {}
    for k, path in enumerate(files):
        for r in read_jsonl(path):
            vals[r["id"]][k] = float(bool(r["row_correct"]))
            fam[r["id"]] = r["family"]
    return vals, fam


def cluster_bootstrap(base, arm, ids, fam, n_base, n_arm, reps=4000, seed=0):
    """(mean, lo, hi) of arm - base row_correct. Each replicate resamples
    families (a family's rows move together) and, independently, the seeds of
    each side -- families crossed with seeds (spec §9)."""
    by = collections.defaultdict(list)
    for i in ids:
        by[fam[i]].append(i)
    fams = sorted(by)
    rng = random.Random(seed)

    def mean_diff(rows, kb, ka):
        d = [sum(arm[i][k] for k in ka) / len(ka) - sum(base[i][k] for k in kb) / len(kb) for i in rows]
        return sum(d) / len(d)
    point = mean_diff(ids, range(n_base), range(n_arm))
    stats = []
    for _ in range(reps):
        rows = [i for f in (rng.choice(fams) for _ in fams) for i in by[f]]
        kb = [rng.randrange(n_base) for _ in range(n_base)]
        ka = [rng.randrange(n_arm) for _ in range(n_arm)]
        stats.append(mean_diff(rows, kb, ka))
    stats.sort()
    return point, stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def gates(base_files, arm_files):
    base, fam = _by_seed(base_files)
    arm, fam2 = _by_seed(arm_files)
    fam.update(fam2)
    ids = sorted(i for i in set(base) & set(arm)
                 if len(base[i]) == len(base_files) and len(arm[i]) == len(arm_files))
    if len(ids) != len(base) or len(ids) != len(arm):
        print(f"warning: paired on {len(ids)} rows (base {len(base)}, arm {len(arm)})")
    point, lo, hi = cluster_bootstrap(base, arm, ids, fam, len(base_files), len(arm_files))
    base_s = summarise([r for p in base_files for r in read_jsonl(p)])
    arm_s = summarise([r for p in arm_files for r in read_jsonl(p)])
    per_seed = [summarise(read_jsonl(p))["pooled"]["row_correct"][0] for p in arm_files]
    base_rc_pooled = base_s["pooled"]["row_correct"][0]
    g1 = lo > 0 and all(s >= base_rc_pooled for s in per_seed)
    g2 = (arm_s["invented_report"][0] or 0.0) <= (base_s["invented_report"][0] or 0.0)
    g3 = arm_s["executor_stopped"][0] >= 0.95
    print_summary("base", base_s)
    print_summary("arm", arm_s)
    n_fam = len(set(fam[i] for i in ids))
    print(f"\nrow_correct arm - base: {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] "
          f"(bootstrap over {n_fam} families x seeds; with few families the interval is coarse)")
    print(f"per-seed arm row_correct: {', '.join(f'{s:.3f}' for s in per_seed)} vs base {base_rc_pooled:.3f}")
    print(f"G1 beats base (lower bound > 0, no seed below base): {'PASS' if g1 else 'FAIL'}")
    print(f"G2 invented among valid REPORTs {arm_s['invented_report'][0]:.3f} "
          f"<= base {base_s['invented_report'][0]:.3f}: {'PASS' if g2 else 'FAIL'}")
    print(f"G3 executor stopped {arm_s['executor_stopped'][0]:.3f} >= 0.95: {'PASS' if g3 else 'FAIL'}")
    return {"G1": g1, "G2": g2, "G3": g3, "diff": (point, lo, hi)}


def selftest(rows_path, limit=None):
    """Gold responses as generations: every row must score row_correct."""
    tok = load_tok()
    rows = read_jsonl(rows_path)[:limit]
    bad = []
    recs = []
    for r in rows:
        rec = score_row(r, {"id": r["id"], "text": r["response"], "eos": True, "n_new": None}, tok)
        recs.append(rec)
        if not rec["row_correct"]:
            bad.append(rec)
    print_summary("gold", summarise(recs))
    for b in bad[:20]:
        print("FAIL", b["id"], {k: b[k] for k in ("action", "valid", "action_ok", "answer_correct",
                                                   "invented", "coverage_ok", "nesting_ok", "why")})
    return recs, bad


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--rows", required=True)
    s.add_argument("--gens", required=True)
    s.add_argument("--out", required=True)
    g = sub.add_parser("gates")
    g.add_argument("--base", nargs="+", required=True)
    g.add_argument("--arm", nargs="+", required=True)
    t = sub.add_parser("selftest")
    t.add_argument("--rows", required=True)
    t.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.cmd == "score":
        print_summary(os.path.basename(a.gens), summarise(score_file(a.rows, a.gens, a.out)))
    elif a.cmd == "gates":
        gates(a.base, a.arm)
    else:
        _, bad = selftest(a.rows, a.limit)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
