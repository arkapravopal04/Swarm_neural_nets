"""
fine_tune_v2/validate.py -- every row through the runtime's own code
(TRAINING_SPEC.md §8). A check returns None on pass or a short reason.

V1  decide() itself parses the gold: a stand-in model streams the gold tokens
    through decide()'s real stopping criterion, so a row the runtime would
    cut early, misread, repair or re-sample fails here.
V1b a gold SPAWN batch goes through Agent.request_spawn and the orchestrator's
    real batch handling (no embedder: embedding gates are not exercised).
V2  figures: text_utils.classify_figures with the chain passes of
    Orchestrator._ungrounded_figures; sources = request, task, dependency /
    child results and tool outputs (spec §8; the runtime itself omits tool
    output -- see TRAINING_SPEC risk note).
V3  arithmetic: every "a op b = c" in the gold is recomputed.
V6  lint with the runtime's trims/detectors: the REPORT must come out of the
    orchestrator's pre-judge cleaning (drop_incomplete_tail,
    cut_prompt_header_echo, trim_degenerate_tails, trim_to_sentences) unchanged.
V7  lengths (tokens under the S1 chat helper).
V8  contamination against every benchmark prompt.
"""
import json
import re

import torch

import text_utils as tu
from agent_node import (Agent, EXEMPLAR_SUBTASK_DESCRIPTIONS, PROMPT_EXEMPLARS,
                        StoppingCriteriaList)
from render import PRODUCTION_ROLES, build_agent

# ---------------------------------------------------------------------- V1


class _GoldModel:
    """Stands in for the model inside decide(): yields the gold response one
    token at a time and stops when decide()'s own stopping criteria fire, as
    model.generate() would. Counts calls so a re-sample (degenerate retry)
    is visible."""

    def __init__(self, tokenizer, response):
        self.tok = tokenizer
        self.ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        self.calls = 0
        self.prompt_text = None
        self.device = torch.device("cpu")

    def generate(self, input_ids=None, attention_mask=None, stopping_criteria=None, **kw):
        self.calls += 1
        self.prompt_text = self.tok.decode(input_ids[0])
        seq = input_ids
        crit = stopping_criteria or StoppingCriteriaList([])
        for t in self.ids:
            seq = torch.cat([seq, torch.tensor([[t]])], dim=1)
            res = crit(seq, None)
            if bool(res) if isinstance(res, bool) else bool(res.all()):
                break
        return seq


def v1_decide(state, response, tokenizer, prompt):
    agent = build_agent(state)
    gm = _GoldModel(tokenizer, response)
    agent.model = gm
    agent.tokeniser = tokenizer
    fail_before = agent.fail_reason
    try:
        action, payload = agent.decide(PRODUCTION_ROLES, list(state.available_tools),
                                       list(state.requirements))
    except Exception as e:  # a crash is a fail
        return f"V1 decide crashed: {type(e).__name__}: {e}"
    if gm.calls != 1:
        return f"V1 decide re-sampled ({gm.calls} calls)"
    if gm.prompt_text is not None and gm.prompt_text != prompt:
        return "V1 decide built a different prompt than the renderer"
    want = response.split("\n", 1)[0].split(":", 1)[1].strip()
    if action != want:
        return f"V1 action {action} != gold {want}"
    if agent.fail_reason not in (None, fail_before):
        return f"V1 decide set fail_reason: {str(agent.fail_reason)[:80]}"
    body = response.split("PAYLOAD:", 1)[1].strip()
    if want in ("SPAWN", "TOOL"):
        if not isinstance(payload, dict) or payload != json.loads(body):
            return "V1 payload changed by decide()"
    elif str(payload).strip() != body:
        return "V1 payload text changed by decide()"
    return None


# ---------------------------------------------------------------------- V1b

def v1b_spawn(state, response):
    """Real request_spawn + orchestrator batch handling for a SPAWN gold."""
    if not response.startswith("ACTION: SPAWN"):
        return None
    from colony_state import AgentNode, ColonyState
    from event_queue import Messenger
    from orchestrator import Orchestrator
    from task_graph import TaskGraph, TaskNode

    payload = json.loads(response.split("PAYLOAD:", 1)[1])
    orch = Orchestrator(ColonyState(initial_budget=5000, goal_embedding=None), TaskGraph(), Messenger())
    orch.spec = {"raw_text": state.request, "goal": state.goal or state.request,
                 "requirement": list(state.requirements), "supplies_data": True}
    tid = "t-parent"
    orch.task_graph.add_task(TaskNode(task_id=tid, description=state.task, agent_id="p", status=1,
                                      required_role="decomposer"))
    if state.generation == 0:
        orch.root_task_id = tid
    node = AgentNode(agent_id="p", role="decomposer", status="running",
                     parent_id=None, task=state.task, task_id=tid, generation=state.generation,
                     request_text=state.request)
    orch.colony.register_agent(node)
    # Siblings that already completed (a U2 re-SPAWN): without them the
    # coverage fill would append their parts again.
    for i, row in enumerate(state.children or []):
        if row.get("status") == "completed":
            orch.task_graph.add_task(TaskNode(task_id=f"t-done-{i}", description=row["task"],
                                              status=2, result=row.get("excerpt") or "done",
                                              parent_task_id=tid, label=row.get("label")))
    before = {t.task_id for t in orch.task_graph.tasks.values()}
    node.has_spawned = bool(state.has_spawned)
    agent = Agent(tokeniser=None, model=None, message=orch.messenger, node=node)
    orch.live_agents["p"] = agent
    try:
        agent.request_spawn(payload)
        orch._route_events(orch.messenger.drain())
    except Exception as e:
        return f"V1b spawn crashed: {type(e).__name__}: {e}"
    kids = [t for t in orch.task_graph.tasks.values()
            if t.parent_task_id == tid and t.task_id not in before]
    queued = sum(len(v) for v in getattr(orch, "pending_overflow", {}).values()) \
        if isinstance(getattr(orch, "pending_overflow", None), dict) else 0
    want = len(payload["subtasks"])
    if len(kids) + queued != want:
        return f"V1b batch changed: {len(kids)} spawned + {queued} queued != {want}"
    roles = {t.description: t.required_role for t in kids}
    for sub in payload["subtasks"]:
        got = roles.get(sub["task"])
        if got is not None and got != sub["role"]:
            return f"V1b role changed: {sub['role']} -> {got}"
    for sub in payload["subtasks"]:
        if tu.is_exemplar_echo(sub["task"], EXEMPLAR_SUBTASK_DESCRIPTIONS):
            return "V1b subtask copies a prompt exemplar"
    return None


# ---------------------------------------------------------------------- V2

def _sources(state):
    parts = [state.request, state.task]
    parts += [d.result for d in state.dependencies]
    parts += [text for _, text in state.child_results]
    if state.children:
        parts += [r.get("excerpt") or "" for r in state.children]
    parts += [str(c.result.get("data") or "") for c in state.tool_calls]
    return "\n".join(parts)


def v2_figures(state, response, estimate_ok=False):
    if not response.startswith("ACTION: REPORT"):
        return None
    text = response.split("PAYLOAD:", 1)[1].strip()
    source = _sources(state)
    first = tu.classify_figures(source, text)
    passed = {e["value"] for e in first if e["class"] != tu.FIGURE_NEITHER}
    for _ in range(2):  # Orchestrator._ungrounded_figures' chain passes
        steps = " ".join(e["written"] for e in first if e["value"] in passed)
        newly = [e for e in tu.classify_figures(source + "\n" + steps, text)
                 if e["class"] == tu.FIGURE_DERIVABLE and e["value"] not in passed]
        if not newly:
            break
        passed |= {e["value"] for e in newly}
    small = {e["value"] for e in first if float(e["value"]).is_integer() and abs(float(e["value"])) <= 10}
    loose = [e for e in first if e["value"] not in passed | small]
    if estimate_ok and loose:
        hedged = tu.hedged_figure_values(text)
        pool = set(tu.figure_values(source)) | passed | small | set(hedged)
        rest = []
        for e in loose:
            if e["value"] in hedged:
                continue
            pairs = [(a, b) for a in pool for b in pool if a != b]
            if tu._one_op_match(e["value"], pairs) is not None:
                pool.add(e["value"])
                continue
            rest.append(e)
        loose = rest
    if loose:
        return "V2 ungrounded figure(s): " + ", ".join(e["written"] for e in loose[:4])
    return None


# ---------------------------------------------------------------------- V3

_NUM = r"\d[\d,]*(?:\.\d+)?"
_EQ_RE = re.compile(rf"((?:{_NUM}\s*[x×*/+\-]\s*)+{_NUM})\s*=\s*({_NUM})")


def v3_arithmetic(response):
    """Every written equation, including chains ("a + b + c = d"), is
    evaluated whole with the usual precedence."""
    for left, right in _EQ_RE.findall(response):
        expr = left.replace(",", "").replace("x", "*").replace("×", "*")
        try:
            got = eval(expr, {"__builtins__": {}}, {})  # digits and operators only
        except ZeroDivisionError:
            return f"V3 division by zero: {left}"
        want = float(right.replace(",", ""))
        if abs(got - want) > max(1e-6, 5e-9 * abs(want)):
            return f"V3 arithmetic: {left} = {got} != {right}"
    return None


# ---------------------------------------------------------------------- V4 / V5

def v4_answer(response, answer_text):
    """The solver's answer, as printed, appears in an answer row's REPORT."""
    if answer_text and response.startswith("ACTION: REPORT") and answer_text not in response:
        return f"V4 solver answer {answer_text} not in the REPORT"
    return None


def v5_program(response, printed, domain):
    """A gold run_code program, run in the runtime's own sandbox with the
    family's domain preamble, prints exactly the solver's value."""
    if not response.startswith("ACTION: TOOL") or printed is None:
        return None
    from tools import ToolRegistry
    args = json.loads(response.split("PAYLOAD:", 1)[1])["args"]
    r = ToolRegistry.execute("run_code", args, domain=domain)
    out = str(r.get("data") or "").replace("\r\n", "\n").strip()
    if r.get("status") != "success" or out != printed:
        return f"V5 program printed {out[:40]!r} ({r.get('status')}), solver says {printed!r}"
    return None


# ---------------------------------------------------------------------- V6

_FORBIDDEN = [
    (re.compile(r"\bwait\b", re.I), "says 'Wait'"),
    (re.compile(r"final answer\s*:", re.I), "'Final answer:' preamble"),
    (re.compile(r"\\\(|\\\[|\$[^$]+\$|\\frac"), "LaTeX"),
    (re.compile(r"^\s*\|.*\|\s*$", re.M), "markdown table"),
    (re.compile(r"^\s*#+\s", re.M), "heading"),
    (re.compile(r"<\/?think>"), "think tag"),
    (re.compile(r"\b(perfect|flawless|fully verified|this plan is (?:complete|ideal))\b", re.I),
     "self-assessment"),
]


def v6_lint(state, response, n_children=0):
    if response.startswith("\n") or "\n\n" in response.split("PAYLOAD:", 1)[-1]:
        return "V6 leading newline or blank line (the stop criterion ends on a blank line)"
    for rx, why in _FORBIDDEN:
        if rx.search(response):
            return f"V6 {why}"
    if '"verifier"' in response:
        return "V6 verifier role"
    if state.role == "decomposer" and response.startswith("ACTION: TOOL"):
        return "V6 decomposer TOOL"
    if not response.startswith("ACTION: REPORT"):
        return None
    text = response.split("PAYLOAD:", 1)[1].strip()
    if tu.is_exemplar_echo(text, PROMPT_EXEMPLARS):
        return "V6 REPORT copies a prompt exemplar"
    if tu.looks_degenerate(text):
        return "V6 looks degenerate"
    grounding = state.request + "\n" + state.task
    out = tu.drop_incomplete_tail(text)
    out, _ = tu.cut_prompt_header_echo(out)
    out, _ = tu.trim_degenerate_tails(out, grounding)
    if state.role == "decomposer":
        sents = min(8, max(3, n_children + 1))
        words = max(60, sents * 25)
    else:
        sents, words = 3, 60
    out = tu.trim_to_sentences(out, max_sentences=sents, max_words=words, marker=False)
    if out.strip() != text:
        return "V6 runtime pre-judge trim changes the REPORT"
    return None


# ---------------------------------------------------------------------- V7

# TOOL 260: gold programs carry the 3-line print helper (spec §7.2 amended, §17).
LIMITS = {"REPORT": 180, "SPAWN": 350, "TOOL": 260, "DIE": 120}


def v7_lengths(response, tokenizer):
    action = response.split("\n", 1)[0].split(":", 1)[1].strip()
    n = len(tokenizer(response, add_special_tokens=False)["input_ids"])
    if n > LIMITS.get(action, 200):
        return f"V7 {action} is {n} tokens > {LIMITS[action]}"
    return None


# ---------------------------------------------------------------------- V8

def ngrams(text, n=8):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def figure_tuple(text):
    return tuple(sorted(tu.figure_values(text)))


class Contamination:
    """Benchmark prompts are read only into this index (hashes, figure tuples,
    embeddings); nothing here prints or stores their text."""

    def __init__(self, bench_texts, embed, threshold):
        self.grams = set()
        for t in bench_texts:
            self.grams |= ngrams(t)
        self.tuples = {figure_tuple(t) for t in bench_texts if len(tu.figure_values(t)) >= 3}
        self.embed = embed
        self.vecs = embed(bench_texts) if embed else None
        self.threshold = threshold

    def check(self, request):
        if ngrams(request) & self.grams:
            return "V8 8-gram overlap with a benchmark prompt"
        ft = figure_tuple(request)
        if len(ft) >= 3 and ft in self.tuples:
            return "V8 figure tuple equals a benchmark item's"
        if self.embed is not None:
            v = self.embed([request])
            sim = float((self.vecs @ v.T).max())
            if sim >= self.threshold:
                return f"V8 embedding cosine {sim:.3f} >= {self.threshold}"
        return None
