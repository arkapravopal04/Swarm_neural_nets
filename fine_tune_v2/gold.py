"""
fine_tune_v2/gold.py -- behaviour classes and their gold responses
(TRAINING_SPEC.md §7).

Each builder takes a scenario (families_single.Single or families_multi.Multi)
and yields Candidate rows: the runtime State the agent is in (rendered later by
render.render) and the gold decide() output. The non-gold text in a state --
previous thoughts, verdicts, tool results, child results -- is TEMPLATED here
(source "templated"): the spec's harvested text (§5, >= 70%) needs GPU colony
runs and is the v2.1 data step.

Verdict and error strings are the runtime's own, copied from orchestrator.py
and agent_node.py (S6's prompt-builder hash covers them).
"""
import json
from dataclasses import dataclass, field

from agent_node import MAX_DECOMPOSER_GENERATION, may_convert_to_decomposer
from text_utils import first_clause
from families_multi import first_sentence, rollup
from render import Dependency, State, ToolCall, child_row

# ----------------------------------------------------------- runtime strings

ROOT_DIRECT_REPORT_REJECT = (
    "Previous attempt was REJECTED: you REPORTed a final answer "
    "directly instead of decomposing the task. As the ROOT "
    "decomposer (generation 0) your job is to SPAWN subtasks, "
    "not answer the question yourself.")
PARSE_FAILURES = [
    ("no_action", "No ACTION: line was found in your last response, so nothing was done. "
                  "Reply with exactly one ACTION: line followed by one PAYLOAD: line."),
    ("json", "JSONDecodeError: Payload was malformed."),
    ("truncated", "Payload appears TRUNCATED (starts with '{' but never reaches a balanced "
                  "closing '}') -- likely ran out of generation budget before finishing."),
    ("empty_report", "Your REPORT had no content after the ACTION line. Put the answer after PAYLOAD:."),
]


TOO_LARGE = [
    "TASK TOO LARGE: the task has {n} separate parts ({parts}), each needing its own answer.",
    "TASK TOO LARGE: this is {n} pieces of work ({parts}) that should be split, not one answer.",
    "TASK TOO LARGE: it bundles {n} independent parts: {parts}.",
    "TASK TOO LARGE: answering it in one pass would mix {n} separate tasks ({parts}).",
    "TASK TOO LARGE: {parts} are {n} distinct parts and need separate agents.",
]
TOO_LARGE_REASONS = [
    "this part has {n} separate pieces ({parts}).",
    "it needs {parts} done separately.",
    "{parts} are separate pieces of work.",
    "it is {n} tasks in one: {parts}.",
]
NONE_DONE = [
    "None of the subtasks completed ({parts}), so there are no results to combine.",
    "Every subtask ({parts}) failed or was abandoned; there is nothing to report.",
    "No subtask finished ({parts}), so no combined result exists.",
    "All of {parts} ended without a result, so this task cannot be reported.",
    "The subtasks {parts} produced no completed results to roll up.",
]


def review_reject(critique):
    """REJECTED BY REVIEW -- (tier-3 execute; the respawn's ghost carries it)."""
    return f"REJECTED BY REVIEW -- {first_clause(critique, max_chars=120)}"


def review_warn(critique):
    """SENT BACK BY REVIEW -- (a WARN; the same agent sees it as a verdict)."""
    return f"SENT BACK BY REVIEW -- {first_clause(critique, max_chars=120)}"


# --------------------------------------------------------- templated thoughts

EXEC_THOUGHTS = [
    "Okay, the task asks me to {t}. The figures I need are in the request: {f}. First I should",
    "Let me see. I have to {t}. I'll take {f} straight from the request and work out",
    "Okay, so the question is to {t}. The numbers given are {f}. The careful way is to",
    "Alright, I need to {t}. Using {f} from the request, I can",
    "The task is to {t}. Key figures: {f}. I should check whether",
]
DEC_THOUGHTS = [
    "Okay, let's break this down. My task has {k} parts: {p}. Each part can be",
    "Let me see. The task covers {p}, so that is {k} separate pieces. I should give each",
    "Alright, there are {k} things to do here: {p}. Some of these",
    "Okay, so I need to split this into {p}. That's {k} pieces, and I should check which",
    "The work divides into {k} parts: {p}. I'll keep each child's task in the request's own",
]


def exec_thought(rng, task, figs):
    t = task.rstrip(".")
    t = t[0].lower() + t[1:]
    if len(t) > 110:
        t = t[:110].rsplit(" ", 1)[0]
    f = ", ".join(figs[:4]) if figs else "the values given"
    return rng.choice(EXEC_THOUGHTS).format(t=t, f=f)


def dec_thought(rng, labels):
    return rng.choice(DEC_THOUGHTS).format(k=len(labels), p=", ".join(labels))


# ----------------------------------------------------------------- responses

def r_spawn(children):
    """children: [(label, role, task, deps)] -> SPAWN gold, one JSON line."""
    payload = {"subtasks": [{"label": l, "role": r, "task": t, "dependencies": list(d)}
                            for l, r, t, d in children]}
    return "ACTION: SPAWN\nPAYLOAD: " + json.dumps(payload, ensure_ascii=False)


def r_tool(code):
    return "ACTION: TOOL\nPAYLOAD: " + json.dumps(
        {"tool_name": "run_code", "args": {"code_string": code}}, ensure_ascii=False)


def r_report(text):
    return "ACTION: REPORT\nPAYLOAD: " + text


def r_die(text):
    return "ACTION: DIE\nPAYLOAD: " + text


@dataclass
class Candidate:
    cls: str
    state: State
    response: str
    family: str
    domain: str
    scenario_id: str
    source: str = "solver+templated"
    meta: dict = field(default_factory=dict)


def _figs(text):
    """Request figures for a thought, without the answer-format sentence's
    precision ("at least 6 decimal places" is not a figure of the problem)."""
    from text_utils import figure_values
    text = text.split(" Give the answer to")[0].split("\n\nOptions:")[0]
    return list(figure_values(text).values())


def single_requirements(request):
    """Phaser-style constraint lines for a single question: its sentences that
    state figures (the phaser keeps those), minus the question and the
    answer-format sentence."""
    from text_utils import split_sentences
    body = request.split("\n\nOptions:")[0]
    out = []
    for sent in split_sentences(body):
        sent = sent.strip()
        if (not sent or sent.endswith("?") or sent.startswith("Give the answer")
                or not any(ch.isdigit() for ch in sent)):
            continue
        out.append(sent if sent.endswith(".") else sent + ".")
    return out[:5]


# ----------------------------------------------------------- single question

def single_rows(s, sid, rng):
    """R4 and E-class rows for one single-question scenario."""
    out = []
    task = s.request  # R4 hands the question down verbatim
    figs = _figs(s.request)
    reqs = single_requirements(s.request)
    # Like the runtime: the root gets the phaser's full list (when it found
    # any); an executor gets a filtered subset, and the goal referent when
    # that subset is empty.
    root_reqs = reqs if rng.random() < 0.75 else []
    exec_reqs = reqs if (reqs and rng.random() < 0.5) else []

    def ex_state(**kw):
        base = dict(role="executor", generation=1, request=s.request, task=task,
                    requirements=list(exec_reqs), goal_referent=None if exec_reqs else s.request,
                    thought=exec_thought(rng, "answer the question", figs))
        base.update(kw)
        return State(**base)

    def C(cls, state, resp, **meta):
        meta.setdefault("printed", s.printed)
        meta.setdefault("tool_domain", s.domain)
        out.append(Candidate(cls, state, resp, s.family, s.domain, sid, meta=meta))

    # R4: the root passes a single question down whole.
    C("R4", State(role="decomposer", generation=0, request=s.request, task=s.request,
                  requirements=list(root_reqs), thought=dec_thought(rng, ["the one question"])),
      r_spawn([("q", "executor", task, [])]))

    short = s.steps is not None
    tool = bool(s.code) and s.tool_ok
    if short:
        C("E1", ex_state(), r_report(" ".join(s.steps)))
    if tool:
        C("E2", ex_state(), r_tool(s.code))
        result = {"status": "success", "data": s.printed + "\n"}
        prior = r_tool(s.code)
        e3 = s.answer_sentence
        C("E3", ex_state(tool_calls=[ToolCall("run_code", result)], earlier_choices=[prior],
                         thought=""), r_report(e3), printed=s.printed)
    good = r_report(" ".join(s.steps)) if short else (r_tool(s.code) if tool else None)
    if s.wrong and good:
        wrong_text, critique = s.wrong
        # WARN on the same agent: its wrong REPORT is in its thoughts.
        C("E6", ex_state(earlier_choices=[r_report(wrong_text)], fail_reason=review_warn(critique)),
          good, variant="warn")
        # Rejected and respawned: the critique arrives in the ghost.
        C("E6", ex_state(respawn_failures=[review_reject(critique)]), good, variant="respawn")
    if short and s.family not in ("compliance_choice", "comms_choice"):
        # A wrong critique of a right answer: the figure stands.
        bogus = f"The answer {s.printed} looks wrong; check the arithmetic."
        C("E6", ex_state(earlier_choices=[r_report(" ".join(s.steps))],
                         fail_reason=review_warn(bogus)), r_report(" ".join(s.steps)), variant="stands")
    if good:
        C("E7", ex_state(respawn_failures=[
            "Previous attempt DIED with reason: I cannot verify this value without a lookup."]), good)
    if short and s.tool_ok and s.family not in ("compliance_choice", "comms_choice"):
        bad_code = "print(x)"
        fails = [ToolCall("run_code", {"status": "error", "reason": "script_crash",
                                       "message": "NameError: name 'x' is not defined"})] * 3
        hand = " ".join(s.steps) + " This was worked by hand; run_code could not confirm it."
        C("E8", ex_state(tool_calls=fails, earlier_choices=[r_tool(bad_code)] * 3), r_report(hand),
          variant="circuit_open")
        ok = [ToolCall("run_code", {"status": "success", "data": "done\n"})] * 4
        C("E8", ex_state(tool_calls=ok, earlier_choices=[r_tool(s.code)] * 4, tool_call_count=4),
          r_report(" ".join(s.steps)), variant="budget")
    if good:
        kind, msg = PARSE_FAILURES[rng.randrange(len(PARSE_FAILURES))]
        C("E9", ex_state(fail_reason=msg), good, variant=kind)
    return out


# --------------------------------------------------------------- multi part

def _part_children(parts, parent_generation):
    may_nest = parent_generation <= MAX_DECOMPOSER_GENERATION
    return [(p.label, p.role if (p.role != "decomposer" or may_nest) else "executor",
             p.text, p.deps) for p in parts]


def _reqs_for(part, reqs):
    keys = tuple(k.lower() for k in part.keywords)
    return [r for r in reqs if any(k in r.lower() for k in keys)]


def _deps_for(part, siblings):
    by = {p.label: p for p in siblings}
    return [Dependency(by[d].text.rstrip("."), by[d].final_report or rollup(by[d].pieces))
            for d in part.deps if d in by]


def multi_rows(m, sid, rng):
    out = []

    def C(cls, state, resp, **meta):
        out.append(Candidate(cls, state, resp, m.family, m.domain, sid, meta=meta))

    labels = [p.label for p in m.parts]
    cls = "R3" if any(p.role == "decomposer" for p in m.parts) else (
        "R2" if any(p.deps for p in m.parts) else "R1")
    root_children = _part_children(m.parts, 0)
    root = dict(role="decomposer", generation=0, request=m.request, task=m.request,
                requirements=list(m.requirements))
    C(cls, State(**root, thought=dec_thought(rng, labels)), r_spawn(root_children))
    # R5: the root answered directly and was structurally rejected.
    C("R5", State(**root, earlier_choices=[r_report("Here is the full plan: " + first_sentence(
        m.parts[0].final_report or rollup(m.parts[0].pieces)))], fail_reason=ROOT_DIRECT_REPORT_REJECT),
      r_spawn(root_children))
    # E5: an executor handed the whole request.
    C("E5", State(role="executor", generation=1, request=m.request, task=m.request,
                  thought=exec_thought(rng, "do the whole request", _figs(m.request))),
      r_die(rng.choice(TOO_LARGE).format(n=len(m.parts), parts=", ".join(labels))))

    # Roll-up states for every decomposer level (root and nested).
    levels = [(m.parts, 0, m.request)] + [(d.pieces, g, d.text) for d, g in m.decomposers()]
    for parts, gen, task in levels:
        rows_done = [child_row(p.text, "completed", p.label,
                               p.final_report or rollup(p.pieces)) for p in parts]
        base = dict(role="decomposer", generation=gen, request=m.request, task=task,
                    requirements=list(m.requirements), has_spawned=True,
                    earlier_choices=[r_spawn(_part_children(parts, gen))])
        last = parts[-1]
        C("U1", State(**base, children=rows_done,
                      child_results=[(f"task_{last.label}", last.final_report or rollup(last.pieces))]),
          r_report(rollup(parts)))
        # U2: one subtask without dependencies failed; redo only that part.
        redo = [p for p in parts if not p.deps and not any(p.label in q.deps for q in parts)]
        if len(parts) > 1 and redo:
            bad = rng.choice(redo)
            rows = [child_row(p.text, "abandoned" if p is bad else "completed", p.label,
                              None if p is bad else (p.final_report or rollup(p.pieces))) for p in parts]
            C("U2", State(**base, children=rows),
              r_spawn([(bad.label + "-redo", *(_part_children([bad], gen)[0][1:3]), [])]))
        # U3: nothing completed.
        rows_none = [child_row(p.text, rng.choice(["abandoned", "failed"]), p.label) for p in parts]
        C("U3", State(**base, children=rows_none),
          r_die(rng.choice(NONE_DONE).format(parts=", ".join(p.label for p in parts))))

    # Nested decomposers: N2 (may nest) / N1 (executors only), and conversions.
    for d, gen in m.decomposers():
        children = _part_children(d.pieces, gen)
        n_cls = "N2" if any(r == "decomposer" for _, r, _, _ in children) else "N1"
        st = dict(role="decomposer", generation=gen, request=m.request, task=d.text,
                  requirements=list(m.requirements))
        C(n_cls, State(**st, thought=dec_thought(rng, [p.label for p in d.pieces])), r_spawn(children))
        if may_convert_to_decomposer(gen):
            C(n_cls, State(**st, respawn_failures=[
                "Previous attempt DIED with reason: TASK TOO LARGE: "
                + rng.choice(TOO_LARGE_REASONS).format(n=len(d.pieces),
                                                       parts=", ".join(p.label for p in d.pieces))]),
              r_spawn(children), variant="conversion")

    # Executor leaves.
    def walk(parts, gen):
        for p in parts:
            if p.role == "decomposer":
                walk(p.pieces, gen + 1)
                continue
            reqs = _reqs_for(p, m.requirements)
            st = dict(role="executor", generation=gen, request=m.request, task=p.text,
                      requirements=reqs, goal_referent=None if reqs else m.request,
                      dependencies=_deps_for(p, parts))
            th = exec_thought(rng, p.text, _figs(p.text) or _figs(m.request))
            if p.code:
                C("E2", State(**st, thought=th), r_tool(p.code), printed=p.printed, tool_domain=m.domain)
                C("E3", State(**st, tool_calls=[ToolCall("run_code", {"status": "success",
                                                                       "data": p.printed + "\n"})],
                              earlier_choices=[r_tool(p.code)]), r_report(p.after_tool))
                continue
            C("E4" if p.estimate else "E1", State(**st, thought=th), r_report(p.report))
            if p.wrong:
                wrong_text, critique = p.wrong
                C("E6", State(**st, earlier_choices=[r_report(wrong_text)],
                              fail_reason=review_warn(critique)), r_report(p.report), variant="warn")
                C("E6", State(**st, respawn_failures=[review_reject(critique)]), r_report(p.report),
                  variant="respawn")
            if rng.random() < 0.3:
                kind, msg = PARSE_FAILURES[rng.randrange(len(PARSE_FAILURES))]
                C("E9", State(**st, fail_reason=msg), r_report(p.report), variant=kind)
    walk(m.parts, 1)
    return out
