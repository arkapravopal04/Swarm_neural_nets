"""
judge.py

Tiered verification of agent outputs for Project Hive. Decides life and death.

Three tiers, run in escalating order, each gated behind the previous passing:

    Tier 1 -- fast_check     : sub-10ms syntax/execution check, runs on every output
    Tier 2 -- semantic_check : cosine similarity vs the target embedding the
                               caller supplies for THIS task (its own subtask
                               description, or the colony goal only for the
                               root task) -- runs only if tier 1 passed, and
                               is skipped entirely for a decomposer (a plan
                               isn't supposed to resemble the answer)
    Tier 3 -- deep_critique  : full LLM call, runs only when the caller flags
                               the output as needing a deep check (a child
                               promoting a result to its parent, or the root
                               task's final answer)

Energy-conservation principle: a tier 1 failure short-circuits immediately --
tiers 2 and 3 never run on broken output. Running an expensive check to
confirm something already known to be broken wastes energy the colony needs
elsewhere, and pollutes the eventual ghost record with noise instead of signal.

Judge does not own an embedding model. semantic_check takes pre-computed
embedding vectors (output_embedding, target_embedding) as arguments -- the
caller (agent_node / problem_parser) is responsible for producing them. This
keeps judge decoupled from any specific embedding backend.
"""

import re
import numpy as np
from text_utils import dedupe_and_cap as _dedupe_and_cap


# --- Tier 2 thresholds -----------------------------------------------------
# The entire tuning surface for semantic drift detection. Adjust these two
# numbers after watching real colony runs -- if good agents get executed or
# drifting agents survive, these are the knobs to turn.
SEMANTIC_WARN_THRESHOLD = 0.6      # below this -> warn
SEMANTIC_EXECUTE_THRESHOLD = 0.3   # at or below this -> execute, too far gone

# Strike system: after this many warnings, skip the similarity check
# entirely on the next tier 2 pass and execute regardless of current score.
MAX_WARNINGS = 3

# PATCH 41: WARNs a task may draw (across all its attempts) before a further
# low-similarity REPORT goes to the tier-3 critique instead of another WARN.
WARN_ESCALATE_AFTER = 1

# --- Tier 1 empty-answer detection ------------------------------------------
# A single-line REPORT matching either pattern is a label, not an answer --
# an ATX markdown heading ("## Summary") or a line wrapped entirely in bold
# markup ("**Conclusion**") with nothing else in the payload.
_HEADING_ONLY_RE = re.compile(r'^\s*#{1,6}\s+\S.*$')
_BOLD_ONLY_RE = re.compile(r'^\s*\*{1,3}[^*\n]+\*{1,3}\s*$')

# Common LLM non-answers that survive the "non-empty" check but carry no
# actual content. Compared against the REPORT with surrounding punctuation
# and markdown decoration stripped and casefolded.
_PLACEHOLDER_REPORTS = {
    "n/a", "na", "tbd", "todo", "none", "null", "done", "ok", "okay",
    "no answer", "no content", "pending", "n a",
}


class Judge:
    """
    Tiered verification of agent outputs. One instance shared by the
    orchestrator, called once per agent output (routine check) or once per
    promotion attempt (graduation check).

    llm_call_fn is bound once here, not passed per-call -- the orchestrator
    constructs a single Judge and never touches this again. Injected rather
    than owned outright so judge.py stays decoupled from any specific model
    client (mirrors how semantic_check takes pre-computed embedding vectors
    instead of owning an embedding model -- see semantic_check docstring for
    why: target_embedding and output_embedding must come from the SAME
    embedding model instance for cosine similarity to mean anything, and
    that instance belongs wherever problem_parser's generate_embedding()
    lives, not duplicated here).
    """

    # deep_critique's prompt ends with this, so the ruling is the first word
    # of the completion instead of whatever preamble the model opens with.
    VERDICT_PREFILL = "VERDICT:"

    # A line the model starts after its ruling that is not critique: an
    # invented few-shot sample ("Example:", "Example 2:") or another
    # VERDICT line. Markdown bullets/emphasis in front are allowed. The
    # colon is required, so reasoning that merely opens with the word
    # ("Example output lacks units") is kept.
    _INVENTED_CONTINUATION_RE = re.compile(
        r"^[\s*#>\-]*(?:example(?:\s*\d+)?\s*:|verdict\s*:)", re.IGNORECASE
    )

    def __init__(self, llm_call_fn=None):
        self.llm_call_fn = llm_call_fn

    @classmethod
    def _cut_invented_continuation(cls, text: str) -> str:
        """Keep the first line (the prefilled ruling) and the reasoning under
        it, up to the first invented Example:/VERDICT: line. Everything from
        there on is the model writing its own few-shot samples, and would
        otherwise both decide the verdict and become the reasoning a
        respawned agent is handed."""
        lines = text.splitlines()
        for i, line in enumerate(lines[1:], start=1):
            if cls._INVENTED_CONTINUATION_RE.match(line):
                return "\n".join(lines[:i]).rstrip()
        return text

    # ------------------------------------------------------------------
    # Tier 1 -- fast check
    # ------------------------------------------------------------------

    def fast_check(self, output, output_type: str) -> dict:
        """
        Cheap, near-instant validity check. Runs after every tool call / output.

        output_type: "code_result" | "math_result" | "text"

        For code_result / math_result: `output` is expected to be the dict
        already returned by ToolRegistry.execute() (e.g. run_code, verify_math).
        Their "status" field already encodes success/failure -- fast_check
        just reads it rather than re-deriving anything, since re-verification
        would duplicate work ToolRegistry already did.

        For text: presence check only (non-empty, non-None) -- there's
        nothing cheaper to check about free-form text than "did anything
        come back at all".

        Returns {"pass": bool, "error": str | None}
        """
        if output_type in ("code_result", "math_result"):
            if not isinstance(output, dict):
                return {"pass": False, "error": "malformed tool output: expected dict"}

            status = output.get("status")
            if status == "success":
                return {"pass": True, "error": None}

            # ToolRegistry already produced a condensed, path-scrubbed verdict
            # (TracebackSummarizer for code_result, SymPy error for math_result).
            # fast_check surfaces it verbatim -- this string is exactly what
            # the ghost extractor wants for "what not to do" context.
            # "message" first: it is the key ToolRegistry's error results
            # actually use (the condensed TracebackSummarizer verdict lives
            # there). Without it every code_result failure collapsed to the
            # contentless "status=error", discarding the one string the ghost
            # extractor and the agent both need.
            error = (
                output.get("message")
                or output.get("data")
                or output.get("error")
                or f"status={status}"
            )
            return {"pass": False, "error": str(error)}

        if output_type == "text":
            if output is None or not isinstance(output, str) or output.strip() == "":
                return {"pass": False, "error": "empty or missing text output"}

            stripped = output.strip()

            # FIX (empty-answer hole): the presence check above only catches
            # a REPORT that is literally empty. It let through the far more
            # common degenerate shape -- a REPORT that trails off into a
            # heading with nothing behind it ("Summary:", "## Final
            # Answer"). Confirmed in a real run: "short" was being treated
            # as "trustworthy" by everything downstream once tier 1 passed
            # it, so one of these worthless REPORTs got promoted with zero
            # actual content. These three checks close that hole without
            # touching legitimate short answers (e.g. a bare "42") --
            # none of them fire on a complete short answer, only on the
            # shapes above.
            if stripped.endswith(":"):
                return {
                    "pass": False,
                    "error": f"REPORT ends on a colon with nothing after it: {stripped!r}",
                }

            lines = [line for line in stripped.splitlines() if line.strip()]
            if len(lines) == 1 and (
                _HEADING_ONLY_RE.match(lines[0]) or _BOLD_ONLY_RE.match(lines[0])
            ):
                return {
                    "pass": False,
                    "error": f"REPORT is a bare heading with no content: {lines[0]!r}",
                }

            alnum_chars = re.sub(r"[^A-Za-z0-9]", "", stripped)
            normalized = stripped.strip(" \t\n.-_*#").lower()
            if len(alnum_chars) < 2 or normalized in _PLACEHOLDER_REPORTS:
                return {
                    "pass": False,
                    "error": f"REPORT has no substantive content: {stripped!r}",
                }

            return {"pass": True, "error": None}

        return {"pass": False, "error": f"unknown output_type: {output_type!r}"}

    # ------------------------------------------------------------------
    # Tier 2 -- semantic check
    # ------------------------------------------------------------------

    def semantic_check(self, output_embedding, target_embedding) -> float:
        """
        Cosine similarity between the agent's output embedding and the
        colony's master target embedding (from problem_parser). Returns the
        raw float score only -- decide() owns all thresholding logic, this
        method has no opinion about what the score means.
        """
        a = np.asarray(output_embedding, dtype="float32")
        b = np.asarray(target_embedding, dtype="float32")

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Tier 3 -- deep critique
    # ------------------------------------------------------------------

    @staticmethod
    def _figures_checked_note(derivations=None, request_labels=None):
        """PATCH 71. The tier-3 note when every figure checked out: the
        arithmetic code verified, with each operand's request label, and no
        ruling on whether the operands are the right quantities."""
        stated = (f"The request states: {', '.join(request_labels)}. "
                  if request_labels else "")
        if derivations:
            return (f"ARITHMETIC CHECKED BY CODE (the OUTPUT's own equations): "
                    f"{'; '.join(derivations)}. These steps are right; do not "
                    "recompute it. This concerns only that arithmetic; it is not a "
                    "ruling on the OUTPUT. " + stated
                    + "On these steps, reject only if an operand is the wrong "
                    "quantity for the subtask.\n\n")
        return ("FIGURES CHECKED: an automatic check found every figure in the OUTPUT "
                "in the request or the results it builds on. " + stated + "Still reject if a figure stands for "
                "the wrong quantity or the reasoning is wrong.\n\n")

    def deep_critique(self, output, subtask_spec, project_goal=None,
                      ungrounded_figures=None, figures_follow=False,
                      derivations_checked=None, request_labels=None,
                      tool_output=None, code_note=None,
                      labelled_estimates=None) -> dict:
        """
        Full LLM call. Only invoked when an agent is attempting to promote
        its result to its parent -- the most expensive tier, reserved for
        the one moment it actually matters.

        Uses self.llm_call_fn, bound once at Judge construction.

        Looks beyond surface correctness:
            - did the agent engage with the actual problem, or collapse it
              into something simpler?
            - is confidence calibrated -- does certainty match accuracy?
            - is the output actually on-topic relative to the subtask?

        Returns {"verdict": "accept" | "reject", "reasoning": str}
        """
        if self.llm_call_fn is None:
            raise ValueError("deep_critique requires llm_call_fn to be set at Judge construction")
        # PATCH 62: "a = b - c" entries are derivations, the rest figures.
        figures = [str(f) for f in (ungrounded_figures or [])]
        chained = [f for f in figures if " = " in f]
        loose = [f for f in figures if " = " not in f]
        # PATCH 77: figures the OUTPUT itself marks as estimates.
        estimated = [str(f) for f in (labelled_estimates or [])]

        # FIX (tier-3 scope rewrite): the docstring above has always promised
        # this critique looks at engagement depth, confidence calibration,
        # and on-topic-ness -- but the prompt itself never told the model
        # any of that; it was just "critique this", so the promised scope
        # existed only in the docstring, never in what the model actually
        # saw. Now stated explicitly.
        #
        # Also closes the leak where invented content bled into the
        # reviewed text: the old prompt ended cold on `f"OUTPUT: {output}\n"`
        # with no marker saying the output was finished. A small model reads
        # that as "continue this text", not "now critique it", and would
        # keep extending OUTPUT with its own invented content before ever
        # reaching a verdict -- which then rode into `reasoning` looking
        # like part of the thing being judged. BEGIN/END fencing plus an
        # explicit "this is finished, don't continue it" instruction mirrors
        # the same delimiter fix already used for agent_node.py's
        # [VERDICT FROM REVIEWER]/[END VERDICT] block.
        prompt = (
            "You are the judge for an AI agent colony. You will be shown a "
            "subtask and the OUTPUT an agent produced for it, then critique "
            "that output -- and only that output.\n\n"
            "Judge specifically on:\n"
            "- did the agent actually engage with the subtask, or quietly "
            "collapse it into something simpler?\n"
            "- is its confidence calibrated -- does how sure it sounds match "
            "how accurate it actually is?\n"
            "- is the output on-topic for the subtask, not a tangent or a "
            "restatement of the question?\n"
            # PATCH 8: the fourth question, and the only one that can see
            # outside the subtask. Every run-4 drift verdict was an accept
            # because the critique's whole frame of reference was
            # subtask_spec -- an ISBN deduplication answer to an ISBN
            # deduplication subtask is perfectly on-topic for itself. The
            # goal is shown as context, not as a rival task: a subtask is
            # SUPPOSED to be narrower than the project, so only a subtask
            # that could not plausibly belong to it at all is a problem.
            + ("- given the PROJECT GOAL below, is this subtask's output "
               "plausibly a piece of that project? A subtask is meant to be "
               "much narrower than the goal, so narrowness alone is fine -- "
               "flag it only if the output belongs to a different problem "
               "than the one the goal describes.\n" if project_goal else "")
            + "\n"
            + (f"PROJECT GOAL (context for the question above -- the agent "
               f"was NOT asked to do all of this): {project_goal}\n\n"
               if project_goal else "")
            + f"SUBTASK: {subtask_spec}\n\n"
            # PATCH 62. The run already knows which figures came from
            # nowhere; the critique read prose and accepted invented ones.
            + (f"FIGURES TO CHECK: the OUTPUT states {', '.join(loose)}, which "
               "appear neither in the project request, the subtask, nor accepted "
               "results, and are not simple arithmetic on their figures. Reject if "
               "the output's answer depends on one of these and it does not present "
               "it as an assumption.\n\n" if loose else "")
            # PATCH 77. Run 25 rejected "about Rs. 700 per meal" as invented
            # on a request that gives no meal price. Limits still bind.
            + (f"LABELLED ESTIMATES (figures the OUTPUT marks as estimates): "
               f"{', '.join(estimated)}. An estimate is allowed only for a value the "
               "request does not give and does not let you compute; for such a value "
               "it is not fabrication. Reject an estimate of a value the request "
               "states or determines, an implausible one, or one whose totals break "
               "a limit the request states.\n\n" if estimated else "")
            # PATCH 62 (run 21). Arithmetic on the output's own numbers.
            + (f"DERIVATIONS TO CHECK: {'; '.join(chained)}. Each shows one way "
               "the figure follows from the output's own numbers, not necessarily "
               "the output's working. Find where the output gets the figure and "
               "reject if it builds on a number the output itself rejected (for "
               "example the cost of an option it did not choose).\n\n"
               if chained else "")
            # PATCH 67. Run 22 rejected a correct "15,000 / 50 = 300 per
            # person" as bad arithmetic; the figures had all checked out.
            # PATCH 71. The old note blessed a wrong 45 x 900 (students only)
            # as "checked". Code now shows WHICH request quantities each step
            # used and leaves whether they are the right ones to the critique.
            + (self._figures_checked_note(derivations_checked, request_labels)
               if derivations_checked or (figures_follow and not figures) else "")
            + (f"CODE THE AGENT RAN PRINTED: {tool_output} (computed; inputs not "
               "checked)\n\n" if tool_output else "")
            + "=== BEGIN OUTPUT (verbatim, produced by the agent being judged) ===\n"
            f"{output}\n"
            "=== END OUTPUT ===\n\n"
            "The OUTPUT block above is already finished. Do not continue, "
            "extend, or add to it -- everything you write from here on is "
            "your own critique, not a continuation of the agent's answer.\n"
            + (f"A code check found: {code_note}.\n" if code_note else "")
            + "Respond with exactly one verdict line: 'VERDICT: accept' or "
            "'VERDICT: reject', followed by a brief reasoning.\n"
            # Prefilled: the completion's first word is the ruling. Left to
            # open its own reply, the model wrote "Example:" and invented a
            # few-shot pair (VERDICT: reject ... VERDICT: accept ...) --
            # every tier-3 completion in a run had that shape, and the
            # last-match parse read the invented accept as the verdict.
            + self.VERDICT_PREFILL
        )

        completion = self.llm_call_fn(prompt)
        response = self._cut_invented_continuation(
            f"{self.VERDICT_PREFILL} {completion.lstrip()}"
        )

        # Escalating fallback, kept from the earlier fix (a strict per-line
        # match failed on every attempt for a model that does not reliably
        # follow single-line conventions, and the fail-closed default then
        # rejected everything): the "verdict:" pattern, then a bare
        # accept/reject, then the oddly-spaced spelling a strong repetition
        # penalty produced ("ac ce pt").
        #
        # FIRST match at every tier, not last. The last-match rule was added
        # for a model narrating through a self-correction, but the text after
        # the ruling turned out to be invented exemplars, not corrections:
        # "Example:\nVERDICT: reject ... VERDICT: accept ...", where the last
        # line is a made-up sample. With the prompt prefilled, the first
        # verdict is the one written straight after "VERDICT:", and
        # _cut_invented_continuation has already dropped any later
        # Example:/VERDICT: line.
        verdict = "reject"  # fail closed: still the default if nothing below matches
        matched = False

        # No trailing \b here, as before: "VERDICT: accepted" is an accept.
        match = re.search(r"verdict\s*:\s*(accept|reject)", response, re.IGNORECASE)
        if match is None:
            match = re.search(r"\b(accept|reject)\b", response, re.IGNORECASE)
        if match is not None:
            verdict = match.group(1).lower()
            matched = True

        if not matched:
            # Allow whitespace between every letter of each word.
            spaced_pattern = r"\b" + r"\s*".join("a c c e p t".split()) + r"\b|\b" + r"\s*".join("r e j e c t".split()) + r"\b"
            match = re.search(spaced_pattern, response, re.IGNORECASE)
            if match is not None:
                cleaned = re.sub(r"\s+", "", match.group(0)).lower()
                if cleaned in ("accept", "reject"):
                    verdict = cleaned
                    matched = True

        # FIX: confirmed via testing -- this generate() call has no stop
        # condition tied to the model actually finishing its critique, so a
        # chat-tuned model reliably drifts into conversational filler once
        # the real critique is done ("Let me know what you think!", "Want
        # help designing tests?"). That filler doesn't just look odd in
        # logs -- it was the majority of the text making it into
        # agent_node.fail_reason / ghost records once orchestrator.py
        # started forwarding this reasoning to respawned agents, crowding
        # out the one specific, actionable finding within the respawn's
        # limited prompt budget. Cut at the first sign-off/conversational
        # marker rather than fixing this via a hard length cap alone -- a
        # length cap can still land mid filler-sentence and leave a
        # dangling fragment; cutting at a marker removes the filler
        # cleanly regardless of exactly where it starts.
        filler_markers = [
            "Let me know", "Want help", "Thank you very much", "I'm always looking",
            "You're welcome to", "Stay curious", "Submitting revised", "Ready for final review",
        ]
        for marker in filler_markers:
            idx = response.find(marker)
            if idx != -1:
                response = response[:idx].strip()

        # FIX: see _dedupe_and_cap's docstring above -- this is the fix
        # site. Applied after filler-marker trimming (so a legitimate
        # sign-off doesn't skew the sentence-boundary split) and before
        # this becomes the "reasoning" every caller consumes.
        response = _dedupe_and_cap(response)

        return {"verdict": verdict, "reasoning": response}

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def decide(
        self,
        agent,
        output,
        output_type: str,
        output_embedding=None,
        target_embedding=None,
        needs_deep_check: bool = False,
        goal_embedding=None,
        project_goal=None,
        prior_warnings: int = 0,
        ungrounded_figures=None,
        figures_follow=False,
        derivations_checked=None,
        request_labels=None,
        tool_output=None,
        critique_check=None,
        labelled_estimates=None,
    ) -> dict:
        """
        The single entry point the orchestrator calls. Aggregates all three
        tiers into one verdict.

        PATCH 71. critique_check(reason) -> note or None. A tier-3 reject
        whose ground it finds checkably false is critiqued once more with
        the note shown; that second verdict is final ("arith_recheck").

        agent: the AgentNode being judged. Must expose `warning_count`
               (int) and `fail_reason` (str) -- decide() mutates both as a
               side effect of a warn verdict.

        Returns {"verdict": "pass" | "warn" | "execute" | "promote",
                 "reason": str, "tier": int}

        FIX (confirmed via code audit): added "tier" -- which of the three
        checks actually produced this verdict. Without it, a caller sees
        "execute" and has no way to tell tier 1 (near-free), tier 2
        (near-free), and tier 3 (deep_critique -- a full separate LLM call
        via self.llm_call_fn, run completely outside any Agent's own
        think()/decide() loop) apart. That distinction matters because tier
        3's real compute cost was going completely untracked by the
        colony's energy budget -- orchestrator.py's per-tick debit only
        measures growth in a live Agent's own thought_process, which
        deep_critique never touches. "tier" lets the orchestrator debit
        specifically for tier-3 calls (see orchestrator.py's
        handle_completion) without needing to guess from "verdict" alone
        whether the expensive path actually ran.
        """
        # ---------------- Tier 1 ----------------
        fast_result = self.fast_check(output, output_type)
        if not fast_result["pass"]:
            return {"verdict": "execute", "reason": fast_result["error"], "tier": 1}

        # ---------------- Tier 2 ----------------
        # FIX (confirmed via code audit): the strike-limit check previously
        # ran unconditionally here, BEFORE checking whether embeddings were
        # even supplied for this call. That meant an agent already at
        # MAX_WARNINGS got force-executed even on a round where tier 2 was
        # deliberately skipped (e.g. orchestrator.py's short-answer bypass
        # for a terse, factual result too short for goal-similarity
        # comparison to mean anything) -- punishing the agent for past
        # drift with zero new evidence that THIS answer drifted at all.
        # The strike system's whole point is "stop trusting the similarity
        # score once someone's drifted three times" -- that only makes
        # sense as a modifier ON a similarity check, not as a blanket
        # override that also fires when no similarity check is happening
        # this round. Moved inside the `else` branch so it only applies
        # when tier 2 is actually about to run.
        # N2a: a decomposer's output is a plan (a SPAWN batch / DIE reason),
        # not an attempt at the goal itself -- it is not supposed to
        # resemble the target text, and no SEMANTIC_WARN/EXECUTE threshold
        # is meaningful against it. Exempting by role, ahead of the
        # embedding-presence check below, so this holds even on a call that
        # happens to have both embeddings available.
        # PATCH 8 (measure-only, like PATCH 7). A second, project-level score
        # for the same output: cosine against the colony goal. NOTHING is
        # gated on it this round -- it only annotates a tier-2 verdict that
        # the task-level score already produced, so a WARN now tells the
        # agent (and the log) whether it drifted from its own task, from the
        # project, or from both. Thresholding waits for the spawn-time
        # distribution PATCH 7 is collecting; picking a cutoff now would
        # re-create exactly the N2b failure the own-task target was
        # introduced to fix, where deep subtasks were executed for not
        # resembling the whole project.
        goal_similarity = None
        if output_embedding is not None and goal_embedding is not None:
            goal_similarity = self.semantic_check(output_embedding, goal_embedding)
        goal_note = (f" [project-goal similarity={goal_similarity:.3f}, "
                     f"not gated]" if goal_similarity is not None else "")

        escalated = False
        if agent.role == "decomposer":
            similarity = None
        elif output_embedding is None or target_embedding is None:
            # No embeddings supplied (e.g. Phase 1 text-only agent with no
            # embedding pipeline wired up yet, or a short/terse result
            # deliberately exempted from tier 2 -- see orchestrator.py's
            # SHORT_ANSWER_WORD_THRESHOLD) -- skip tier 2 entirely rather
            # than fabricate a score. Falls through to tier 3 gate below.
            similarity = None
        else:
            if agent.warning_count >= MAX_WARNINGS:
                return {
                    "verdict": "execute",
                    "reason": f"strike limit reached ({agent.warning_count} warnings)",
                    "tier": 2,
                }

            similarity = self.semantic_check(output_embedding, target_embedding)

            if similarity <= SEMANTIC_EXECUTE_THRESHOLD:
                return {
                    "verdict": "execute",
                    "reason": (f"semantic drift too severe "
                               f"(similarity={similarity:.3f}){goal_note}"),
                    "tier": 2,
                }

            # PATCH 41. One WARN per task, then the deep critique decides.
            # Runs 17 and 18 warned on-topic prose answers at 0.35-0.58 again
            # and again (19 WARNs, 5 agents cycled out, one task abandoned at
            # 118 energy in run 18) -- a topic score cannot tell a revised
            # answer from a reworded one. prior_warnings carries WARNs earlier
            # attempts on the same task drew, so a respawn does not restart
            # the loop. Only a call that may run tier 3 escalates; the
            # SEMANTIC_EXECUTE_THRESHOLD floor above still applies first.
            if (similarity <= SEMANTIC_WARN_THRESHOLD and needs_deep_check
                    and agent.warning_count + max(0, prior_warnings)
                    >= WARN_ESCALATE_AFTER):
                escalated = True
            elif similarity <= SEMANTIC_WARN_THRESHOLD:
                agent.warning_count += 1
                # N2b: target_embedding is now this agent's OWN task (its
                # subtask's description embedding, or the colony goal only
                # for the root) -- never mention "the colony goal" here
                # unconditionally, since for a non-root agent that's not
                # what similarity was actually measured against, and
                # agent.task already names whatever it really was.
                correction = (
                    f"WARNING: Your last output had low semantic similarity "
                    f"to your assigned task (similarity={similarity:.3f}). You "
                    f"are drifting from it. Refocus on: "
                    f"{getattr(agent, 'task', '<task unavailable>')}"
                    f"{goal_note}"
                )
                agent.fail_reason = correction
                return {"verdict": "warn", "reason": correction, "tier": 2}

        # tier 2 passed cleanly (or was skipped) -> agent continues normally
        # unless this call needs the deep tier-3 critique

        # ---------------- Tier 3 ----------------
        if not needs_deep_check:
            return {
                "verdict": "pass",
                "reason": "tier 1/2 clear, deep check not required",
                "tier": 2,
            }

        # PATCH 62. Passed only when there are figures, so a critique
        # without the parameter (a stand-in) still runs.
        figures = list(ungrounded_figures or [])
        optional = {"ungrounded_figures": figures, "figures_follow": figures_follow,
                    "derivations_checked": derivations_checked,
                    "request_labels": request_labels, "tool_output": tool_output,
                    "labelled_estimates": list(labelled_estimates or [])}
        optional = {k: v for k, v in optional.items() if v}
        critique = self.deep_critique(output, getattr(agent, "task", None),
                                      project_goal=project_goal, **optional)
        # PATCH 71. One re-critique, never on an accept; code never rules.
        recheck = None
        if critique["verdict"] == "reject" and critique_check is not None:
            try:
                note = critique_check(critique["reasoning"])
            except Exception:
                note = None  # a failed check is no finding
            if note:
                first_reason = critique["reasoning"]
                critique = self.deep_critique(output, getattr(agent, "task", None),
                                              project_goal=project_goal,
                                              code_note=note, **optional)
                recheck = {"note": note, "verdict": critique["verdict"],
                           "first_reason_chars": len(str(first_reason or ""))}
        out = {"verdict": "execute" if critique["verdict"] == "reject" else "promote",
               "reason": critique["reasoning"], "tier": 3,
               "escalated": escalated, "similarity": similarity,
               "ungrounded_figures": figures,
               "labelled_estimates": list(labelled_estimates or [])}
        if recheck:
            out["arith_recheck"] = recheck
        return out