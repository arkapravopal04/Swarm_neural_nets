"""
judge.py

Tiered verification of agent outputs for Project Hive. Decides life and death.

Three tiers, run in escalating order, each gated behind the previous passing:

    Tier 1 -- fast_check     : sub-10ms syntax/execution check, runs on every output
    Tier 2 -- semantic_check : cosine similarity vs the colony's master target
                               embedding, runs only if tier 1 passed
    Tier 3 -- deep_critique  : full LLM call, runs only when an agent is
                               attempting to promote its result to its parent

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


def _dedupe_and_cap(text, max_chars: int = 500):
    """
    FIX (confirmed via a real run): deep_critique's own output degenerated
    into runaway repetition after its real content twice in one run --
    "The response is entirely fabricated. The simulation claim is entirely
    absent..." repeated 6x verbatim, and a chain of ~25 unrelated "The X
    must be Y." filler sentences with no connection to the actual
    critique. This is the same greedy-decoding degeneration pattern seen
    elsewhere in this project (agent_node.py's _dedupe_repeated_sentences,
    added for DIE payloads and ghost context). Fixing it HERE, at the
    source, rather than only downstream (ghost_extractor.py independently
    patches this same value before persisting it -- see that file's
    _dedupe_and_cap) means every consumer of deep_critique's output
    benefits: the verdict text printed to logs, the text that becomes the
    next respawn's fail_reason, AND the persisted ghost record, instead of
    relying on each caller to separately defend itself against the same
    degenerate text. ghost_extractor's copy remains as defense-in-depth,
    not the only protection.

    Small and local rather than importing agent_node's version -- judge.py
    already keeps its own boundaries deliberately (no embedding model, no
    tool access, llm_call_fn injected rather than owned -- see this
    module's docstring), and this mirrors the same "self-contained, don't
    reach across files" choice ghost_extractor.py already made for its
    own copy of this exact logic.
    """
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    deduped = []
    for s in sentences:
        if not deduped or s.strip() != deduped[-1].strip():
            deduped.append(s)
    result = " ".join(deduped)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0] + "..."
    return result


# --- Tier 2 thresholds -----------------------------------------------------
# The entire tuning surface for semantic drift detection. Adjust these two
# numbers after watching real colony runs -- if good agents get executed or
# drifting agents survive, these are the knobs to turn.
SEMANTIC_WARN_THRESHOLD = 0.6      # below this -> warn
SEMANTIC_EXECUTE_THRESHOLD = 0.3   # at or below this -> execute, too far gone

# Strike system: after this many warnings, skip the similarity check
# entirely on the next tier 2 pass and execute regardless of current score.
MAX_WARNINGS = 3


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

    def __init__(self, llm_call_fn=None):
        self.llm_call_fn = llm_call_fn

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
            error = output.get("data") or output.get("error") or f"status={status}"
            return {"pass": False, "error": str(error)}

        if output_type == "text":
            if output is None or (isinstance(output, str) and output.strip() == ""):
                return {"pass": False, "error": "empty or missing text output"}
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

    def deep_critique(self, output, subtask_spec) -> dict:
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

        prompt = (
            "You are the judge for an AI agent colony. Critique the following "
            "output against its assigned subtask. Respond with exactly one "
            "verdict line: 'VERDICT: accept' or 'VERDICT: reject', followed by "
            "a brief reasoning.\n\n"
            f"SUBTASK: {subtask_spec}\n\n"
            f"OUTPUT: {output}\n"
        )

        response = self.llm_call_fn(prompt)

        # FIX: confirmed in testing -- this strict "line must start with
        # 'verdict:'" parse was failing on every single attempt for a small,
        # non-fine-tuned model that doesn't reliably follow single-line
        # output conventions (the exact same class of brittleness already
        # found and fixed in agent_node.py's decide() ACTION/PAYLOAD parsing).
        # Combined with the fail-closed default, this meant deep_critique
        # ALWAYS rejected, deterministically, every respawn -- burning the
        # entire energy budget on identical repeated failures with no chance
        # of success, since decoding is greedy/deterministic and nothing
        # about the underlying miss changes between attempts.
        #
        # Escalating fallback: try the original strict per-line match first
        # (preserves exact behavior when the model DOES follow the format),
        # then a substring search for "verdict:" anywhere in the response
        # (not just at a line start), then a last-resort keyword search for
        # a standalone accept/reject anywhere in the text. This can only ever
        # improve on the previous behavior -- the previous behavior was
        # "reject always"; a false-positive "accept" from the keyword search
        # is no worse than that guaranteed failure, and a correctly detected
        # "accept" is strictly better.
        # FIX (confirmed via testing, second pass): the strict per-line loop
        # used to `break` on the FIRST "verdict:" line it found. Confirmed
        # in a real run: a small model's self-critique commonly narrates
        # through a correction -- "VERDICt: reject - ...", then goes on to
        # reconsider and write "VERDICt: accept - fixed the issue..." twice
        # more. Breaking on the first line locked in the REJECTED first
        # draft and threw away the model's own two corrections, causing an
        # EXECUTE/respawn on work the model itself had already decided was
        # fine. Now scans the full response and keeps the LAST verdict line
        # found, honoring whatever the model concluded by the time it
        # finished, not what it said first. Same reasoning applies to the
        # regex fallbacks below -- findall/last-match instead of search's
        # first-match.
        verdict = "reject"  # fail closed: still the default if nothing below matches
        matched = False

        for line in response.splitlines():
            if line.strip().lower().startswith("verdict:"):
                value = line.split(":", 1)[1].strip().lower()
                if value in ("accept", "reject"):
                    verdict = value
                    matched = True
                # no break -- keep scanning so a later correction wins

        if not matched:
            matches = re.findall(r"verdict\s*:\s*(accept|reject)", response, re.IGNORECASE)
            if matches:
                verdict = matches[-1].lower()
                matched = True

        if not matched:
            matches = re.findall(r"\b(accept|reject)\b", response, re.IGNORECASE)
            if matches:
                verdict = matches[-1].lower()
                matched = True

        if not matched:
            # FIX: confirmed via a real run -- an aggressive repetition
            # penalty (since reduced, see main.py's build_llm_call_fn) can
            # push the model to spell "accept"/"reject" as oddly-spaced
            # subword fragments ("ac ce pt") to avoid repeating an exact
            # token sequence. None of the tiers above match text with
            # internal spacing like that. This is a last-resort net: allow
            # 0-2 whitespace characters between every letter of each word.
            spaced_pattern = r"\b" + r"\s*".join("a c c e p t".split()) + r"\b|\b" + r"\s*".join("r e j e c t".split()) + r"\b"
            matches = re.findall(spaced_pattern, response, re.IGNORECASE)
            if matches:
                cleaned = re.sub(r"\s+", "", matches[-1]).lower()
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
        is_promotion_attempt: bool = False,
    ) -> dict:
        """
        The single entry point the orchestrator calls. Aggregates all three
        tiers into one verdict.

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
        if output_embedding is None or target_embedding is None:
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
                    "reason": f"semantic drift too severe (similarity={similarity:.3f})",
                    "tier": 2,
                }

            if similarity <= SEMANTIC_WARN_THRESHOLD:
                agent.warning_count += 1
                correction = (
                    f"WARNING: Your last output had low semantic similarity "
                    f"to the colony goal (similarity={similarity:.3f}). You "
                    f"are drifting from your assigned subtask. Refocus on: "
                    f"{getattr(agent, 'task', '<task unavailable>')}"
                )
                agent.fail_reason = correction
                return {"verdict": "warn", "reason": correction, "tier": 2}

        # tier 2 passed cleanly (or was skipped) -> agent continues normally
        # unless this call is a promotion attempt

        # ---------------- Tier 3 ----------------
        if not is_promotion_attempt:
            return {
                "verdict": "pass",
                "reason": "tier 1/2 clear, not a promotion attempt",
                "tier": 2,
            }

        critique = self.deep_critique(output, getattr(agent, "task", None))
        if critique["verdict"] == "reject":
            return {"verdict": "execute", "reason": critique["reasoning"], "tier": 3}

        return {"verdict": "promote", "reason": critique["reasoning"], "tier": 3}