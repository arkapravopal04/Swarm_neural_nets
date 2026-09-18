"""
synthesizer.py

Takes final results from all completed tasks and produces the single
human-readable answer -- the one place in the system where the colony's
work becomes English again.

Phase 1 honesty: results are plain strings (Agent.report() calls
str(payload)), there is exactly one root task, and agents run sequentially,
not in parallel. So there is no real conflict to resolve yet -- multiple
agents don't compete to answer the same subtask. resolve_conflicts() exists
as a stub that Phase 2 fills in with real embedding-based voting, once
parallel agents on the same subtask and confidence-scored embeddings both
exist. Building fake conflict resolution now would be dishonest about what
the system can actually do -- an unused method that returns results[0] says
exactly what it is, nothing more.

llm_call_fn is injected at construction, same pattern as judge.py -- the
synthesizer doesn't own a model instance, the orchestrator wires in whatever
already wraps the shared model.
"""

from text_utils import (
    dedupe_global_and_cap as _dedupe_and_cap,
    degeneracy_cut as _degeneracy_cut,
    drop_incomplete_tail as _drop_incomplete_tail,
    looks_degenerate as _looks_degenerate,
    trim_closer_tail as _trim_closer_tail,
)
# The prompt's own SPAWN examples, read from the one place they are defined
# rather than restated here -- same reason Orchestrator._matching_exemplar
# imports them. A copy would keep matching text the prompt no longer uses
# the moment someone rewords an example, which is worse than no check.
from agent_node import EXEMPLAR_SUBTASK_DESCRIPTIONS


class Synthesizer:
    # FIX: problem_phaser.py caps raw input at 3000 chars before it ever reaches
    # the model; format_output had no equivalent cap on the *output* side. A
    # colony with many completed subtasks could build a results_block that
    # exceeds the model's context window with no truncation at all. Mirrors
    # the same discipline problem_phaser.py already applies on the way in.
    MAX_RESULTS_BLOCK_CHARS = 6000

    # What the user is told when the final decode collapsed from its very
    # first sentence, so there is nothing left after the cut in
    # format_output. Said plainly rather than returned as an empty string or
    # as the degenerate text itself: the run failed at the last step, and a
    # caller (or a reader) that cannot tell that apart from a real answer is
    # exactly the hole this guard exists to close.
    DEGENERATE_ANSWER_MESSAGE = (
        "The final answer collapsed into repeated or scaffolding text from "
        "its first sentence -- no usable answer was produced."
    )

    def __init__(self, llm_call_fn=None):
        self.llm_call_fn = llm_call_fn
        # What format_output had to do to its own output, by kind: "cut",
        # "empty", "shipped_degenerate". A print line is enough to debug one
        # run but invisible in the end-of-run ledger, which is where anyone
        # comparing runs actually looks -- and a guard nobody can see firing
        # is how the cycle cap ended up suspected of not existing.
        #
        # Kept here rather than pushed to ColonyState.record_verdict so this
        # class stays free of the colony: it is handed results and a
        # callable, nothing else. The orchestrator reads this dict when it
        # prints the ledger.
        self.trim_counts = {}

    def _record_trim(self, kind: str):
        self.trim_counts[kind] = self.trim_counts.get(kind, 0) + 1

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_results(self, colony_state, task_graph, root_task_id=None) -> list:
        """
        Gather results from all completed task nodes, in dependency order.

        Does its own lightweight topological sort over task_graph.tasks
        using each TaskNode's .dependencies list, rather than assuming a
        specific topological-sort method exists on TaskGraph -- keeps this
        self-contained regardless of TaskGraph's internal API.

        Only tasks with status == 2 (complete, per task_graph.py's
        0=pending/1=running/2=complete/3=failed convention) and a non-None
        result are included. Incomplete or failed tasks contribute nothing --
        there's no partial result to synthesize from a task that never
        finished.

        root_task_id, if given, is excluded from the results: it's the final
        answer being assembled, not a subtask contributing to it. Without
        this exclusion, a root task with zero real subtask results still
        shows up as one "result" (its own REPORT), making an empty
        decomposition indistinguishable from a real one.

        Returns a list of {"task_id": str, "description": str, "result": str}
        dicts, ordered so that a task never appears before any task it
        depends on.
        """
        tasks = task_graph.tasks  # dict: task_id -> TaskNode

        completed = {
            tid: t for tid, t in tasks.items()
            if tid != root_task_id
            and getattr(t, "status", None) == 2 and getattr(t, "result", None) is not None
        }

        ordered_ids = []
        visited = set()

        def visit(task_id):
            if task_id in visited or task_id not in completed:
                return
            visited.add(task_id)
            task = completed[task_id]
            for dep_id in getattr(task, "dependencies", []) or []:
                visit(dep_id)
            ordered_ids.append(task_id)

        for tid in completed:
            visit(tid)

        return [
            {
                "task_id": tid,
                "description": getattr(completed[tid], "description", ""),
                "result": completed[tid].result,
            }
            for tid in ordered_ids
        ]

    # ------------------------------------------------------------------
    # Conflict resolution -- Phase 1 stub
    # ------------------------------------------------------------------

    def resolve_conflicts(self, results: list) -> str:
        """
        Phase 1: no parallel agents attempt the same subtask, so there is
        nothing to resolve. Returns the first available result as-is.

        Phase 2: this becomes cosine-similarity-weighted voting across
        multiple agents' embeddings for the same subtask, using each
        agent's confidence score. This is the ONLY method that changes
        when Phase 2 lands -- collect_results and format_output stay as is.
        """
        return results[0]["result"] if results else ""

    # ------------------------------------------------------------------
    # Final English decode
    # ------------------------------------------------------------------

    def format_output(self, results: list, problem_spec: str) -> str:
        """
        The single English decode for the entire colony run. One LLM call,
        combining all collected results into a coherent final answer for
        the user.

        Uses self.llm_call_fn, bound at construction (same pattern as
        judge.deep_critique).
        """
        if self.llm_call_fn is None:
            raise ValueError("format_output requires llm_call_fn to be set at Synthesizer construction")

        if not results:
            return "The colony was unable to produce a result for this problem."

        results_block = "\n\n".join(
            f"[{r['task_id']}] {r['description']}\nResult: {r['result']}"
            for r in results
        )
        if len(results_block) > self.MAX_RESULTS_BLOCK_CHARS:
            results_block = (
                results_block[: self.MAX_RESULTS_BLOCK_CHARS]
                + "\n... [TRUNCATED -- additional subtask results omitted for length]"
            )

        # FIX (synthesizer grounding): collect_results already restricts
        # SUBTASK RESULTS to promoted (status==2) tasks, so the results
        # block itself is honest -- but nothing told the model to stay
        # inside it. "Combine these into a coherent answer" is an open
        # invitation to bridge gaps with invented facts once it starts
        # writing prose, and a gap in subtask coverage would come out
        # looking exactly like a confidently synthesized claim. Every
        # claim in the final answer now has to trace back to one of the
        # results actually promoted by the judge, or say plainly that the
        # colony didn't cover it.
        prompt = (
            "You are the final synthesizer for an AI agent colony that just "
            "solved a problem by decomposing it into subtasks. Combine the "
            "following subtask results into a single, coherent answer to the "
            "original problem. Do not mention the colony, agents, or subtasks "
            "in your answer -- write as if you solved the problem directly.\n\n"
            "Ground every claim in the SUBTASK RESULTS below -- do not "
            "introduce facts, figures, or conclusions that are not traceable "
            "to one of them. If the results leave part of the problem "
            "uncovered, say plainly that it wasn't addressed rather than "
            "inventing an answer for it.\n\n"
            f"ORIGINAL PROBLEM: {problem_spec}\n\n"
            f"SUBTASK RESULTS:\n{results_block}\n\n"
            "FINAL ANSWER:"
        )

        # Same greedy-decoding sentence-looping failure mode every other
        # llm_call_fn caller already guards against (see judge.deep_critique,
        # agent_node's REPORT/DIE paths) -- undeduped here, it's the last
        # LLM call in the whole run, so a loop survives straight into what
        # the user reads as the finished answer instead of getting caught
        # partway through the pipeline. Capped at the same budget as the
        # input side (MAX_RESULTS_BLOCK_CHARS) for symmetry.
        #
        # FIX (repeat survived into the final answer): a real run ended with
        # "THESE ACTIONS DIRECTLY ADDRESS THE TWO PRIMARY" repeated verbatim
        # at the very end, past a single dedupe pass. Three separate holes,
        # all of which had to be open at once for that to happen:
        #
        #   1. the repeat was ADJACENT but not IDENTICAL -- the second copy
        #      was cut off mid-sentence by the generation budget, so
        #      "...THE TWO PRIMARY GOALS." and "...THE TWO PRIMARY" compared
        #      unequal and dedupe_and_cap kept both. A trailing fragment has
        #      no terminator, so dropping it first is what actually removes
        #      this shape.
        #   2. dedupe_and_cap only collapses ADJACENT duplicates. A repeat
        #      separated by even one intervening sentence survived it.
        #      dedupe_global_and_cap is the same function without that
        #      restriction, and there is no reason the LAST decode in the
        #      run should be the one using the weaker of the two.
        #   3. nothing here caught closer-cycling at all.
        #
        # Order is load-bearing: clean on sentence boundaries first, cap
        # last, because the cap appends an ellipsis that later passes would
        # misread as a sentence terminator.
        #
        # FIX (a degenerate tail reached the user): the passes above CLEAN,
        # and cleaning is the wrong verb for this failure. One real run
        # signed its user-facing answer off with "ACTION REQUESTED: RUN OR
        # ABORT?" and a repeated exemplar sentence -- neither is a duplicate
        # to collapse or a closer to trim, so every pass above left them
        # exactly where they were. handle_completion runs a degeneracy check
        # over each individual REPORT before the judge reads it; nothing ran
        # one over the answer those REPORTs are stitched into, which is the
        # only text in the run the user actually reads.
        #
        # Cut rather than filtered, because past the point where a decode
        # starts reciting its own output or the harness's scaffolding, the
        # rest is the collapse. Keeping the good prefix and dropping the
        # rest is honest; deduping the tail and pasting the survivors back
        # together builds something that reads finished out of the failure.
        raw = self.llm_call_fn(prompt)
        cleaned = _trim_closer_tail(_drop_incomplete_tail(raw))

        kept, reason = _degeneracy_cut(cleaned, exemplars=EXEMPLAR_SUBTASK_DESCRIPTIONS)
        if reason is not None:
            self._record_trim("cut")
            print(f"  [synthesis-trim] final answer cut at {reason} "
                  f"({len(cleaned)} -> {len(kept)} chars).")
        if not kept.strip():
            self._record_trim("empty")
            print("  [synthesis-trim] nothing survived the cut -- the final "
                  "decode was degenerate from its first sentence.")
            return self.DEGENERATE_ANSWER_MESSAGE

        # keep_line_breaks, unlike every other caller of this helper: those
        # are cleaning a string that goes back into a prompt as one line,
        # this is the document the user reads. Without it a final answer
        # written as a numbered plan or a bulleted list arrived as one
        # flattened paragraph -- the cleaning passes above all preserve the
        # layout, and then the last one threw it away.
        answer = _dedupe_and_cap(kept, max_chars=self.MAX_RESULTS_BLOCK_CHARS,
                                 keep_line_breaks=True)

        # Reported, not cut. What can still reach here is closer-cycling
        # buried mid-answer, which degeneracy_cut leaves alone on purpose
        # (trim_closer_tail already handles the tail, where cutting is the
        # only place it is safe). A line in the run log beats silently
        # shipping it as if nothing happened, and beats destroying the real
        # content that follows it.
        if _looks_degenerate(answer):
            self._record_trim("shipped_degenerate")
            print("  [synthesis-trim] WARNING: the final answer still reads "
                  "as degenerate after trimming -- shipping it, but the last "
                  "decode of this run did not go cleanly.")
        return answer

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, colony_state, task_graph, problem_spec: str, root_task_id=None) -> str:
        """
        The orchestrator's one call site: gather everything, decode once.

        FIX (root-acceptance bug): if no non-root subtask produced a result,
        there is nothing to synthesize -- calling the LLM anyway on an empty
        results_block let it fabricate a plausible-looking answer from the
        problem_spec alone, indistinguishable from a real synthesis. Return
        an explicit failure string instead of ever making that call.
        """
        results = self.collect_results(colony_state, task_graph, root_task_id=root_task_id)
        if not results:
            return "No subtask results were produced -- there is nothing to synthesize."
        return self.format_output(results, problem_spec)