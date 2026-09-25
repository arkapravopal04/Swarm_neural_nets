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
    cut_adjacent_repeat as _cut_adjacent_repeat,
    dedupe_global_and_cap as _dedupe_and_cap,
    degeneracy_cut as _degeneracy_cut,
    drop_incomplete_tail as _drop_incomplete_tail,
    looks_degenerate as _looks_degenerate,
    cut_model_not_completed as _cut_model_not_completed,
    not_completed_sentence as _not_completed_sentence,
    strip_task_ids as _strip_task_ids,
    trim_degenerate_tails as _trim_degenerate_tails,
    ungrounded_telemetry as _ungrounded_telemetry,
)
# The prompt's own SPAWN examples, read from the one place they are defined
# rather than restated here -- same reason Orchestrator._matching_exemplar
# imports them. A copy would keep matching text the prompt no longer uses
# the moment someone rewords an example, which is worse than no check.
from agent_node import PROMPT_EXEMPLARS


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
        # PATCH 32 (cont.). Reset first so a second call never reports (or,
        # in collect_not_completed, covers with) roll-ups from the first.
        self.last_rollups_omitted = []
        tasks = task_graph.tasks  # dict: task_id -> TaskNode

        completed = {
            tid: t for tid, t in tasks.items()
            if tid != root_task_id
            and getattr(t, "status", None) == 2 and getattr(t, "result", None) is not None
        }

        # PATCH 32 (cont.). Drop a roll-up whose direct children all
        # completed: its children are already in the list, and run 10's
        # synthesizer got a roll-up AND its two children -- three of five
        # items the same fee chain -- copied them, and ignored a concrete
        # rule that appeared once. A child counts as covered when it is in
        # `completed`, or when it failed (status 3) and a completed sibling
        # has the same normalised description (the parent spawned the same
        # subtask again; a respawn reuses the task_id, so it never does
        # this). Decided against the ORIGINAL `completed` set, so a
        # grandparent whose child is itself an omitted roll-up goes too.
        # Omitted roll-ups are completed work, not abandoned work.
        omitted = self._rollups_to_omit(completed, task_graph)
        self.last_rollups_omitted = [tid for tid in completed if tid in omitted]
        for _ in self.last_rollups_omitted:
            self._record_trim("rollups_omitted")

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
            if tid not in omitted
        ]

    @staticmethod
    def _rollups_to_omit(completed, task_graph) -> set:
        """PATCH 32 (cont.). IDs in `completed` whose direct children (graph
        parent_task_id, not AgentNode.children) are all covered -- see
        collect_results. A task with no children is never omitted."""
        def _norm(text):
            return " ".join(str(text or "").lower().split())

        def _kids(task_id):
            direct_children = getattr(task_graph, "direct_children", None)
            if callable(direct_children):
                return list(direct_children(task_id) or [])
            return [t for t in task_graph.tasks.values()
                    if getattr(t, "parent_task_id", None) == task_id]

        omitted = set()
        for tid in completed:
            kids = _kids(tid)
            if not kids:
                continue
            done_descs = {_norm(getattr(k, "description", None))
                          for k in kids if getattr(k, "task_id", None) in completed}
            done_descs.discard("")
            if all(getattr(k, "task_id", None) in completed
                   or (getattr(k, "status", None) == 3
                       and _norm(getattr(k, "description", None)) in done_descs)
                   for k in kids):
                omitted.add(tid)
        return omitted

    def collect_not_completed(self, task_graph, abandoned_ids, results) -> list:
        """PATCH 19. Descriptions of the abandoned subtasks, for the NOT
        COMPLETED block of the synthesis prompt, in a stable order.

        Before this the synthesizer saw ONLY promoted results, so "none of
        the subtasks left any portion unresolved" was the one conclusion its
        input allowed -- run 7 said exactly that with six subtasks abandoned.
        The [PARTIAL RESULT] banner told the user, but it is added by the
        orchestrator after this call and the model never saw it.

        A description that ALSO appears among the completed results is left
        out. Respawns and success-cache hits routinely leave an abandoned
        task and a completed one with the same text (run 6: task_f0bb2e25
        abandoned, task_03a760ea completed, both "Assign individualized
        reading schedules accounting for current progress status"), and
        listing it under both headings would ask the model to report as
        missing a part the answer just covered.
        """
        def _norm(text):
            return " ".join(str(text or "").lower().split())

        covered = {_norm(r.get("description")) for r in results}
        # PATCH 32 (cont.). A roll-up omitted by collect_results is completed
        # work that is no longer in `results`; its description still covers
        # an abandoned same-description attempt, or that attempt would be
        # listed as NOT COMPLETED.
        for task_id in getattr(self, "last_rollups_omitted", None) or ():
            task = task_graph.tasks.get(task_id)
            if task is not None:
                covered.add(_norm(getattr(task, "description", None)))
        seen = set()
        out = []
        for task_id in sorted(abandoned_ids or ()):
            task = task_graph.tasks.get(task_id)
            description = getattr(task, "description", None) if task is not None else None
            key = _norm(description)
            if not key or key in covered or key in seen:
                continue
            seen.add(key)
            out.append(" ".join(str(description).split()))
        return out

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

    def format_output(self, results: list, problem_spec: str,
                      not_completed: list = None) -> str:
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

        # PATCH 24. No task IDs in the material: run 8's answer opened with
        # "[task_e03ea144] 200 rupees/m^3", the labels copied straight out
        # of this block. The description says what the result is for.
        results_block = "\n\n".join(
            f"- {r['description']}\n  Result: {r['result']}"
            for r in results
        )
        if len(results_block) > self.MAX_RESULTS_BLOCK_CHARS:
            results_block = (
                results_block[: self.MAX_RESULTS_BLOCK_CHARS]
                + "\n... [TRUNCATED -- additional subtask results omitted for length]"
            )

        # PATCH 19. The prompt is procedural on purpose: it says what to
        # write, in what order, from which block -- and contains NO word the
        # model can hand back as a verdict on its own answer.
        #
        # The version it replaces asked the model to "Ground every claim in
        # the SUBTASK RESULTS ... do not introduce facts, figures, or
        # conclusions that are not traceable to one of them. If the results
        # leave part of the problem uncovered, say plainly that it wasn't
        # addressed". Run 7's answer ended "All claims here are grounded in
        # these results. There are no invented facts or conclusions beyond
        # what is explicitly stated. ... None of the subtasks left any
        # portion of the problem unresolved." -- the instruction recited
        # back as a claim of compliance, right after a sentence that turned
        # a subtask's "80% of the original proposed titles" into "over 80% of
        # readers' preferences". An instruction phrased as a quality
        # standard ("grounded", "traceable", "coherent", "no invented
        # facts") comes back as an assertion that the standard was met;
        # instructions phrased as steps ("copy", "name each item") have no
        # such form to come back in.
        #
        # PATCH 24. The not-completed sentence is no longer the model's to
        # write. PATCH 19 asked for it as "part 2", and run 8's answer ended
        # "Nothing else. Leave this part out if nothing is missing." -- the
        # instruction itself, shipped to the user. It is built in code from
        # the abandoned list and appended after every cut below, so the
        # model sees only the completed work and writes only the answer.
        #
        # Removed with the grounding sentence (PATCH 19): "just solved a
        # problem" (asserts success before a word is read) and "a single,
        # coherent answer" (another quality word). Kept: the ban on naming
        # the colony, agents or subtasks.
        prompt = (
            "Write the answer to the problem below for the person who asked "
            "it. Your material is the COMPLETED WORK. Do not mention agents, "
            "subtasks, or how the work was divided up.\n\n"
            f"PROBLEM: {problem_spec}\n\n"
            f"COMPLETED WORK:\n{results_block}\n\n"
            "Write the answer, built from the COMPLETED WORK. Copy names, "
            "identifiers and numbers from it as they are written there.\n\n"
            "ANSWER:"
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
        #
        # FIX (a deploy-log tail reached the user, twice): a final answer
        # ended "...Ready to deploy. Done. Go. Deployment timestamp:
        # 2023-11-03T14:23:10Z. Metrics report: accuracy=1.000000,
        # consistency=1.000000... Final state: COMPLETED... Report end." --
        # the loop seen in single REPORTs earlier in the same run, plus a
        # made-up timestamp and metrics. None of the passes here caught it:
        # nothing in it repeats three times, "Go." breaks the closer run, and
        # the global dedupe collapsed the repeated "Ready to deploy. Done.
        # Go." into one copy that read as clean. Upstream scrubbing now cuts
        # it from each REPORT, but this decode can write it fresh, so it is
        # guarded here as well:
        #   * cut_adjacent_repeat cuts at the second copy of a block repeated
        #     back to back, taking everything after it,
        #   * trim_degenerate_tails cuts the status-report tail (timestamps
        #     and metrics checked against the problem and the results the
        #     answer had to stay inside), then the closer and restating tails.
        # Tails are trimmed after degeneracy_cut, since its cut can leave one
        # exposed.
        raw = self.llm_call_fn(prompt)
        grounding = "\n".join([str(problem_spec or "")]
                              + [str(r.get("result", "")) for r in results])
        # PATCH 24. Cut FIRST, so the empty-answer handling below still
        # applies if the model wrote nothing but its own not-completed list.
        raw, wrote_not_completed = _cut_model_not_completed(raw)
        if wrote_not_completed:
            self._record_trim("model_not_completed_cut")
            print("  [synthesis-trim] cut a not-completed section the model "
                  "wrote itself -- the code-built one is appended instead.")
        cleaned = _drop_incomplete_tail(raw)

        uncut = cleaned
        cleaned, reason = _cut_adjacent_repeat(cleaned)
        if reason is not None:
            self._record_trim("repeat_cut")
            print(f"  [synthesis-trim] final answer cut at {reason} "
                  f"({len(uncut)} -> {len(cleaned)} chars).")

        kept, reason = _degeneracy_cut(cleaned, exemplars=PROMPT_EXEMPLARS)
        if reason is not None:
            self._record_trim("cut")
            print(f"  [synthesis-trim] final answer cut at {reason} "
                  f"({len(cleaned)} -> {len(kept)} chars).")

        untrimmed = kept
        kept, tail_reasons = _trim_degenerate_tails(kept, grounding)
        if "status-report tail" in tail_reasons:
            self._record_trim("status_tail")
            print(f"  [synthesis-trim] final answer cut at a status-report tail "
                  f"({len(untrimmed)} -> {len(kept)} chars).")

        # The walk-back again, after the cuts. _drop_incomplete_tail above ran
        # on the raw decode only; a cut at a line boundary (the repeat and
        # status-tail cuts walk lines too) or mid-line (a scaffolding echo)
        # can leave the answer ending on an unfinished sentence.
        if kept != uncut and kept.strip():
            walked = _drop_incomplete_tail(kept)
            if walked != kept:
                self._record_trim("incomplete_tail_after_cut")
                print(f"  [synthesis-trim] dropped an unfinished sentence the cut "
                      f"exposed ({len(kept)} -> {len(walked)} chars).")
                kept = walked
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

        answer, removed_ids = _strip_task_ids(answer)
        if removed_ids:
            self._record_trim("task_ids_stripped")
            print(f"  [synthesis-trim] removed {removed_ids} internal task "
                  f"ID(s) from the final answer.")

        # Same stance for a made-up log line that is not at the tail, where
        # cutting it would take real content with it.
        made_up = _ungrounded_telemetry(answer, grounding)
        if made_up:
            self._record_trim("shipped_telemetry")
            print(f"  [synthesis-trim] WARNING: the final answer states "
                  f"{len(made_up)} timestamp/metric line(s) no subtask result "
                  f"contains -- shipping it: {made_up[0][:80]!r}")

        # PATCH 24. Appended last, after every cut, so no trim can reach it.
        sentence = _not_completed_sentence(not_completed)
        if sentence:
            self._record_trim("not_completed_appended")
            answer = f"{answer}\n\n{sentence}"
        return answer

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, colony_state, task_graph, problem_spec: str, root_task_id=None,
            abandoned_ids=None) -> str:
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
        not_completed = self.collect_not_completed(task_graph, abandoned_ids, results)
        return self.format_output(results, problem_spec, not_completed=not_completed)