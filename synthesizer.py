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


class Synthesizer:
    # FIX: problem_phaser.py caps raw input at 3000 chars before it ever reaches
    # the model; format_output had no equivalent cap on the *output* side. A
    # colony with many completed subtasks could build a results_block that
    # exceeds the model's context window with no truncation at all. Mirrors
    # the same discipline problem_phaser.py already applies on the way in.
    MAX_RESULTS_BLOCK_CHARS = 6000

    def __init__(self, llm_call_fn=None):
        self.llm_call_fn = llm_call_fn

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect_results(self, colony_state, task_graph) -> list:
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

        Returns a list of {"task_id": str, "description": str, "result": str}
        dicts, ordered so that a task never appears before any task it
        depends on.
        """
        tasks = task_graph.tasks  # dict: task_id -> TaskNode

        completed = {
            tid: t for tid, t in tasks.items()
            if getattr(t, "status", None) == 2 and getattr(t, "result", None) is not None
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

        prompt = (
            "You are the final synthesizer for an AI agent colony that just "
            "solved a problem by decomposing it into subtasks. Combine the "
            "following subtask results into a single, coherent answer to the "
            "original problem. Do not mention the colony, agents, or subtasks "
            "in your answer -- write as if you solved the problem directly.\n\n"
            f"ORIGINAL PROBLEM: {problem_spec}\n\n"
            f"SUBTASK RESULTS:\n{results_block}\n\n"
            "FINAL ANSWER:"
        )

        return self.llm_call_fn(prompt)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, colony_state, task_graph, problem_spec: str) -> str:
        """
        The orchestrator's one call site: gather everything, decode once.
        """
        results = self.collect_results(colony_state, task_graph)
        return self.format_output(results, problem_spec)