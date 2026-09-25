"""
PATCH 32 (cont.) -- what the final synthesis is given, and one sign-off it
used to ship.

  2a  The synthesizer's PROBLEM is the user's request (spec["raw_text"]), not
      the phaser's goal, on BOTH synthesis paths (root completion and
      terminate()'s partial synthesis). Empty raw_text falls back to the
      goal and is counted.
  2b  collect_results drops a roll-up whose direct children all completed,
      so the model does not see the same chain three times.
  2c  _SIGNOFF_RE cuts a trailing "Final check passed." and nothing mid-answer.

Synthetic fixtures only; the run 10 log is unavailable.
"""
from colony_state import AgentNode, ColonyState
from event_queue import Event, Messenger
from orchestrator import Orchestrator
from synthesizer import Synthesizer
from task_graph import TaskGraph, TaskNode
from text_utils import _SIGNOFF_RE, trim_degenerate_tails


# --------------------------------------------------------------- helpers

def _graph(*nodes):
    graph = TaskGraph()
    for node in nodes:
        graph.add_task(node)
    return graph


def _done(tid, desc, parent=None, deps=()):
    node = TaskNode(task_id=tid, description=desc, status=2,
                    parent_task_id=parent, dependencies=list(deps))
    node.result = f"result of {tid}"
    return node


def _ids(results):
    return [r["task_id"] for r in results]


# --------------------------------------------------------------- 2b roll-ups

def test_rollup_with_both_children_completed_is_omitted():
    graph = _graph(
        TaskNode(task_id="root", description="goal", status=1),
        _done("fee", "Work out the fee", parent="root"),
        _done("fee-a", "Plot cost", parent="fee"),
        _done("fee-b", "Multiply by 12", parent="fee"),
        _done("rule", "Abandonment rule", parent="root"),
    )
    synth = Synthesizer()
    results = synth.collect_results(None, graph, root_task_id="root")
    assert _ids(results) == ["fee-a", "fee-b", "rule"]
    assert synth.last_rollups_omitted == ["fee"]
    assert synth.trim_counts["rollups_omitted"] == 1


def test_rollup_with_a_pending_or_failed_child_is_kept():
    for status in (0, 1, 3):
        graph = _graph(
            _done("fee", "Work out the fee"),
            _done("fee-a", "Plot cost", parent="fee"),
            TaskNode(task_id="fee-b", description="Multiply by 12", status=status,
                     parent_task_id="fee"),
        )
        synth = Synthesizer()
        results = synth.collect_results(None, graph)
        assert _ids(results) == ["fee", "fee-a"], status
        assert synth.last_rollups_omitted == []
        assert "rollups_omitted" not in synth.trim_counts


def test_failed_child_with_a_completed_same_description_sibling_is_covered():
    graph = _graph(
        _done("fee", "Work out the fee"),
        TaskNode(task_id="fee-a1", description="Plot  COST", status=3, parent_task_id="fee"),
        _done("fee-a2", "plot cost", parent="fee"),
        _done("fee-b", "Multiply by 12", parent="fee"),
    )
    synth = Synthesizer()
    assert _ids(synth.collect_results(None, graph)) == ["fee-a2", "fee-b"]
    assert synth.last_rollups_omitted == ["fee"]


def test_failed_child_without_a_matching_sibling_keeps_the_rollup():
    graph = _graph(
        _done("fee", "Work out the fee"),
        TaskNode(task_id="fee-a1", description="Plot cost", status=3, parent_task_id="fee"),
        _done("fee-b", "Multiply by 12", parent="fee"),
    )
    assert _ids(Synthesizer().collect_results(None, graph)) == ["fee", "fee-b"]


def test_three_levels_grandparent_and_parent_omitted_grandchildren_kept():
    graph = _graph(
        TaskNode(task_id="root", description="goal", status=1),
        _done("gp", "Plan the garden", parent="root"),
        _done("p", "Fee chain", parent="gp"),
        _done("g1", "Plot cost", parent="p"),
        _done("g2", "Multiply by 12", parent="p"),
    )
    synth = Synthesizer()
    assert _ids(synth.collect_results(None, graph, root_task_id="root")) == ["g1", "g2"]
    assert sorted(synth.last_rollups_omitted) == ["gp", "p"]
    assert synth.trim_counts["rollups_omitted"] == 2


def test_remaining_order_is_unchanged_and_counter_matches():
    # Dependency order: b depends on a, c depends on b; roll-up r (children
    # x, y) sits between them in insertion order.
    graph = _graph(
        _done("c", "third", deps=["b"]),
        _done("r", "roll-up"),
        _done("x", "child x", parent="r"),
        _done("a", "first"),
        _done("y", "child y", parent="r"),
        _done("b", "second", deps=["a"]),
    )
    before = [t for t in _ids(Synthesizer().collect_results(None, _graph(
        _done("c", "third", deps=["b"]), _done("r", "roll-up"),
        _done("x", "child x"), _done("a", "first"),
        _done("y", "child y"), _done("b", "second", deps=["a"]))))]
    synth = Synthesizer()
    after = _ids(synth.collect_results(None, graph))
    assert after == [t for t in before if t != "r"]
    assert synth.trim_counts["rollups_omitted"] == len(synth.last_rollups_omitted) == 1


def test_a_leaf_is_never_omitted():
    graph = _graph(_done("a", "first"), _done("b", "second"))
    synth = Synthesizer()
    assert _ids(synth.collect_results(None, graph)) == ["a", "b"]
    assert synth.last_rollups_omitted == []


def _retried_rollup_graph():
    # The root spawned the fee roll-up twice: the first attempt was abandoned,
    # the retry completed with both children and is omitted as a roll-up.
    return _graph(
        TaskNode(task_id="root", description="goal", status=1),
        TaskNode(task_id="fee-1", description="Work out the fee", status=3,
                 parent_task_id="root"),
        _done("fee-2", "work out  the FEE", parent="root"),
        _done("fee-a", "Plot cost", parent="fee-2"),
        _done("fee-b", "Multiply by 12", parent="fee-2"),
        TaskNode(task_id="rule", description="Abandonment rule", status=3,
                 parent_task_id="root"),
    )


def test_abandoned_attempt_of_an_omitted_rollup_is_not_listed_not_completed():
    graph = _retried_rollup_graph()
    synth = Synthesizer()
    results = synth.collect_results(None, graph, root_task_id="root")
    assert synth.last_rollups_omitted == ["fee-2"]
    missing = synth.collect_not_completed(graph, {"fee-1", "rule"}, results)
    assert "Work out the fee" not in missing


def test_abandoned_task_without_a_completed_match_is_still_listed():
    graph = _retried_rollup_graph()
    synth = Synthesizer()
    results = synth.collect_results(None, graph, root_task_id="root")
    assert synth.collect_not_completed(graph, {"fee-1", "rule"}, results) == [
        "Abandonment rule"]


def test_second_collect_results_does_not_reuse_omitted_ids_from_the_first():
    synth = Synthesizer()
    synth.collect_results(None, _retried_rollup_graph(), root_task_id="root")
    assert synth.last_rollups_omitted == ["fee-2"]
    # Second call on a graph with no roll-ups: the old ID must not linger or
    # cover the abandoned "Work out the fee" attempt.
    graph = _graph(
        TaskNode(task_id="fee-1", description="Work out the fee", status=3),
        TaskNode(task_id="fee-2", description="Work out the fee", status=0),
        _done("a", "Plot cost"),
    )
    results = synth.collect_results(None, graph)
    assert synth.last_rollups_omitted == []
    assert synth.collect_not_completed(graph, {"fee-1"}, results) == ["Work out the fee"]


# --------------------------------------------------------------- 2a problem text

class _Store:
    def write(self, *args, **kwargs):
        return 1

    def get_success_cache(self, *args, **kwargs):
        return None


class _Promotes:
    def decide(self, agent_node, **kwargs):
        return {"verdict": "promote", "reason": "ok"}


class _MockSynth:
    """Records the problem argument each synthesis path passes."""

    def __init__(self):
        self.problems = []
        self.trim_counts = {}

    def _record_trim(self, kind):
        self.trim_counts[kind] = self.trim_counts.get(kind, 0) + 1

    def run(self, colony, graph, problem, root_task_id=None, abandoned_ids=None):
        self.problems.append(problem)
        return "Each member gets one plot."

    def collect_results(self, colony, graph, root_task_id=None):
        return [{"task_id": "t-1", "description": "Split the plots",
                 "result": "One plot each."}]

    def collect_not_completed(self, graph, abandoned_ids, results):
        return []

    def format_output(self, results, problem, not_completed=None):
        self.problems.append(problem)
        return "Each member gets one plot."


RAW = "Share 12 plots among the members.\n... [TRUNCATED]"
GOAL = "Share 12 gardens."


def _orch(spec):
    orch = Orchestrator(ColonyState(initial_budget=1000, goal_embedding=None),
                        TaskGraph(), Messenger(), judge=_Promotes(), memory_store=_Store())
    orch.synthesizer = _MockSynth()
    orch.spec = spec
    orch.root_task_id = "root"
    return orch


def _complete_root(orch):
    orch.task_graph.add_task(TaskNode(task_id="root", description=GOAL, agent_id="r", status=1))
    orch.colony.register_agent(AgentNode(agent_id="r", role="executor", status="running",
                                         parent_id=None, task=GOAL, task_id="root"))
    event = Event(type="completion_request", from_agent="r")
    event.payload.update({"task_id": "root", "result": "Each member gets one plot."})
    orch.handle_completion(event)


def _partial(orch):
    orch.task_graph.add_task(TaskNode(task_id="root", description=GOAL, status=3))
    orch.abandoned_tasks.add("root")
    orch.terminate()


def test_success_path_passes_raw_text():
    orch = _orch({"raw_text": RAW, "goal": GOAL})
    _complete_root(orch)
    assert orch.synthesizer.problems == [RAW]
    assert "problem_fell_back_to_goal" not in orch.synthesizer.trim_counts


def test_partial_path_passes_raw_text():
    orch = _orch({"raw_text": RAW, "goal": GOAL})
    _partial(orch)
    assert orch.synthesizer.problems == [RAW]
    assert "problem_fell_back_to_goal" not in orch.synthesizer.trim_counts


def test_success_path_falls_back_to_goal_and_counts_it():
    orch = _orch({"raw_text": "", "goal": GOAL})
    _complete_root(orch)
    assert orch.synthesizer.problems == [GOAL]
    assert orch.synthesizer.trim_counts["problem_fell_back_to_goal"] == 1


def test_partial_path_falls_back_to_goal_and_counts_it(capsys):
    orch = _orch({"goal": GOAL})
    _partial(orch)
    assert orch.synthesizer.problems == [GOAL]
    assert orch.synthesizer.trim_counts["problem_fell_back_to_goal"] == 1
    assert "synthesis problem fell back to goal" in capsys.readouterr().out


def test_the_real_prompt_reads_the_request():
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return "Each member gets one plot."
    orch = _orch({"raw_text": RAW, "goal": GOAL})
    orch.synthesizer = Synthesizer(llm_call_fn=llm)
    orch.task_graph.add_task(TaskNode(task_id="root", description=GOAL, status=3))
    orch.task_graph.add_task(_done("t-1", "Split the plots", parent="root"))
    orch.abandoned_tasks.add("root")
    orch.terminate()
    assert f"PROBLEM: {RAW}\n" in seen["prompt"]
    assert GOAL not in seen["prompt"]


def test_no_spec_gives_an_empty_problem():
    orch = _orch(None)
    assert orch._synthesis_problem_text() == ""
    assert orch.synthesizer.trim_counts["problem_fell_back_to_goal"] == 1


# --------------------------------------------------------------- 2c sign-off

REPRO = ("Each member pays a fee of 12 x plot_cost per year. Members who stop "
         "tending lose the plot after two warnings. Final check passed. Ready "
         "to submit. Submission ID: submission_8c4f7a2b. Output Format: "
         "<OUTPUT>...<OUTPUT> END OF OUTPUT.")
BODY = ("Each member pays a fee of 12 x plot_cost per year. Members who stop "
        "tending lose the plot after two warnings.")


def test_reproduction_loses_the_final_check():
    kept, _ = trim_degenerate_tails(REPRO, None)
    assert kept == BODY


def test_final_check_colon_passed_is_cut():
    kept, _ = trim_degenerate_tails(BODY + " Final check: passed.", None)
    assert kept == BODY


def test_all_checks_passed_is_cut():
    kept, _ = trim_degenerate_tails(BODY + " All checks passed.", None)
    assert kept == BODY


def test_mid_answer_final_check_is_not_cut():
    text = ("Each member pays a fee of 12 x plot_cost per year. Final check passed. "
            "Members who stop tending lose the plot after two warnings. "
            "Plots are reassigned in spring.")
    kept, _ = trim_degenerate_tails(text, None)
    assert kept == text


def test_a_content_sentence_about_the_final_check_is_not_cut():
    text = BODY + " The final check passed inspection on three plots."
    kept, _ = trim_degenerate_tails(text, None)
    assert kept == text
    assert not _SIGNOFF_RE.match("The final check passed inspection on three plots.")
