"""
PATCH 38: the pre-judge trim keeps "Rs. 750" whole and sizes a decomposer's
roll-up by its subtasks. PATCH 39: a decomposer respawned on a task that
already has subtasks starts as having spawned. PATCH 40: a goal draft that
adds parts the request never asked for fails fidelity.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode
from text_utils import (added_goal_clauses, drop_incomplete_tail, goal_fidelity,
                        split_sentences, trim_to_sentences)

REQUEST = ("A community garden has 12 plots and 20 members who want one, decide "
           "how to allocate plots fairly, set a yearly fee so that 9,000 rupees "
           "of costs are covered, split watering duty across the summer months, "
           "and decide what to do with members who stop tending their plot.")

RUN17_ROOT_REPORT = (
    "Plot assignment uses a round-robin voting method to allocate 12 out of 20 "
    "plots fairly based on preferences. Annual fees at Rs. 750 per plot ensure "
    "full coverage of Rs. 9,000 costs. Watering is divided evenly into two "
    "weekly assignments per plot spread across six summer months. Inactive "
    "members receive a two-week notice and a 24-hour transfer window, after "
    "which the plot becomes vacant until the next allocation cycle.")


# ------------------------------------------------------------ PATCH 38

def test_rs_abbreviation_does_not_end_a_sentence():
    sentences = split_sentences("Fees are Rs. 750 per plot. Done.")
    assert sentences == ["Fees are Rs. 750 per plot.", "Done."]


def test_other_abbreviations_are_kept_together():
    assert len(split_sentences("Use a draw, e.g. a lottery. Then vote.")) == 2
    assert len(split_sentences("Plot No. 4 is free. Ask Dr. Rao.")) == 2


def test_ordinary_sentences_still_split():
    assert len(split_sentences("One. Two! Three? Four.")) == 4


def test_trim_keeps_the_fee_sentence_whole():
    trimmed = trim_to_sentences(RUN17_ROOT_REPORT, max_sentences=3,
                                max_words=60, marker=False)
    assert "Rs. 750 per plot" in trimmed
    assert not trimmed.rstrip().endswith("Rs.")


def test_drop_incomplete_tail_does_not_stop_at_rs():
    text = "The fee is fixed. Each plot pays Rs. 750 per"
    assert drop_incomplete_tail(text) == "The fee is fixed."


def _orch_with_root(children=4):
    orch = Orchestrator(ColonyState(initial_budget=500, goal_embedding=None),
                        TaskGraph(), Messenger())
    orch.task_graph.add_task(TaskNode(task_id="root", description="goal",
                                      agent_id="r", status=1,
                                      required_role="decomposer"))
    for i in range(children):
        orch.task_graph.add_task(TaskNode(task_id=f"c{i}", description=f"part {i}",
                                          status=2, result=f"answer {i}.",
                                          parent_task_id="root"))
    node = AgentNode(agent_id="r", role="decomposer", status="running",
                     parent_id=None, task="goal", task_id="root", generation=0)
    orch.colony.register_agent(node)
    return orch


def test_rollup_budget_covers_one_sentence_per_subtask():
    orch = _orch_with_root(children=4)
    sentences, words = orch._report_trim_budget("r", "root")
    assert sentences == 5
    assert words >= 5 * Orchestrator.ROLLUP_TRIM_WORDS_PER_SENTENCE


def test_rollup_budget_is_capped():
    orch = _orch_with_root(children=20)
    sentences, _ = orch._report_trim_budget("r", "root")
    assert sentences == Orchestrator.ROLLUP_TRIM_MAX_SENTENCES


def test_executor_budget_is_unchanged():
    orch = _orch_with_root(children=0)
    orch.colony.register_agent(AgentNode(agent_id="e", role="executor",
                                         status="running", parent_id="r",
                                         task="x", task_id="c9", generation=1))
    assert orch._report_trim_budget("e", "c9") == (3, 60)


def test_run17_root_report_reaches_the_judge_whole():
    orch = _orch_with_root(children=4)
    sentences, words = orch._report_trim_budget("r", "root")
    trimmed = trim_to_sentences(RUN17_ROOT_REPORT, max_sentences=sentences,
                                max_words=words, marker=False)
    assert "Inactive members" in trimmed
    assert "Rs. 750" in trimmed


# ------------------------------------------------------------ PATCH 39

def test_respawned_decomposer_inherits_has_spawned():
    orch = _orch_with_root(children=4)
    node = AgentNode(agent_id="r2", role="decomposer", status="running",
                     parent_id=None, task="goal", task_id="root", generation=0)
    orch.colony.register_agent(node)
    orch._inherit_existing_children("r2", "root", node)
    assert node.has_spawned


def test_first_decomposer_on_an_empty_task_has_not_spawned():
    orch = _orch_with_root(children=0)
    node = AgentNode(agent_id="r2", role="decomposer", status="running",
                     parent_id=None, task="goal", task_id="root", generation=0)
    orch.colony.register_agent(node)
    orch._inherit_existing_children("r2", "root", node)
    assert not node.has_spawned


def test_inherited_results_are_injected_into_a_live_agent():
    orch = _orch_with_root(children=2)
    node = AgentNode(agent_id="r2", role="decomposer", status="running",
                     parent_id=None, task="goal", task_id="root", generation=0)
    orch.colony.register_agent(node)
    agent = Agent(tokeniser=None, model=None, message=orch.messenger, node=node)
    orch.live_agents["r2"] = agent
    orch._inherit_existing_children("r2", "root", node)
    assert "[CHILD RESULT - c0]" in agent.thought_process
    assert "[CHILD RESULT - c1]" in agent.thought_process


def test_respawned_root_with_all_subtasks_done_can_only_report():
    orch = _orch_with_root(children=4)
    node = AgentNode(agent_id="r2", role="decomposer", status="running",
                     parent_id=None, task="goal", task_id="root", generation=0)
    orch.colony.register_agent(node)
    agent = Agent(tokeniser=None, model=None, message=orch.messenger, node=node)
    agent.child_status_fn = lambda: orch._direct_child_statuses("root")
    orch._inherit_existing_children("r2", "root", node)
    assert agent.final_only
    assert agent.final_actions == ("REPORT", "DIE")


# ------------------------------------------------------------ PATCH 40

RUN17_GOAL = ("Allocate 12 garden plots fairly among 20 applicants while covering "
              "$9k fees and assigning water duties evenly over summer seasons. "
              "Split up inactive members by fall harvests when they stop caring. "
              "Make sure it all fits into your budget plan document. Fix any "
              "conflicts during meetings before finalizing anything! Write this "
              "down clearly now!")


def test_run17_goal_added_parts_are_found():
    added = [c for c, _ in added_goal_clauses(REQUEST, RUN17_GOAL)]
    assert any("budget plan document" in c for c in added)
    assert any("conflicts during meetings" in c for c in added)
    assert not any("inactive members" in c for c in added)


def test_goal_with_added_parts_fails_fidelity():
    check = goal_fidelity(REQUEST, RUN17_GOAL)
    assert not check["ok"]
    assert check["added"]


def test_faithful_goal_has_no_added_parts():
    goal = ("Allocate 12 plots fairly among 20 members, set a yearly fee covering "
            "9,000 rupees, split watering duty across the summer months, and "
            "decide what to do with members who stop tending their plot.")
    assert goal_fidelity(REQUEST, goal)["added"] == []


# ------------------------------------------------------------ review fixes

def test_sentence_ending_on_ambiguous_abbreviation_is_kept():
    assert drop_incomplete_tail("Mix it. Water for 30 min.") == "Mix it. Water for 30 min."
    assert drop_incomplete_tail("Is it free? The answer is no.") == "Is it free? The answer is no."


def test_inner_periods_of_eg_are_not_sentence_ends():
    assert drop_incomplete_tail("Use a draw, e.g. a lott") == "Use a draw, e.g. a lott"


def test_ambiguous_abbreviation_splits_before_a_capital():
    assert split_sentences("Water for 30 min. Then rest.") == ["Water for 30 min.", "Then rest."]
    assert split_sentences("Plot No. 4 is free.") == ["Plot No. 4 is free."]


def test_one_clause_synonym_goal_is_not_flagged_as_added():
    request = ("Our hiking group argues every month about which trail to do. "
               "How should we decide?")
    assert added_goal_clauses(request, "Settle the monthly route dispute with a voting method.") == []


def test_respawn_keeping_all_completed_subtasks_is_priced_as_spawned():
    orch = _orch_with_root(children=4)
    assert orch._kept_children_count("root") == 4


def test_respawn_with_a_failed_subtask_is_priced_as_planning():
    orch = _orch_with_root(children=2)
    orch.task_graph.add_task(TaskNode(task_id="cx", description="failed part",
                                      status=3, parent_task_id="root"))
    assert orch._kept_children_count("root") == 0


# ------------------------------------------------------------ integration fixes

def test_abandoned_subtask_redone_by_a_later_sibling_is_replaced():
    orch = _orch_with_root(children=0)
    orch.task_graph.add_task(TaskNode(task_id="a1", description="Draft an exit policy",
                                      status=3, parent_task_id="root",
                                      label="inactive_policy"))
    orch.task_graph.add_task(TaskNode(task_id="a2", description="Define a notice policy",
                                      status=2, result="Two weeks notice.",
                                      parent_task_id="root", label="inactive_policy"))
    statuses = [r["status"] for r in orch._direct_child_statuses("root")]
    assert statuses == ["replaced", "completed"]
    assert orch._kept_children_count("root") == 1


def test_root_with_a_replaced_subtask_can_only_report():
    orch = _orch_with_root(children=2)
    orch.task_graph.add_task(TaskNode(task_id="a1", description="part 9 policy",
                                      status=3, parent_task_id="root", label="part9_policy"))
    orch.task_graph.add_task(TaskNode(task_id="a2", description="part 9 policy redo",
                                      status=2, result="done.", parent_task_id="root",
                                      label="part9_policy"))
    node = AgentNode(agent_id="r2", role="decomposer", status="running",
                     parent_id=None, task="goal", task_id="root", generation=0,
                     has_spawned=True)
    agent = Agent(tokeniser=None, model=None, message=orch.messenger, node=node)
    agent.child_status_fn = lambda: orch._direct_child_statuses("root")
    assert agent.final_actions == ("REPORT", "DIE")
    assert orch._report_trim_budget("r", "root")[0] == 4


def test_unrelated_failed_subtask_is_not_replaced():
    orch = _orch_with_root(children=0)
    orch.task_graph.add_task(TaskNode(task_id="a1", description="Split watering duty",
                                      status=3, parent_task_id="root", label="water"))
    orch.task_graph.add_task(TaskNode(task_id="a2", description="Set the yearly fee",
                                      status=2, result="750.", parent_task_id="root",
                                      label="fee"))
    assert [r["status"] for r in orch._direct_child_statuses("root")][0] == "failed"


def test_goal_cap_keeps_rs_figure_whole():
    from text_utils import dedupe_global_and_cap
    capped = dedupe_global_and_cap(
        "Set a fee of Rs. 9,000 so costs are covered. Split watering duty. "
        "Decide what to do with members who stop tending.", max_chars=300)
    assert "Rs. 9,000" in capped


def test_goal_clause_split_keeps_rs_figure_whole():
    from text_utils import goal_clauses
    assert goal_clauses("Set a yearly fee covering Rs. 9,000 of costs for the garden.") == [
        "Set a yearly fee covering Rs. 9,000 of costs for the garden"]


def _replaced_status(first, second, label1=None, label2=None):
    orch = _orch_with_root(children=0)
    orch.task_graph.add_task(TaskNode(task_id="f", description=first, status=3,
                                      parent_task_id="root", label=label1))
    orch.task_graph.add_task(TaskNode(task_id="g", description=second, status=2,
                                      result="ok.", parent_task_id="root", label=label2))
    return orch._direct_child_statuses("root")[0]["status"]


def test_different_parts_with_shared_words_are_not_replaced():
    assert _replaced_status("set the fee per plot", "set the fee per member") == "failed"
    assert _replaced_status("schedule summer watering duty",
                            "schedule winter watering duty") == "failed"
    assert _replaced_status("decide how to allocate plots fairly",
                            "decide what to do with inactive plots") == "failed"


def test_single_letter_labels_do_not_mark_replaced():
    assert _replaced_status("split watering duty", "set the yearly fee", "a", "a") == "failed"


def test_near_identical_rewording_is_replaced():
    assert _replaced_status("Draft exit protocol for inactive garden members",
                            "Draft an exit protocol for inactive garden members") == "replaced"


def test_ghost_goal_cap_keeps_rs_figure_whole():
    from text_utils import dedupe_and_cap
    capped = dedupe_and_cap("Set a fee of Rs. 9,000 so costs are covered. Split watering duty.",
                            max_chars=300)
    assert "Rs. 9,000" in capped


def test_goal_line_keeps_a_long_ask_sentence():
    orch = _orch_with_root(children=0)
    asks = ("Decide how to pick books each month, set a yearly fee of Rs. 200 per member, "
            "plan the meeting rota, choose a venue that fits everyone, and decide what to "
            "do with members who stop attending for three months in a row without notice.")
    line = orch._goal_line("We run a 40-member book club. " + asks)
    assert "Rs. 200" in line and "meeting rota" in line


def test_goal_line_collapses_a_looping_goal():
    orch = _orch_with_root(children=0)
    loop = "Allocate the plots fairly. " * 40
    assert orch._goal_line(loop).count("Allocate the plots fairly") == 1


def test_goal_line_cuts_a_very_long_sentence_at_a_word():
    orch = _orch_with_root(children=0)
    line = orch._goal_line("Short opener. " + "word " * 300)
    assert len(line) <= Orchestrator.GOAL_LINE_MAX_CHARS + 3
    assert "word" in line


def test_goal_line_caps_one_very_long_sentence():
    orch = _orch_with_root(children=0)
    line = orch._goal_line("Plan " + "and decide the garden rota " * 80 + "now.")
    assert len(line) <= Orchestrator.GOAL_LINE_MAX_CHARS + 3


def test_no_goal_cap_left_at_300_chars():
    import inspect
    import orchestrator
    assert "max_chars=300" not in inspect.getsource(orchestrator)
