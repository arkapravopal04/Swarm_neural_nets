"""
Regression tests for the degeneracy check on the FINAL synthesized answer.

handle_completion has run a repeated-sentence/degeneracy pass over every
individual REPORT since the repetition-collapse fix, but nothing ran one
over the answer those REPORTs are stitched into -- the only text in a run
the user actually reads. One real run therefore ended on "ACTION REQUESTED:
RUN OR ABORT?" and a repeated exemplar sentence: neither is an adjacent
duplicate to collapse nor a closer to trim, so drop_incomplete_tail,
trim_closer_tail and dedupe_global_and_cap all left them untouched.

Covered:

  * looks_degenerate is now the one detector, with Agent._looks_degenerate
    delegating to it rather than keeping its own copy of the rules,
  * degeneracy_cut cuts at the FIRST bad sentence and keeps the good prefix,
    for each of the four shapes it is responsible for,
  * ordinary English that merely opens a line with "Action required:" is
    not cut, which is why the shout pattern is case-sensitive,
  * closer-cycling buried mid-text is deliberately NOT cut,
  * Synthesizer.format_output applies the cut to the real decode, reports
    it, and refuses to hand back an answer that was degenerate from its
    first sentence.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent, EXEMPLAR_SUBTASK_DESCRIPTIONS
from synthesizer import Synthesizer
from text_utils import degeneracy_cut, looks_degenerate


CLEAN = (
    "Hold the review on the second Tuesday of each month. "
    "Give every member one vote and settle ties with a coin toss. "
    "Publish the minutes within three working days."
)
REPEATED = "These actions directly address the two primary goals of the review."
BACKTICK = chr(96)


# ---------------------------------------------------------------------------
# One detector
# ---------------------------------------------------------------------------

def test_agent_delegates_to_the_shared_detector():
    # Not "the two agree today" -- the method calls the function, so gating
    # generation and gating the final answer can never drift apart.
    import agent_node
    assert agent_node.looks_degenerate is looks_degenerate
    assert Agent._looks_degenerate(CLEAN) is looks_degenerate(CLEAN) is False
    looping = "The two primary goals are cost and safety. " * 3
    assert Agent._looks_degenerate(looping) is looks_degenerate(looping) is True


def test_detector_still_catches_its_three_original_shapes():
    # A long sentence repeating ONCE more (threshold 2 above 60 chars).
    assert looks_degenerate(REPEATED + " " + REPEATED) is True
    # Bracket/backtick soup in the tail.
    assert looks_degenerate("Here is the plan. " + ("[{}]" + BACKTICK + "<>") * 12) is True
    # Closer-cycling: every sentence unique, no brackets.
    assert looks_degenerate("Ready. Finalized. Deploying. Deployment. Done. Submitted.") is True


# ---------------------------------------------------------------------------
# Where it cuts
# ---------------------------------------------------------------------------

def test_clean_prose_is_returned_untouched():
    assert degeneracy_cut(CLEAN) == (CLEAN, None)
    assert degeneracy_cut("") == ("", None)


def test_cuts_at_the_repeat_and_keeps_the_prefix():
    kept, reason = degeneracy_cut(CLEAN + " " + REPEATED + " " + REPEATED)
    assert reason == "a sentence repeated"
    # The first occurrence is content; only the loop is dropped.
    assert kept == CLEAN + " " + REPEATED


def test_cuts_at_shouted_harness_scaffolding():
    kept, reason = degeneracy_cut(CLEAN + " ACTION REQUESTED: RUN OR ABORT?")
    assert reason == "harness scaffolding echoed back"
    assert kept == CLEAN


def test_cuts_at_a_bare_action_or_payload_label():
    for echoed in ("ACTION: REPORT", "PAYLOAD: the final answer"):
        kept, reason = degeneracy_cut(CLEAN + " " + echoed)
        assert reason == "harness scaffolding echoed back"
        assert kept == CLEAN


def test_a_bulleted_answer_keeps_its_bullets_and_loses_only_the_scaffolding():
    # The shape that matters most and the one a sentence-boundary cut got
    # catastrophically wrong: a bulleted plan carries no sentence terminator
    # before its trailing scaffolding line, so the whole answer is one
    # "sentence". Cutting at the sentence start threw the answer away.
    answer = ("Here is the final plan:\n"
              "- Hold the review monthly\n"
              "- Publish minutes in three days")
    kept, reason = degeneracy_cut(answer + "\nACTION REQUESTED: RUN OR ABORT?")
    assert reason == "harness scaffolding echoed back"
    assert kept == answer  # line breaks and bullets intact


def test_prose_with_angle_brackets_is_not_a_placeholder_echo():
    # Both of these were read as placeholders by "anything in angle
    # brackets" and cost the entire answer.
    for prose in (
        "Keep the tank between <5 and >10 degrees at all times. That is the rule.",
        "Send the minutes to Smith <smith@example.com> each month. Then archive them.",
        "Hold the review monthly.<br>Publish the minutes.",
    ):
        assert degeneracy_cut(prose) == (prose, None)


def test_ordinary_english_beginning_with_action_is_not_cut():
    # The shout pattern is case-sensitive precisely so this survives: it is
    # a real sentence in a finished answer, not the harness talking.
    prose = CLEAN + " Action required: book the venue by Friday."
    assert degeneracy_cut(prose) == (prose, None)


def test_cuts_at_an_echoed_prompt_exemplar():
    echoed = EXEMPLAR_SUBTASK_DESCRIPTIONS[1]
    kept, reason = degeneracy_cut(
        CLEAN + " Then handle " + echoed + " before the deadline.",
        exemplars=EXEMPLAR_SUBTASK_DESCRIPTIONS,
    )
    assert reason in ("a prompt exemplar echoed back", "a prompt placeholder echoed back")
    assert kept == CLEAN


def test_every_exemplar_is_covered_by_one_check_or_the_other():
    # Why both checks exist rather than just the cheaper one: the shape
    # regex is words-and-spaces only, so the exemplar carrying commas
    # ("<final step that needs the results of parts A, B and C>") is caught
    # only by the exemplar list -- which is what the synthesizer passes in.
    for echoed in EXEMPLAR_SUBTASK_DESCRIPTIONS:
        text = CLEAN + " Next, " + echoed + "."
        assert degeneracy_cut(text, exemplars=EXEMPLAR_SUBTASK_DESCRIPTIONS) == (
            CLEAN,
            "a prompt placeholder echoed back"
            if degeneracy_cut(text)[1] else "a prompt exemplar echoed back",
        )


def test_cuts_at_a_half_rewritten_placeholder_without_the_exemplar_list():
    # Matches nothing in EXEMPLAR_SUBTASK_DESCRIPTIONS exactly, and is
    # caught on shape alone -- no exemplars argument passed.
    kept, reason = degeneracy_cut(CLEAN + " Next, do <part B of the task>.")
    assert reason == "a prompt placeholder echoed back"
    assert kept == CLEAN


def test_cuts_at_bracket_noise():
    noise = "[[{}]]" + BACKTICK + "<><>[]{}" + BACKTICK + "[]<>"
    kept, reason = degeneracy_cut(CLEAN + " " + noise)
    assert reason == "bracket/backtick noise"
    assert kept == CLEAN


def test_mid_text_closer_run_is_left_alone():
    # A degeneracy SIGNAL, but not a cut: the sentences after it are real
    # content, and trim_closer_tail already handles the tail case.
    text = "Ready. Finalized. Deploying. Done. " + CLEAN
    assert degeneracy_cut(text) == (text, None)
    assert looks_degenerate(text) is True


def test_a_text_degenerate_from_its_first_sentence_cuts_to_nothing():
    kept, reason = degeneracy_cut("ACTION REQUESTED: RUN OR ABORT?")
    assert reason == "harness scaffolding echoed back"
    assert kept == ""


# ---------------------------------------------------------------------------
# The synthesizer applies it
# ---------------------------------------------------------------------------

RESULTS = [{"task_id": "t1", "description": "Decide the review cadence", "result": "Monthly."}]
PROBLEM = "How should the club run its reviews?"


def _synth(answer):
    return Synthesizer(llm_call_fn=lambda prompt: answer)


def test_format_output_trims_the_degenerate_tail_off_the_final_answer(capsys):
    # The exact shape of the real run: a repeat whose second copy the
    # generation budget cut off mid-sentence, then scaffolding.
    raw = (CLEAN + " " + REPEATED + " " + REPEATED[:-14]
           + "\nACTION REQUESTED: RUN OR ABORT?")
    out = _synth(raw).format_output(RESULTS, PROBLEM)
    assert "ACTION REQUESTED" not in out
    assert out.count(REPEATED) == 1
    assert out.startswith("Hold the review on the second Tuesday")
    assert "[synthesis-trim]" in capsys.readouterr().out


def test_format_output_refuses_an_answer_that_was_degenerate_throughout(capsys):
    out = _synth("ACTION REQUESTED: RUN OR ABORT?").format_output(RESULTS, PROBLEM)
    assert out == Synthesizer.DEGENERATE_ANSWER_MESSAGE
    assert "[synthesis-trim]" in capsys.readouterr().out


def test_format_output_leaves_a_clean_answer_alone(capsys):
    assert _synth(CLEAN).format_output(RESULTS, PROBLEM) == CLEAN
    assert "[synthesis-trim]" not in capsys.readouterr().out


def test_format_output_warns_but_still_ships_a_mid_text_closer_run(capsys):
    out = _synth("Ready. Finalized. Deploying. Done. " + CLEAN).format_output(RESULTS, PROBLEM)
    assert "Publish the minutes within three working days." in out
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# A truncating caller cannot use the IGNORECASE label pattern
#
# degeneracy_cut used to reach for _SCAFFOLD_LINE_RE, which is IGNORECASE and
# matches a bare "Action:"/"Payload:" label anywhere a line starts with one.
# That is fine for strip_scaffolding_lines, which DELETES the matching line
# and keeps everything around it -- a false positive there costs one line. It
# is not fine here, which truncates from the match onward: a plan written with
# labelled lines lost everything after its first label.
# ---------------------------------------------------------------------------

LABELLED_PLAN = (
    "Recommended plan for the launch:\n\n"
    "Action: Book the venue by Friday.\n"
    "Owner: Priya.\n"
    "Payload: 200 attendees, catering included.\n\n"
    "The remaining budget covers signage and travel."
)


def test_a_plan_written_with_action_and_payload_labels_survives_intact():
    assert degeneracy_cut(LABELLED_PLAN) == (LABELLED_PLAN, None)


def test_the_labelled_plan_survives_the_real_synthesis_path():
    # The end-to-end shape of the regression: not just the detector agreeing
    # in isolation, but the answer the user would have been handed -- byte
    # for byte, line breaks included.
    assert _synth(LABELLED_PLAN).format_output(RESULTS, PROBLEM) == LABELLED_PLAN


def test_capitalised_protocol_labels_are_still_cut():
    # What the case-sensitivity buys has to keep working: these are the
    # harness's own labels, not a heading someone wrote.
    for echoed in ("ACTION: REPORT", "PAYLOAD: the final answer",
                   "ACTION REQUESTED: RUN OR ABORT?"):
        kept, reason = degeneracy_cut(CLEAN + " " + echoed)
        assert reason == "harness scaffolding echoed back", echoed
        assert kept == CLEAN, echoed


def test_prompt_phrases_are_still_cut_in_either_case():
    # The phrase half of the scaffolding pattern stays IGNORECASE: no
    # finished answer says these, whatever the capitalisation.
    for echoed in ("Your next action: THINK", "AVAILABLE ACTIONS:",
                   "Your output must be exactly one action block"):
        kept, reason = degeneracy_cut(CLEAN + "\n" + echoed)
        assert reason == "harness scaffolding echoed back", echoed
        assert kept == CLEAN, echoed


# ---------------------------------------------------------------------------
# The exit guard: format_output is not the only way an answer leaves the
# orchestrator. A synthesizer that raised, a colony built without one, and
# the root-task fallback all return an agent's REPORT verbatim -- and a
# REPORT is only deduped and trimmed to three sentences on its way through
# handle_completion, so scaffolding or an exemplar echo inside one reached
# the user untouched.
# ---------------------------------------------------------------------------

from colony_state import ColonyState  # noqa: E402
from event_queue import Messenger  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from task_graph import TaskGraph  # noqa: E402


def _orch():
    return Orchestrator(ColonyState(initial_budget=500, goal_embedding=None),
                        TaskGraph(), Messenger())


def test_exit_guard_cuts_scaffolding_off_a_raw_agent_report(capsys):
    orch = _orch()
    out = orch._guard_final_answer(CLEAN + "\nACTION REQUESTED: RUN OR ABORT?")
    assert out == CLEAN
    assert orch.colony.verdict_counts.get("final_answer_cut") == 1
    assert "[final-answer]" in capsys.readouterr().out


def test_exit_guard_cuts_an_echoed_exemplar_off_a_raw_agent_report():
    orch = _orch()
    out = orch._guard_final_answer(
        CLEAN + " Then handle " + EXEMPLAR_SUBTASK_DESCRIPTIONS[1] + " next."
    )
    assert out == CLEAN


def test_exit_guard_reports_a_wholly_degenerate_answer_rather_than_blanking_it():
    orch = _orch()
    out = orch._guard_final_answer("ACTION REQUESTED: RUN OR ABORT?")
    assert out == Synthesizer.DEGENERATE_ANSWER_MESSAGE
    assert orch.colony.verdict_counts.get("final_answer_empty_after_cut") == 1


def test_exit_guard_leaves_a_clean_answer_and_a_non_string_alone(capsys):
    orch = _orch()
    assert orch._guard_final_answer(CLEAN) == CLEAN
    # The success branch can return colony.results, a dict -- nothing to cut.
    results = {"final_spec": CLEAN}
    assert orch._guard_final_answer(results) is results
    assert "[final-answer]" not in capsys.readouterr().out


def test_exit_guard_is_a_no_op_over_an_already_cut_synthesis():
    # Why covering the one return is cheaper than covering each source: the
    # cut is idempotent, so the guard costs nothing on the path that already
    # ran it.
    orch = _orch()
    synthesized = _synth(CLEAN + "\nACTION: REPORT").format_output(RESULTS, PROBLEM)
    assert orch._guard_final_answer(synthesized) == synthesized
    assert "final_answer_cut" not in orch.colony.verdict_counts


def test_terminate_guards_whatever_it_returns(capsys):
    # The real exit, not the helper: a root task whose result carries
    # scaffolding, with no synthesizer configured.
    from task_graph import TaskNode

    orch = _orch()
    orch.root_task_id = "root_task_0"
    orch.task_graph.add_task(TaskNode(task_id="root_task_0", description="plan",
                                      agent_id="a1", status=2))
    orch.task_graph.tasks["root_task_0"].result = CLEAN + "\nACTION REQUESTED: RUN OR ABORT?"

    assert orch.terminate() == CLEAN
    out = capsys.readouterr().out
    assert "FINAL ANSWER GUARDS" in out


def test_the_guard_ledger_stays_quiet_when_nothing_fired(capsys):
    from task_graph import TaskNode

    orch = _orch()
    orch.root_task_id = "root_task_0"
    orch.task_graph.add_task(TaskNode(task_id="root_task_0", description="plan",
                                      agent_id="a1", status=2))
    orch.task_graph.tasks["root_task_0"].result = CLEAN

    assert orch.terminate() == CLEAN
    assert "FINAL ANSWER GUARDS" not in capsys.readouterr().out


def test_the_ledger_counts_what_the_synthesizer_trimmed():
    synth = _synth(CLEAN + "\nACTION REQUESTED: RUN OR ABORT?")
    synth.format_output(RESULTS, PROBLEM)
    assert synth.trim_counts == {"cut": 1}

    empty = _synth("ACTION REQUESTED: RUN OR ABORT?")
    empty.format_output(RESULTS, PROBLEM)
    assert empty.trim_counts == {"cut": 1, "empty": 1}

    shipped = _synth("Ready. Finalized. Deploying. Done. " + CLEAN)
    shipped.format_output(RESULTS, PROBLEM)
    assert shipped.trim_counts.get("shipped_degenerate") == 1


# ---------------------------------------------------------------------------
# Layout
#
# degeneracy_cut returns a slice precisely so a bulleted answer keeps its
# bullets -- and then the cleaning passes after it rejoined the sentences
# with single spaces and handed the user one flattened paragraph. Every pass
# on this path now preserves the layout of the part it keeps.
# ---------------------------------------------------------------------------

BULLETED = (
    "Here is the plan for the review.\n\n"
    "1. Hold it on the second Tuesday of each month.\n"
    "2. Give every member one vote.\n"
    "3. Publish the minutes within three working days.\n\n"
    "Costs stay inside the existing budget."
)


def test_a_bulleted_final_answer_reaches_the_user_as_bullets():
    assert _synth(BULLETED).format_output(RESULTS, PROBLEM) == BULLETED


def test_layout_survives_a_cut_and_a_dedupe_together():
    raw = BULLETED + "\n" + BULLETED.splitlines()[2] + "\nACTION REQUESTED: RUN OR ABORT?"
    out = _synth(raw).format_output(RESULTS, PROBLEM)
    assert out == BULLETED                      # repeat and scaffolding gone
    assert "\n2. Give every member one vote." in out   # numbering intact


def test_trim_closer_tail_keeps_the_layout_of_what_it_keeps():
    from text_utils import trim_closer_tail

    bulleted = "- Hold the review monthly.\n- Publish the minutes."
    # CLOSER_MIN_RUN of them, so the trim actually fires.
    assert trim_closer_tail(
        bulleted + "\nDone. Submitted. Confirmed. Completed."
    ) == bulleted


def test_flattening_is_still_the_default_for_prompt_bound_text():
    # The other callers -- a goal, a fail_reason, a DIE line -- are cleaning
    # a string that goes back into a prompt on one line. Preserving breaks
    # there would be noise, so the option is off unless asked for.
    from text_utils import dedupe_global_and_cap

    assert dedupe_global_and_cap("Do X.\nDo Y.") == "Do X. Do Y."
    assert dedupe_global_and_cap("Do X.\nDo Y.", keep_line_breaks=True) == "Do X.\nDo Y."


def test_a_dropped_duplicate_takes_its_own_line_break_with_it():
    from text_utils import dedupe_global_and_cap

    out = dedupe_global_and_cap("- a thing.\n- a thing.\n- other.", keep_line_breaks=True)
    assert out == "- a thing.\n- other."


def test_the_cap_still_counts_the_separators_it_keeps():
    from text_utils import dedupe_global_and_cap

    text = "Aaaa aaaa aaaa.\nBbbb bbbb bbbb.\nCccc cccc cccc."
    for keep in (False, True):
        out = dedupe_global_and_cap(text, max_chars=34, keep_line_breaks=keep)
        assert out.endswith("...")
        assert len(out) - 3 <= 34, out


# ---------------------------------------------------------------------------
# Status-report / deploy-log tails on the final answer
#
# A final answer ended (the second run in which the synthesis itself did it):
#   "...Deployment timestamp: 2023-11-03T14:23:10Z. Metrics report:
#    accuracy=1.000000, consistency=1.000000... Final state: COMPLETED...
#    Report end."
# after the "ready to deploy / done / go" loop seen in single REPORTs. Nothing
# in it repeats three times and "Go." breaks the closer run, so every pass let
# it through, and the global dedupe turned the repeated loop into one copy.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from text_utils import cut_adjacent_repeat, trim_status_tail, ungrounded_telemetry  # noqa: E402

DEPLOY_LOG = (" Deployment timestamp: 2023-11-03T14:23:10Z. Metrics report: "
              "accuracy=1.000000, consistency=1.000000... Final state: COMPLETED... "
              "Report end.")
LOOP = " Ready to deploy. Done. Go."


@pytest.mark.parametrize("tail", [
    LOOP + DEPLOY_LOG,
    DEPLOY_LOG,
    LOOP + LOOP,
    LOOP + LOOP + DEPLOY_LOG,
    # A log block written as unterminated lines, which the sentence splitter
    # alone would read as one long "sentence" glued to the answer.
    "\nDeployment timestamp: 2023-11-03T14:23:10Z\nMetrics report: accuracy=1.000000, "
    "consistency=1.000000\nFinal state: COMPLETED\nReport end.",
    " Final state: COMPLETED. Report end.",
    "\n- Status: DONE\n- accuracy=0.99, recall=0.97",
])
def test_format_output_cuts_a_status_report_tail(tail, capsys):
    synth = _synth(CLEAN + tail)
    assert synth.format_output(RESULTS, PROBLEM) == CLEAN
    assert synth.trim_counts.get("status_tail") == 1
    assert "status-report tail" in capsys.readouterr().out


def test_a_repeated_loop_is_cut_not_collapsed_into_one_clean_looking_copy():
    """The dedupe kept one "Ready to deploy. Done. Go." and everything after
    it. Cut at the second copy, and the first goes with the status tail."""
    synth = _synth(CLEAN + LOOP + LOOP + " Deployment approved for all regions.")
    out = synth.format_output(RESULTS, PROBLEM)
    assert "Ready to deploy" not in out and "all regions" not in out
    assert synth.trim_counts.get("repeat_cut") == 1


@pytest.mark.parametrize("answer", [
    CLEAN + " Decision: APPROVED.",                 # a decision, not a log state
    CLEAN + " Result: PASS.",                       # a verifier's verdict
    CLEAN + " Status: VERIFIED.",
    CLEAN + " Status: approved by all members.",    # not SHOUTED
    CLEAN + " The first meeting is on 2023-11-03.",  # a date in prose
    CLEAN + "\nMeeting date: 2023-11-03.",          # a date with no time of day
    CLEAN + " Go.",                                 # one go-word is not the loop
    "Is the club ready to start? Yes. Go.",         # the tail answers a question
    BULLETED,
])
def test_real_answers_keep_their_endings(answer):
    assert trim_status_tail(answer, grounding=PROBLEM) == (answer, None)
    assert _synth(answer).format_output(RESULTS, PROBLEM) == answer


def test_a_timestamp_the_results_actually_gave_is_kept():
    results = [{"task_id": "t1", "description": "Find the last deploy",
                "result": "The last deploy ran at 2023-11-03T14:23:10Z."}]
    answer = CLEAN + "\nLast deployment timestamp: 2023-11-03T14:23:10Z."
    assert _synth(answer).format_output(results, PROBLEM) == answer


def test_a_metric_is_grounded_only_by_the_same_number():
    grounding = "Recall was 10 percent last year."
    assert trim_status_tail(CLEAN + " Metrics report: recall=1.", grounding)[1] is not None
    assert trim_status_tail(CLEAN + " Metrics report: recall=10.", grounding)[1] is None


def test_a_made_up_metric_mid_answer_is_reported_not_cut(capsys):
    answer = "Metrics report: accuracy=1.000000.\n" + CLEAN
    synth = _synth(answer)
    assert synth.format_output(RESULTS, PROBLEM) == answer
    assert synth.trim_counts.get("shipped_telemetry") == 1
    assert "timestamp/metric line" in capsys.readouterr().out
    assert ungrounded_telemetry(answer, grounding="accuracy=1.000000") == []


def test_an_answer_that_is_only_a_status_report_is_refused():
    synth = _synth("Final state: COMPLETED. Report end.")
    assert synth.format_output(RESULTS, PROBLEM) == Synthesizer.DEGENERATE_ANSWER_MESSAGE
    assert synth.trim_counts.get("empty") == 1


def test_adjacent_repeat_needs_a_block_worth_calling_a_loop():
    assert cut_adjacent_repeat("Done. Done.") == ("Done. Done.", None)   # 4 chars
    assert cut_adjacent_repeat(CLEAN + " " + CLEAN)[0] == CLEAN


def test_exit_guard_cuts_a_status_tail_off_a_raw_root_report(capsys):
    """The root-REPORT fallback leaves without the synthesizer's passes."""
    from task_graph import TaskNode

    orch = _orch()
    orch.root_task_id = "root_task_0"
    orch.task_graph.add_task(TaskNode(task_id="root_task_0", description="plan",
                                      agent_id="a1", status=2))
    orch.task_graph.tasks["root_task_0"].result = CLEAN + LOOP + DEPLOY_LOG

    assert orch.terminate() == CLEAN
    out = capsys.readouterr().out
    assert "status-report tail" in out and "FINAL ANSWER GUARDS" in out


def test_exit_guard_is_a_no_op_over_a_synthesis_it_already_cut():
    orch = _orch()
    synthesized = _synth(CLEAN + LOOP + DEPLOY_LOG).format_output(RESULTS, PROBLEM)
    assert orch._guard_final_answer(synthesized) == synthesized
    assert "final_answer_cut" not in orch.colony.verdict_counts


def test_the_ledger_shows_the_new_synthesis_guards(capsys):
    from task_graph import TaskNode

    orch = _orch()
    orch.synthesizer = _synth(CLEAN + LOOP + LOOP + DEPLOY_LOG)
    orch.synthesizer.format_output(RESULTS, PROBLEM)
    orch.root_task_id = "root_task_0"
    orch.task_graph.add_task(TaskNode(task_id="root_task_0", description="plan",
                                      agent_id="a1", status=2))
    orch.task_graph.tasks["root_task_0"].result = CLEAN
    orch.terminate()
    out = capsys.readouterr().out
    assert "synthesis cut at a repeated block  : 1" in out
    assert "synthesis cut at a status report   : 1" in out


def test_format_output_cuts_a_phrase_loop_in_the_final_answer():
    synth = _synth(CLEAN + " Deploy now, deploy now and go, deploy now and go, deploy now and go.")
    assert synth.format_output(RESULTS, PROBLEM) == CLEAN
    assert synth.trim_counts.get("cut") == 1
