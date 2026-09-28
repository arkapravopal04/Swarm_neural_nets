"""
PATCH 19 and PATCH 24 -- what the final answer says was not completed, and
what the synthesis prompt no longer gives the model to recite.

PATCH 19 gave the model the abandoned list and asked it to write a "Not
completed:" sentence. Run 8's answer ended "Nothing else. Leave this part out
if nothing is missing." -- the instruction itself -- and opened with
"[task_e03ea144] 200 rupees/m^3", the results block's labels. PATCH 24 builds
the sentence in code, lets the model write only the answer, and strips task
IDs on the way out.
"""
import re
from types import SimpleNamespace

from synthesizer import Synthesizer
from text_utils import cut_model_not_completed, not_completed_sentence, strip_task_ids


RESULTS = [{"task_id": "task_e03ea144", "description": "Tally the votes per book",
            "result": "Book A 12, Book B 9."}]
PROBLEM = "Choose books using majority vote; assign tasks based on availability."

RUN8_ANSWER = (
    "[task_e03ea144] 200 rupees/m^3; [task_e0e43944] 47 m³/yr; "
    "[task_326a917f] Insufficient data prevents determining an actual fixed "
    "rate per cubic meter.\n\nNOT COMPLETED: Plot assignment constraints, "
    "fixed cost per plot calculation. Nothing else. Leave this part out if "
    "nothing is missing.")


def _capture(answer="Book A wins with 12 votes."):
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt
        return answer
    return Synthesizer(llm_call_fn=llm), seen


def _graph(**tasks):
    return SimpleNamespace(tasks={
        tid: SimpleNamespace(description=d) for tid, d in tasks.items()})


# --- the prompt ----------------------------------------------------------------

SELF_ASSESSMENT_WORDS = [
    r"ground", r"traceab", r"invent", r"introduce facts", r"coherent",
    r"just solved", r"accura", r"correct", r"verif", r"hallucinat",
    r"fabricat", r"made.up",
]
# What run 8 echoed: none of the not-completed machinery reaches the model.
NOT_COMPLETED_INSTRUCTIONS = [r"not completed", r"leave this part out", r"part 2", r"\(nothing\)"]


def test_the_prompt_has_nothing_to_recite_back():
    synth, seen = _capture()
    synth.format_output(RESULTS, PROBLEM, not_completed=["Assign reading slots"])
    instructions = seen["prompt"].replace(PROBLEM, "").replace(
        RESULTS[0]["result"], "").replace(RESULTS[0]["description"], "")
    for pattern in SELF_ASSESSMENT_WORDS + NOT_COMPLETED_INSTRUCTIONS:
        assert not re.search(pattern, instructions, re.IGNORECASE), pattern


def test_the_prompt_carries_no_task_ids():
    synth, seen = _capture()
    synth.format_output(RESULTS, PROBLEM)
    assert "task_e03ea144" not in seen["prompt"]
    assert "- Tally the votes per book\n  Result: Book A 12, Book B 9." in seen["prompt"]


def test_abandoned_parts_are_not_shown_to_the_model():
    synth, seen = _capture()
    synth.format_output(RESULTS, PROBLEM, not_completed=["Assign reading slots"])
    assert "Assign reading slots" not in seen["prompt"]


# --- the sentence, built in code -------------------------------------------------

def test_the_sentence_is_appended_by_code():
    synth, _ = _capture("Book A wins with 12 votes.")
    out = synth.format_output(RESULTS, PROBLEM,
                              not_completed=["Assign reading slots.", "Replace disliked books"])
    assert out == ("Book A wins with 12 votes.\n\n"
                   "Not completed: assign reading slots; replace disliked books.")
    assert synth.trim_counts.get("not_completed_appended") == 1


def test_nothing_abandoned_appends_nothing():
    synth, _ = _capture("Book A wins with 12 votes.")
    assert synth.format_output(RESULTS, PROBLEM) == "Book A wins with 12 votes."
    assert "not_completed_appended" not in synth.trim_counts


def test_run_8s_answer_ships_clean():
    """Its own not-completed list and the echoed instruction are cut, the
    IDs are stripped, and the code-built sentence is the only one left."""
    synth, _ = _capture(RUN8_ANSWER)
    out = synth.format_output(RESULTS, PROBLEM,
                              not_completed=["Determine a fair allocation method for the 12 plots"])
    assert "task_" not in out
    assert "Leave this part out" not in out and "Nothing else" not in out
    assert out.count("ot completed") == 1
    assert out.startswith("200 rupees/m^3; 47 m³/yr; Insufficient data")
    assert out.endswith("Not completed: determine a fair allocation method for the 12 plots.")
    assert synth.trim_counts.get("model_not_completed_cut") == 1
    assert synth.trim_counts.get("task_ids_stripped") == 1


def test_run_threads_the_abandoned_ids_through():
    synth, _ = _capture()
    synth.collect_results = lambda *a, **k: RESULTS
    out = synth.run(None, _graph(t_x="Assign reading slots"), PROBLEM,
                    root_task_id="root", abandoned_ids={"t_x"})
    assert out.endswith("Not completed: assign reading slots.")


# --- the helpers ------------------------------------------------------------------

def test_strip_task_ids():
    assert strip_task_ids("[task_e03ea144] 200 rupees; [task_e0e43944] 47 m3.") == (
        "200 rupees; 47 m3.", 2)
    assert strip_task_ids("root_task_0 finished; task_01ac5781: allocation.") == (
        "finished; allocation.", 2)
    assert strip_task_ids("A task list with no IDs.") == ("A task list with no IDs.", 0)


def test_model_not_completed_is_cut_only_where_it_opens_a_section():
    assert cut_model_not_completed("Answer.\n\n**Not completed:** x.") == ("Answer.", True)
    assert cut_model_not_completed("Answer. Not completed - x.") == ("Answer.", True)
    text = "The payment was not completed in time: a late fee applies."
    assert cut_model_not_completed(text) == (text, False)


def test_not_completed_sentence():
    assert not_completed_sentence([]) == ""
    assert not_completed_sentence([" Assign   slots. ", ""]) == "Not completed: assign slots."


# --- collect_not_completed (PATCH 19, unchanged) ----------------------------------

def test_collects_abandoned_descriptions_in_stable_order():
    synth = Synthesizer(llm_call_fn=lambda p: "")
    graph = _graph(t_b="Replace disliked books", t_a="Assign reading slots")
    assert synth.collect_not_completed(graph, {"t_b", "t_a"}, RESULTS) == [
        "Assign reading slots", "Replace disliked books"]


def test_a_part_that_was_completed_elsewhere_is_not_listed_as_missing():
    synth = Synthesizer(llm_call_fn=lambda p: "")
    graph = _graph(t_abandoned="  tally the VOTES per book ",
                   t_other="Assign reading slots")
    assert synth.collect_not_completed(
        graph, {"t_abandoned", "t_other"}, RESULTS) == ["Assign reading slots"]


def test_duplicate_and_unknown_ids():
    synth = Synthesizer(llm_call_fn=lambda p: "")
    assert synth.collect_not_completed(
        _graph(t1="Assign slots", t2="Assign slots"), {"t1", "t2"}, []) == ["Assign slots"]
    assert synth.collect_not_completed(_graph(), {"gone"}, []) == []
    assert synth.collect_not_completed(_graph(), None, []) == []
