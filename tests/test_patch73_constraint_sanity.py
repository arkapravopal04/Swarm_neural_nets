"""PATCH 73: constraint sanity in Problem_Phaser._faithful_constraints.

A constraint over CONSTRAINT_MAX_WORDS words is dropped (and recorded) before
the figure check, and counts against its draw when the greedy and sampled
draws are compared."""
import importlib.machinery
import importlib.util
import sys
import types

if importlib.util.find_spec("sentence_transformers") is None:
    _stub = types.ModuleType("sentence_transformers")
    _stub.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    _stub.SentenceTransformer = object
    sys.modules.setdefault("sentence_transformers", _stub)

from problem_phaser import Problem_Phaser  # noqa: E402

TRIP = ("A school is taking 45 students and 5 teachers on a 3-day trip with a budget "
        "of 180,000 rupees. Choose between a bus that costs 40,000 rupees in total and "
        "a train at 900 rupees per person, plan meals so that each day stays under "
        "15,000 rupees, split the students into groups with one teacher each, and "
        "decide what to do if a student falls ill during the trip.")
GARDEN = ("A community garden has 12 plots and 20 members who want one, decide how to "
          "allocate plots fairly, set a yearly fee so that 9,000 rupees of costs are "
          "covered, split watering duty across the summer months, and decide what to "
          "do with members who stop tending their plot.")
RUN_ON = ("Illness contingency protocol required prior agreement from both parties "
          "when choosing transport mode initially decided upon based purely off number "
          "comparison without regard to health risks implied through transportation "
          "choice alone regardless any other possible mitigating factors presented "
          "later including but certainly limited scope here since such would likely "
          "require additional planning outside this task's")
TRIP_GOOD = ["Total cost must not exceed 180,000 rupees.",
             "Each day's meals must stay under 15,000 rupees.",
             "Each group has exactly one teacher.",
             "A plan for a student falling ill is required."]


def _run(items, request):
    return Problem_Phaser.__new__(Problem_Phaser)._faithful_constraints(
        lambda sampled: list(items), request)


def test_run_on_is_dropped_and_recorded(capsys):
    kept, record = _run(TRIP_GOOD + [RUN_ON], TRIP)
    assert kept == TRIP_GOOD
    assert [d["constraint"] for d in record["dropped"]] == [RUN_ON]
    assert "PATCH 73" in record["dropped"][0]["extra"][0]
    out = capsys.readouterr().out
    assert f"DROPPED {RUN_ON[:80]!r}" in out and "PATCH 73" in out


def test_budget_count_includes_sanity_drops():
    _, record = _run(TRIP_GOOD + [RUN_ON], TRIP)
    assert record["extracted"] == 5 and not record["retried"]


def test_all_run_on_sampled_draw_does_not_beat_greedy():
    greedy = ["Total cost must not exceed 190,000 rupees.",   # figure offender
              "Each group has exactly one teacher."]
    sampled = [RUN_ON, RUN_ON + " again"]
    kept, record = Problem_Phaser.__new__(Problem_Phaser)._faithful_constraints(
        lambda is_sampled: list(sampled if is_sampled else greedy), TRIP)
    assert record["retried"]
    assert "Each group has exactly one teacher." in kept
    assert record["extracted"] == 2


def test_low_overlap_paraphrase_is_kept():
    kept, record = _run(["Daily meal spending must remain below 15,000 rupees."], TRIP)
    assert len(kept) == 1 and record["dropped"] == []


def test_example_style_and_paraphrase_kept():
    para = "Students are split into groups, each led by a teacher."
    kept, record = _run(TRIP_GOOD + [para, "Transport is either the bus or the train."], TRIP)
    assert len(kept) == 6 and record["dropped"] == []


def test_garden_constraints_survive():
    items = ["Yearly fees must cover 9,000 rupees of costs.",
             "Plots must be allocated fairly among members.",
             "Watering duty is shared across the summer months.",
             "Must handle members who stop tending their plot."]
    kept, record = _run(items, GARDEN)
    assert kept == items and record["dropped"] == []


def test_prompt_examples_are_well_under_the_cap():
    examples = ["Must be written in Python.", "Execution time must be under 5 seconds.",
                "Must include vegetarian options for 2 guests."]
    assert max(len(e.split()) for e in examples) * 3 < Problem_Phaser.CONSTRAINT_MAX_WORDS
