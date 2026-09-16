"""
Regression tests for the two problem-phaser text cleanups:

  * a code fence that survived goal generation (sampled decoding still
    occasionally dresses the goal sentence as a code block, and the token
    cap cuts the block before its closing fence, so the markers arrive
    unbalanced) is stripped before the goal is printed and embedded,
  * an extracted constraint bullet is reduced to the constraint itself,
    dropping the derivation the extractor copied out of an input that
    asked for shown reasoning.

Both were previously only visible by reading the goal line and the
"Constraints you must satisfy" block of a full run log.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_utils import final_derived_constraint, strip_code_fences


def test_strip_code_fences_handles_unbalanced_markers():
    assert strip_code_fences("```python\nGenerate a parser.\n```") == "Generate a parser."
    # Opening fence only -- the generation cap cut the closing one.
    assert strip_code_fences("```\nGenerate a parser.") == "Generate a parser."
    assert strip_code_fences("```python Generate a parser.") == "Generate a parser."
    # Trailing stray backticks, on the sentence's own line.
    assert strip_code_fences("Generate a parser.```") == "Generate a parser."
    assert strip_code_fences("Generate a parser.`") == "Generate a parser."
    # The commonest cap shape: sentence finished, code block opened, cap
    # fired on the fence line itself.
    assert strip_code_fences("Generate a parser. ```python") == "Generate a parser."
    assert strip_code_fences("Generate a parser.\n```python") == "Generate a parser."
    # No info string: the first real word must not be eaten as one.
    assert strip_code_fences("```Generate a parser.```") == "Generate a parser."
    assert strip_code_fences("```python Generate a parser.") == "Generate a parser."


def test_strip_code_fences_leaves_clean_text_alone():
    assert strip_code_fences("Generate a parser for CSV input.") == "Generate a parser for CSV input."
    # A balanced inline code span keeps its closing backtick.
    assert strip_code_fences("Use the `pandas` library.") == "Use the `pandas` library."
    assert strip_code_fences("") == ""
    assert strip_code_fences(None) is None


def test_final_derived_constraint_drops_the_derivation():
    assert final_derived_constraint(
        "The base alloy cannot survive 1400C alone, therefore an internal "
        "cooling scheme is required."
    ) == "An internal cooling scheme is required."
    assert final_derived_constraint(
        "Since the blade sees cyclic loading, fatigue life must be estimated."
    ) == "Fatigue life must be estimated."
    # Chained derivation: only the last link is the constraint.
    assert final_derived_constraint(
        "Cross-region latency is 50-150ms, so quota state is shared, thus "
        "over-admission must be bounded."
    ) == "Over-admission must be bounded."


def test_final_derived_constraint_keeps_bare_constraints():
    assert final_derived_constraint("Must use BeautifulSoup framework.") == "Must use BeautifulSoup framework."
    # A plain noun-phrase bullet has no derivation structure and no modal;
    # it is still a constraint and must survive.
    assert final_derived_constraint("Python 3.11") == "Python 3.11"


def test_final_derived_constraint_drops_pure_derivation():
    # Derivation structure but nothing normative after it -- nothing was
    # actually constrained, so the caller drops the bullet.
    assert final_derived_constraint(
        "The gas path runs above 1400C, therefore the alloy is stressed."
    ) == ""
    assert final_derived_constraint("") == ""


# ------------------------------------------------ code block after the goal

from text_utils import cut_at_code_block, looks_like_model_reasoning


def test_cut_at_code_block_drops_a_block_opened_mid_line():
    # The shape that leaked: sentence, then a fence on the same line, then code.
    assert cut_at_code_block(
        "Generate a parser. ```python\ndef extract(text):\n    pass"
    ) == "Generate a parser."
    assert cut_at_code_block("Generate a parser. ```python def f():") == "Generate a parser."
    assert cut_at_code_block("Generate a parser.\n```python\ndef f(x):") == "Generate a parser."


def test_cut_at_code_block_keeps_a_sentence_inside_the_block():
    assert cut_at_code_block("```python\nGenerate a parser.\n```") == "Generate a parser."
    assert cut_at_code_block("```\n\nGenerate a parser.") == "Generate a parser."
    assert cut_at_code_block("```Generate a parser.```") == "Generate a parser."
    assert cut_at_code_block("Generate a parser for `csv` input.") == "Generate a parser for `csv` input."


# ------------------------------------------- reasoning lines as constraints

def test_reasoning_line_is_not_a_constraint():
    assert looks_like_model_reasoning(
        "Wait, I need to check if there's an explicit constraint on "
        "scheduling... Let me fix this."
    )
    assert looks_like_model_reasoning("Hmm, is the budget fixed?")
    assert looks_like_model_reasoning("Let me re-read the input.")


def test_real_constraints_are_kept():
    for line in [
        "Must be written in Python.",
        "Execution time must be under 5 seconds.",
        "Meetings need to be scheduled within business hours.",
        "Selenium is strictly prohibited.",
        "Must support I/O on Windows.",
        "Python 3.11",
    ]:
        assert not looks_like_model_reasoning(line), line


# --------------------------- reasoning filter: review probes that were dropped

import importlib.util
import types

# _clean_requirements is pure text handling; only the module's top-level
# import needs sentence_transformers. Stub it where it is not installed so
# these run in a sandbox without the model stack instead of being skipped.
if importlib.util.find_spec("sentence_transformers") is None:
    sys.modules.setdefault(
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=object),
    )

from problem_phaser import Problem_Phaser  # noqa: E402

_LONG_CONSTRAINT = (
    "The exported report must include per-region totals, a monthly breakdown "
    "for each of the last twelve months, currency-normalized figures in USD, "
    "a footnote describing the exchange-rate source, and a summary table that "
    "fits on a single printed A4 page without scaling below ten point type."
)


def test_mid_sentence_filler_and_first_person_do_not_drop_a_constraint():
    assert not looks_like_model_reasoning(
        "Latency must be under 100ms, okay, and throughput above 1k rps")
    assert not looks_like_model_reasoning(
        "Output must be valid JSON, i.e. the API I can call accepts it")


def test_numeric_range_ellipsis_is_not_hedging():
    assert not looks_like_model_reasoning("Values in the range 1...10")
    assert not looks_like_model_reasoning("Port must be in 1024 … 65535")
    assert looks_like_model_reasoning("Budget is probably fixed...")


def test_long_constraint_is_kept_unless_it_is_first_person_narration():
    assert len(_LONG_CONSTRAINT.split()) > 40
    assert not looks_like_model_reasoning(_LONG_CONSTRAINT)
    assert looks_like_model_reasoning(
        "Looking at this again my reading is that " + _LONG_CONSTRAINT)


def test_wait_as_a_noun_is_not_self_talk():
    assert not looks_like_model_reasoning("Wait time must be under 2 seconds.")


def test_question_is_still_dropped_but_logged(capsys):
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    kept = phaser._clean_requirements("- Use SI units?\n- Must be written in Python.")
    assert kept == ["Must be written in Python."]
    assert "dropped bullet constraint (phrased as a question): 'Use SI units?'" in capsys.readouterr().out


def test_review_probes_survive_the_bullet_path(capsys):
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    probes = [
        "Latency must be under 100ms, okay, and throughput above 1k rps",
        "Output must be valid JSON, i.e. the API I can call accepts it",
        "Values in the range 1...10",
        _LONG_CONSTRAINT,
    ]
    kept = phaser._clean_requirements("\n".join(f"- {p}" for p in probes))
    for probe in probes:
        assert any(probe in k or k in probe for k in kept), (probe, kept)
    assert "dropped" not in capsys.readouterr().out


def test_comma_split_path_logs_what_it_drops(capsys):
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    kept = phaser._clean_requirements(
        "Must be written in Python, Hmm is the budget fixed, Execution time must be under 5 seconds")
    assert "Must be written in Python" in kept
    assert "Execution time must be under 5 seconds" in kept
    assert not any("Hmm" in k for k in kept)
    out = capsys.readouterr().out
    assert "dropped comma-split constraint (self-correction marker)" in out
    assert "Hmm is the budget fixed" in out


# ------------------------------------------ goal that is code inside a fence

from text_utils import looks_like_source_code


def test_code_body_left_by_fence_stripping_is_detected():
    raw = "```python def extract(text): return 1"
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    assert phaser._clean_goal_text(raw) == "def extract(text): return 1"
    assert looks_like_source_code(phaser._clean_goal_text(raw))


def test_goal_sentences_are_not_mistaken_for_code():
    for goal in [
        "Generate a parser for CSV input.",
        "Implement a function that returns the median of a list.",
        "Compute f(x) = x + 1 for every x in the dataset.",
        "Write a script to import sales data and plot monthly revenue.",
        "Define a class hierarchy for vehicles.",
    ]:
        assert not looks_like_source_code(goal), goal


def test_code_goal_is_resampled(capsys):
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    samples = iter(["```python def extract(text): return 1",
                    "Extract the intent from free-form text."])
    goal = phaser._pick_goal(lambda: next(samples), "please extract intent")
    assert goal == "Extract the intent from free-form text."
    assert "is source code" in capsys.readouterr().out


def test_goal_falls_back_to_user_text_when_every_sample_is_code(capsys):
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    goal = phaser._pick_goal(lambda: "```python def extract(text): return 1",
                             "Write an intent\nextractor for chat logs.")
    assert goal == "Write an intent extractor for chat logs."
    assert "every goal sample was source code" in capsys.readouterr().out
