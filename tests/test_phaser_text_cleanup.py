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
