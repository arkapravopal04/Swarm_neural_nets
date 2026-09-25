"""
PATCH 31. The budget's constraint multiplier counts the constraints the
phaser EXTRACTED (before the Patch 22/30 fidelity filter), not the ones it
KEPT, so a constraint the filter drops no longer also shrinks the budget.

The run 10 replay needs no log: budget 342 with 2 kept constraints back-solves
to domain*gap = 1.245; with the 3 extracted it must come out at 407. It is
reproduced through estimate_complexity itself, with the domain and the goal /
context vectors stubbed so that domain_mult * semantic_gap is the target.
"""
import importlib.machinery
import importlib.util
import sys
import types

import numpy as np
import pytest

if importlib.util.find_spec("sentence_transformers") is None:
    _stub = types.ModuleType("sentence_transformers")
    _stub.__spec__ = importlib.machinery.ModuleSpec("sentence_transformers", None)
    _stub.SentenceTransformer = object
    sys.modules.setdefault("sentence_transformers", _stub)

from problem_phaser import Problem_Phaser  # noqa: E402


def _phaser():
    return Problem_Phaser.__new__(Problem_Phaser)


def _vectors_for_gap(gap):
    """Goal / context vectors whose semantic gap is `gap` (>= 1.0)."""
    sim = 1.0 - (gap - Problem_Phaser.SEMANTIC_BASE) / Problem_Phaser.SEMANTIC_COEF
    return np.array([1.0, 0.0]), np.array([sim, np.sqrt(1.0 - sim * sim)])


def _spec(kept, fidelity="absent", gap=1.0, domain="General Discourse > Stub"):
    goal, context = _vectors_for_gap(gap)
    spec = {
        "requirement": [f"constraint {i}" for i in range(kept)],
        "domain": domain,
        "goal_vector": goal,
        "context_vector": context,
    }
    if fidelity != "absent":
        spec["constraint_fidelity"] = fidelity
    return spec


def _record(extracted):
    return {"dropped": [], "repaired": [], "retried": False, "extracted": extracted}


def test_multiplier_counts_the_extracted_constraints():
    phaser = _phaser()
    wt = phaser._estimate_by_constraints(_spec(2, _record(3)))
    assert wt == pytest.approx(1.0 + 0.5 * np.sqrt(3))
    # Same domain and gap: only the count differs from the kept-count weight.
    assert wt > phaser._estimate_by_constraints(_spec(2))


def test_no_fidelity_key_falls_back_to_kept():
    assert _phaser()._estimate_by_constraints(_spec(2)) == pytest.approx(1.0 + 0.5 * np.sqrt(2))


def test_fidelity_none_falls_back_to_kept():
    assert _phaser()._estimate_by_constraints(_spec(2, None)) == pytest.approx(1.0 + 0.5 * np.sqrt(2))


def test_record_without_extracted_key_falls_back_to_kept():
    record = {"dropped": [], "repaired": [], "retried": False}
    assert _phaser()._estimate_by_constraints(_spec(4, record)) == pytest.approx(2.0)


def test_cap_still_applies_at_large_n():
    assert _phaser()._estimate_by_constraints(_spec(3, _record(50))) == 2.5
    assert _phaser()._estimate_by_constraints(_spec(50)) == 2.5


def test_tier_line_and_breakdown_show_both_counts(capsys):
    spec = _phaser().estimate_complexity(_spec(2, _record(3)))
    assert "constraints=1.87x (3 before fidelity, 2 kept)" in capsys.readouterr().out
    assert spec["complexity_breakdown"]["constraints_extracted"] == 3
    assert spec["complexity_breakdown"]["constraints_kept"] == 2
    # requirement is still the filtered list.
    assert len(spec["requirement"]) == 2


def test_tier_line_without_record_shows_kept_only(capsys):
    spec = _phaser().estimate_complexity(_spec(2))
    out = capsys.readouterr().out
    assert "constraints=1.71x (2 kept)" in out
    assert "before fidelity" not in out
    assert spec["complexity_breakdown"]["constraints_extracted"] is None
    assert spec["complexity_breakdown"]["constraints_kept"] == 2


def test_run10_budget_replay_is_407():
    # Handover figure: domain * gap = 1.245 (General Discourse, 1.0x).
    spec = _phaser().estimate_complexity(_spec(2, _record(3), gap=1.245))
    assert spec["complexity_breakdown"]["domain_multiplier"] == 1.0
    assert spec["complexity_breakdown"]["semantic_gap"] == pytest.approx(1.245, abs=0.005)
    assert spec["complexity_score"] == pytest.approx(2.32, abs=0.005)
    assert spec["colony_budget"] == 407


def test_run10_unrounded_back_solve_brackets_the_replay():
    # 1.245 is a rounded figure: under the kept count it gives 341, not
    # 342. int() truncates, so 342 means a score in [2.126, 2.129), i.e.
    # domain*gap in [1.24538, 1.24714). Across that interval the kept count
    # reproduces 342 and the extracted count gives 408 (407 only in a
    # sliver at the bottom edge, which is where 1.245 sits).
    wt2 = 1.0 + 0.5 * np.sqrt(2)
    for score in (2.1265, 2.127, 2.128):
        gap = score / wt2
        assert _phaser().estimate_complexity(_spec(2, gap=gap))["colony_budget"] == 342
        assert _phaser().estimate_complexity(_spec(2, _record(3), gap=gap))["colony_budget"] == 408
