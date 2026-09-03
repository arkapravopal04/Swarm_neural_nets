"""
Regression guard for main.py's entry-point imports.

main.py's intra-project imports (ColonyState, TaskGraph, Messenger,
Problem_Phaser, Judge, MemoryStore, Synthesizer, Orchestrator) were
commented out -- upstream they're meant to live in notebook cells instead
(hive_kaggle.ipynb's clone cell uncomments them via sed before `import
main` runs there). Left commented out, `python main.py` / `import main`
is broken: `def build_orchestrator() -> Orchestrator:` evaluates its
return annotation at import time, so a bare `import main` raises
`NameError: name 'Orchestrator' is not defined` before any function is
even called -- nothing short of actually importing the module catches
that.

This is the "whatever passes for CI" here: there's no CI pipeline in
this repo yet, so this pytest suite is it. Run with the project's real
dependencies installed (torch/transformers/sentence-transformers/peft/
faiss -- see hive_kaggle.ipynb cell 4); it's skipped, not failed, if
they're unavailable in the environment running pytest, so a dev sandbox
without GPU libs doesn't get a false failure unrelated to main.py's own
code.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MISSING = [
    dep for dep in ("sentence_transformers", "peft", "faiss")
    if importlib.util.find_spec(dep) is None
]


@pytest.mark.skipif(
    _MISSING,
    reason=f"main.py's real dependencies not installed here: {_MISSING}",
)
def test_main_is_importable():
    """`import main` must not raise -- catches the commented-import regression."""
    import main  # noqa: F401
