"""
text_utils.py

Shared text-cleanup helpers for degenerate LLM output -- greedy decoding
occasionally locks onto a conclusion and repeats it verbatim, or loops
through short filler sentences once it runs out of real content.

agent_node.py, judge.py, and ghost_extractor.py each independently grew
a byte-identical dedupe_and_cap implementation (see their docstrings --
each argued for staying self-contained rather than importing another
file's copy). That discipline is reasonable per-file, but it's also
exactly how the three copies drifted into existing at all. Centralizing
the logic here -- while leaving max_chars as a per-call-site parameter --
keeps each caller's own tuning (300 for in-run fail_reason, 500 for a
persistent ghost record, 400 for a problem-spec goal) without requiring
every new caller to re-derive the fix from scratch.

Deliberately depends on nothing but re -- no torch, no tools, no
colony_state -- so problem_phaser.py (which must not drag in agent_node's
transitive dependencies just to clean a string) can import it too.
"""

import re


def dedupe_and_cap(text, max_chars: int = 500):
    """
    Collapses consecutive repeated sentences and caps overall length.

    Catches a degenerate generation that loops the same sentence
    back-to-back (e.g. "The task is impossible." x5) -- a known failure
    mode of greedy decoding once a model locks onto a conclusion. Only
    collapses ADJACENT duplicates; a repeat separated by other content
    survives. See dedupe_global_and_cap for the non-adjacent case.
    """
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    deduped = []
    for s in sentences:
        if not deduped or s.strip() != deduped[-1].strip():
            deduped.append(s)
    result = " ".join(deduped)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0] + "..."
    return result


def dedupe_global_and_cap(text, max_chars: int = 400):
    """
    Collapses ANY repeated sentence (first occurrence wins), not just
    consecutive ones, then caps overall length.

    dedupe_and_cap only catches immediate back-to-back repeats -- that
    matches the failure shapes it was built for (a DIE reason or a
    critique looping one sentence in place), but a degenerate short-form
    extraction (e.g. a one-sentence goal) can interleave a repeated
    clause with other generated content, which an adjacent-only pass
    would miss entirely. Case-insensitive comparison so trivial
    capitalization drift doesn't defeat the dedupe.
    """
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen = set()
    deduped = []
    for s in sentences:
        key = s.strip().lower()
        if not key or key not in seen:
            if key:
                seen.add(key)
            deduped.append(s)
    result = " ".join(deduped)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0] + "..."
    return result


def dedupe_list_exact(items):
    """
    Removes exact (case/whitespace-insensitive) duplicate entries from a
    list, preserving first-occurrence order. List-level analogue of the
    sentence-level dedupe above -- for cases like a requirements list
    where the LLM repeats an entire bullet verbatim rather than a
    sentence within one string.
    """
    seen = set()
    deduped = []
    for item in items:
        key = item.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped
