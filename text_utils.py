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
import difflib


def normalize_identifier(raw, candidates, cutoff: float = 0.75, strip_chars: str = "_*-/."):
    """
    Best-effort match of a possibly mistyped/mis-formatted LLM-generated
    identifier (an action token, tool name, role, dict key, dependency
    label, ...) against a known set of valid ones.

    Pipeline, applied identically to `raw` AND to every candidate (so
    "normalize both sides" holds even when a candidate itself came from
    another noisy LLM payload rather than a hardcoded constant):
        1. strip ALL whitespace, not just leading/trailing -- catches a
           decode artifact like "rol e" -> "role".
        2. strip leading/trailing characters in `strip_chars` -- the
           dashes/asterisks/bullets/underscores a model tends to wrap an
           identifier in (e.g. "__run_code__" -> "run_code"). Pass
           strip_chars="" for identifiers where such characters are
           semantically part of the value rather than decoration -- a
           dependency label like "__coating_thickness__" would collapse
           into a different, wrong label if its underscores were
           stripped instead of compared as-is.
        3. casefold.
        4. exact match against the cleaned candidates.
        5. else difflib.get_close_matches (n=1) as a fuzzy fallback.

    Returns None on no match (including empty/non-string raw, or an
    empty candidates collection) so the caller keeps its own fallback --
    this never silently guesses past the cutoff. Logs whenever step 5 is
    what actually produced the match (an exact hit after cleanup doesn't
    log -- that's normal formatting noise, not a typo worth tracking),
    so the fuzzy-match rate is visible in the log.
    """
    if not raw or not isinstance(raw, str) or not candidates:
        return None

    def _clean(s):
        s = re.sub(r"\s+", "", s)
        if strip_chars:
            s = s.strip(strip_chars)
        return s.casefold()

    cleaned_raw = _clean(raw)
    if not cleaned_raw:
        return None

    folded_map = {}
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        folded_map[_clean(candidate)] = candidate

    if cleaned_raw in folded_map:
        return folded_map[cleaned_raw]

    close = difflib.get_close_matches(cleaned_raw, list(folded_map.keys()), n=1, cutoff=cutoff)
    if close:
        matched = folded_map[close[0]]
        print(f"[normalize_identifier] fuzzy match: {raw!r} -> {matched!r} (cutoff={cutoff})")
        return matched

    return None


def _join_within_budget(sentences, max_chars):
    """
    Joins sentences with a single space, keeping only as many WHOLE
    sentences as fit within max_chars -- never slices into the middle of
    the last one.

    The previous approach (in both callers below) joined everything
    first, then hard-sliced the joined string at max_chars and backed up
    to the nearest space. That still routinely landed mid-sentence (e.g.
    a goal cut off at "...Minimize" with the rest of the clause gone),
    and the leftover fragment reads as a near-duplicate of whatever
    sentence it was chopped out of -- confirmed against a real goal
    string. Sentences are already split out for the dedupe pass in each
    caller; reusing that boundary here instead of re-deriving one from
    raw character offsets is what actually fixes it.

    A single sentence longer than max_chars on its own is kept whole
    rather than dropped or truncated -- an over-budget complete thought
    is still better than a truncated fragment, and max_chars is a soft
    target everywhere it's used here, not a hard wire limit.
    """
    if not sentences:
        return ""
    kept = [sentences[0]]
    total = len(sentences[0])
    for s in sentences[1:]:
        if total + 1 + len(s) > max_chars:
            break
        kept.append(s)
        total += 1 + len(s)
    result = " ".join(kept)
    if len(kept) < len(sentences):
        result += "..."
    return result


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
    return _join_within_budget(deduped, max_chars)


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
    return _join_within_budget(deduped, max_chars)


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^\s>]{0,40}\|>")
_SCAFFOLD_LINE_RE = re.compile(
    r"^[ \t]*(?:ACTION|PAYLOAD)\s*:.*$"
    r"|^[ \t]*Your next action\s*:?.*$"
    r"|^[ \t]*Available actions\s*:?.*$"
    r"|^[ \t]*Your output must be.*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_special_tokens(text):
    """
    Removes any literal "<|...|>" special-token marker from text.

    Guards two distinct leak paths into a stored transcript: a decode
    call that (unlike a sibling call elsewhere that passes
    skip_special_tokens=True) leaves special tokens like <|endoftext|>
    in the decoded string, and a model that has learned to emit a
    special token's string form as ordinary text even when the decoder
    would have suppressed the real token. Matches any <|...|> shape,
    not just tokens seen so far, since a future tokenizer/adapter can
    define different ones.
    """
    if not text:
        return text
    return _SPECIAL_TOKEN_RE.sub("", text)


def strip_scaffolding_lines(text):
    """
    Drops lines that are re-emitted harness formatting -- an "ACTION:"
    or "PAYLOAD:" label, "Your next action", "Available actions", or
    "Your output must be..." -- rather than actual reasoning.

    Free-form reasoning generation is meant to produce prose, but its
    own prompt (and, across a multi-cycle KV-cache continuation, its
    prior output) contains that formatting as instructional text, and
    unconstrained decoding can echo it back verbatim instead of
    reasoning in prose. A stored thought/reasoning transcript should
    only ever contain the latter.
    """
    if not text:
        return text
    return _SCAFFOLD_LINE_RE.sub("", text)


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


def trim_to_sentences(text, max_sentences: int = 3, max_words: int = 60, marker: bool = True):
    """
    Hard-trims free text to the first few whole sentences.

    Belt-and-braces companion to the REPORT generation budget in
    _ActionPayloadStop: even with generation capped, a REPORT payload has
    no closing token, so what arrives at the judge can still be several
    paragraphs of restatement wrapped around a one-line answer. The judge
    reads the whole thing and rejects it for padding. Trimming to the
    first 2-3 sentences before judging keeps the part that actually
    answers the subtask ("list the waste types" -> the list) and drops
    the trailing self-commentary.

    Two independent bounds, whichever bites first: max_sentences, and
    max_words across the kept sentences. The word bound exists because
    "three sentences" is not a length -- three run-on sentences are
    exactly the output this is meant to catch.

    Never slices mid-sentence: if the first sentence alone already blows
    the word budget it is kept whole, same soft-target contract as
    _join_within_budget.

    marker=False suppresses the trailing "[...]". Callers trimming text
    that a JUDGE will read should pass it: an explicit truncation mark
    is honest about what happened to the string, but it also tells the
    judge the answer is cut off, which is the opposite of the signal a
    trim is trying to produce. Keep the marker where a human or a later
    agent reads the string and needs to know something was dropped.
    """
    if not text:
        return text
    stripped = str(text).strip()
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', stripped) if s.strip()]
    if not sentences:
        return stripped

    kept = []
    words = 0
    for s in sentences[:max_sentences]:
        s_words = len(s.split())
        if kept and words + s_words > max_words:
            break
        kept.append(s)
        words += s_words

    if not kept:
        kept = [sentences[0]]

    result = " ".join(kept).strip()
    if marker and len(kept) < len(sentences):
        result += " [...]"
    return result


def first_clause(text, max_chars: int = 120):
    """
    Reduces a block of critique prose to one short clause.

    The judge's tier-3 critique is written as flowing prose, and it used
    to be pasted into the next agent's prompt nearly whole. At that
    length and in that register it is indistinguishable from the agent's
    own "Previous Thoughts" section sitting a few lines below it -- so
    the model does not read it as a verdict to act on, it reads it as
    reasoning it was already producing and simply continues it. Cutting
    it to a single clause removes enough prose texture that it can only
    be read as a label.
    """
    if not text:
        return ""
    flat = re.sub(r"\s+", " ", str(text)).strip()
    if not flat:
        return ""
    # First sentence, then first clause within it.
    first = re.split(r'(?<=[.!?])\s', flat, maxsplit=1)[0].strip()
    if len(first) > max_chars:
        clause = re.split(r'\s[-—;:]\s|,\s', first, maxsplit=1)[0].strip()
        first = clause if clause else first
    if len(first) > max_chars:
        first = first[:max_chars].rsplit(" ", 1)[0].rstrip(",;:-") + "..."
    return first.rstrip()
