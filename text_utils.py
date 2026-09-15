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


# A Markdown fence marker, and the small set of info strings worth
# treating as one. The info string is matched against a list rather than
# as \w+ because both "```python Generate a parser." (info string) and
# "```Generate a parser." (no info string) occur, and a greedy \w+ cannot
# tell them apart -- it ate the first real word of the goal.
_FENCE = r"(?:`{3,}|~{3,})"
_FENCE_INFO = (
    r"(?:python|py|json|markdown|md|text|txt|plaintext|bash|sh|shell|yaml|yml|"
    r"xml|html|latex|tex|sql|cpp|c|java|javascript|js|typescript|ts|go|rust|"
    r"r|matlab|pseudocode|code)"
)
# A line that is nothing but a fence, with or without an info string. Any
# word is allowed here -- alone on its line there is no real text to eat.
# Takes its newline with it, or stripping leaves a blank line in the
# middle of what should be one sentence.
_FENCE_LINE_RE = re.compile(
    rf"^[ \t]*{_FENCE}[ \t]*\w*[ \t]*$\r?\n?", re.MULTILINE
)
# An opening fence welded onto the front of the text...
_LEADING_FENCE_RE = re.compile(
    rf"^[ \t]*{_FENCE}[ \t]*(?:{_FENCE_INFO}\b[ \t]*)?", re.IGNORECASE
)
# ...and one welded onto the END of it. This is the shape the generation
# cap produces most often: the model finishes the sentence, opens a code
# block to elaborate, and the cap fires on the fence line itself.
_TRAILING_FENCE_RE = re.compile(
    rf"[ \t]*{_FENCE}[ \t]*(?:{_FENCE_INFO}\b[ \t]*)?$", re.IGNORECASE
)
# A short dangling run (one or two backticks) -- only ever stripped when
# the text's backticks are unbalanced, so an inline `code span` at the
# end of a sentence keeps its closing tick.
_DANGLING_TICKS_RE = re.compile(r"[ \t]*`{1,2}[ \t]*$")


def strip_code_fences(text):
    """
    Removes Markdown code-fence markers -- whole fence lines, an opening
    fence welded to the front or the end of the text, and an unbalanced
    dangling backtick -- while leaving the text between them alone.

    Written for the problem phaser's goal extraction, which is sampled
    rather than greedy precisely because greedy decoding on a code-shaped
    user input continues the input as code. Sampling stops it emitting a
    whole function body, but it still occasionally dresses the one
    sentence it does produce as a code block, and the generation cap then
    cuts the block before its closing fence -- so the fences arrive
    unbalanced and a symmetric ```...``` match would not catch them. The
    goal string is the root TaskNode.description and (via goal_vector)
    the tier-2 similarity target every child is scored against, so a
    stray fence is both printed to the user and embedded into the target.
    """
    if not text:
        return text
    text = _FENCE_LINE_RE.sub("", text)
    text = _LEADING_FENCE_RE.sub("", text)
    text = _TRAILING_FENCE_RE.sub("", text)
    if text.count("`") % 2 == 1:
        text = _DANGLING_TICKS_RE.sub("", text)
    return text.strip()


# Connectives that separate the reasoning from the constraint it
# produced. Everything before the LAST one is derivation; what follows is
# the constraint itself.
_DERIVATION_JOIN_RE = re.compile(
    r",?\s*(?:therefore|thus|hence|so|meaning|which means|implying that|"
    r"so the constraint is|the constraint is|this means(?: that)?|"
    r"=>|->)\s*[:,]?\s+",
    re.IGNORECASE,
)
# A leading subordinate clause -- "Since the alloy melts at 1400C, the
# blade must be cooled" -- whose main clause is the actual constraint.
_DERIVATION_PREFIX_RE = re.compile(
    r"^(?:since|because|as|given(?: that)?|seeing(?: that)?|"
    r"in order to|to satisfy|due to|owing to)\b[^,]{0,200},\s*",
    re.IGNORECASE,
)
# Normative vocabulary. A line with none of it is a statement about the
# problem, not a bound on the solution.
_NORMATIVE_RE = re.compile(
    r"\b(?:must|shall|should|needs? to|has to|have to|required?|requires?|"
    r"cannot|can't|may not|must not|no more than|at least|at most|under|"
    r"below|above|within|limited to|prohibited|forbidden|avoid|only|"
    r"minimum|maximum|target|budget|not exceed|exceed)\b",
    re.IGNORECASE,
)


def final_derived_constraint(line):
    """
    Reduces one extracted constraint bullet to the constraint itself,
    dropping the reasoning that produced it.

    The phaser's constraint prompt asks for bare bullets, but the test
    prompts it runs on end with "Show reasoning and calculations, not
    just a final answer" -- and the extractor, reading the same text,
    writes the derivation out alongside the bound: "The base alloy cannot
    survive 1400C alone, therefore an internal cooling scheme is
    required." Those strings are not just displayed; each is embedded
    into requirement_vectors, counted by _estimate_by_constraints (so
    derivation prose inflates num_reqs and the budget), and threaded down
    the task graph into every child agent's "Constraints you must
    satisfy" block, where a sentence of reasoning reads as context to
    continue rather than as a bound to meet.

    Returns "" for a line that is pure derivation with no bound in it,
    for the caller to drop.
    """
    if not line:
        return ""
    flat = re.sub(r"\s+", " ", str(line)).strip()
    if not flat:
        return ""

    # Take the text after the LAST derivation connective: a bullet can
    # chain them ("X, so Y, therefore Z") and only the tail is the bound.
    tail = flat
    while True:
        match = None
        for candidate in _DERIVATION_JOIN_RE.finditer(tail):
            match = candidate
        if not match or not tail[match.end():].strip():
            break
        tail = tail[match.end():].strip()

    # Then peel a leading "Since ..., " style subordinate clause.
    peeled = _DERIVATION_PREFIX_RE.sub("", tail).strip()
    if peeled:
        tail = peeled

    if not _NORMATIVE_RE.search(tail):
        # Nothing was derived here -- it is a restatement of the problem.
        # Keep it only if the original had no derivation structure at all,
        # since a plain noun-phrase bullet ("Python 3.11") is still a
        # constraint.
        if tail != flat:
            return ""

    return tail[:1].upper() + tail[1:]


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


# A sentence terminator, plus any closing quotes/brackets that belong with
# it -- so 'he said "stop."' walks back to the quote, not to the period.
_SENTENCE_END_RE = re.compile(r'[.!?]["\'\)\]]*')

def drop_incomplete_tail(text):
    """
    Drops a trailing partial sentence, keeping everything up to and
    including the last '.', '!' or '?'.

    This is what a REPORT generation budget produces: the cap fires
    mid-sentence, because a free-text REPORT has no closing token to wait
    for (see _ActionPayloadStop.REPORT_MAX_NEW_TOKENS). A hard cut leaves a
    stump -- "...and the timing guide should allow roughly fifteen min" --
    which reads to the judge as an answer that gave up halfway. Walking
    back to the last complete sentence turns the same generation into a
    shorter but finished answer.

    Runs BEFORE trim_to_sentences rather than replacing it: this decides
    where the text legitimately ends, trim_to_sentences then decides how
    much of it the judge should read.

    Returns the text unchanged when it already ends on a terminator, and
    when it contains no terminator at all -- notably a REPORT that is one
    unfinished throat-clearing clause, or a bare bullet list whose last
    item was cut. Leaving those intact is deliberate: an answer that never
    completed a single sentence is exactly what the judge's empty-answer
    checks exist to catch, and silently emptying it here would hide that.

    Deliberately unguarded against the case where the only complete
    sentence IS the preamble ("The guide is structured as follows. Fifteen
    minutes per rou" -> the preamble alone). Suppressing the walk-back
    there would just hand the judge the stump instead, which it rejects
    anyway; the fix for that shape is the REPORT prompt line telling the
    model to lead with the answer, not a heuristic here.
    """
    if not text:
        return text
    stripped = str(text).rstrip()
    if not stripped:
        return text

    last_end = None
    for match in _SENTENCE_END_RE.finditer(stripped):
        last_end = match.end()

    if last_end is None or last_end == len(stripped):
        return text
    return stripped[:last_end]


# ---------------------------------------------------------------------------
# Closer-cycling degeneration
#
# A third repetition shape, distinct from the two the dedupe helpers above
# handle and invisible to both. Observed verbatim at the end of a REPORT:
#
#   "...Ready. Finalized. Deploying. Deployment. Done. Ready for review.
#    Submitted. Confirmed. Completed."
#
# No SENTENCE repeats, so dedupe_and_cap and dedupe_global_and_cap both
# pass it through untouched. No TOKEN repeats either, so a token-level
# repetition_penalty has nothing to bite on. The model is cycling
# synonyms for "finished": semantically stuck while lexically novel at
# every step, which is precisely the blind spot between those two
# mechanisms.
#
# DETECTION IS DELIBERATELY NARROW. The first attempt here used prosody
# alone -- a run of N consecutive sub-four-word sentences -- on the theory
# that real prose does not do that. It does: "Here are the steps. 1. Buy
# milk. 2. Buy eggs. 3. Buy flour." splits into exactly that shape (the
# enumerators become their own one-word sentences), and the tail-trimmer
# below cut a real shopping list down to "Here are the steps." A short
# list and a closer-cycle are genuinely indistinguishable by length, so
# length is used only as a gate, and the actual decision is made against
# a vocabulary of completion/status stems.
#
# The cost of that choice, stated plainly: this catches the COMPLETION
# flavour of synonym-cycling and nothing else. A model that cycles
# synonyms in some other register ("Furthermore. Moreover. Additionally.")
# passes straight through. Extending _CLOSER_STEMS is how that gets
# handled; a general semantic-redundancy check needs embeddings, which
# this module deliberately does not depend on (see the module docstring --
# re only). Precision was chosen over recall on purpose: a false negative
# leaves filler in the output, a false positive silently deletes a real
# answer.
# ---------------------------------------------------------------------------

CLOSER_MIN_RUN = 4    # consecutive closer sentences before it counts
CLOSER_MAX_WORDS = 3  # a closer sentence is short; longer ones are real prose

# Matched as PREFIXES, so one entry covers a word's inflections
# ("deploy" -> deploying / deployed / deployment). Over-broad matches are
# possible in principle ("end" also prefixes "endorse") but are gated by
# the all-content-words rule and the length limit below, which together
# require the ENTIRE sentence to be nothing but closers.
_CLOSER_STEMS = (
    "done", "ready", "complet", "finish", "final", "submit", "confirm",
    "deploy", "verif", "approv", "review", "acknowledg", "understood",
    "noted", "ok", "yes", "end", "clos", "sent", "deliver", "ship",
    "resolv", "success", "pending", "process", "execut", "initiat",
    "start", "launch", "accept", "valid", "check", "sign",
)

# Function words that carry no content, so they neither make a sentence a
# closer nor disqualify it ("Ready for review" is still a pure closer).
_CLOSER_STOPWORDS = frozenset((
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "for", "of", "in", "on", "at", "as", "and", "or", "it", "its",
    "this", "that", "these", "those", "all", "we", "i", "you", "now",
    "has", "have", "had", "will", "would", "can", "not", "no", "up",
))

_WORD_CLEAN_RE = re.compile(r"[^a-z]")


def _sentence_list(text):
    """Sentences as split everywhere else in this module, blanks dropped."""
    return [s for s in re.split(r'(?<=[.!?])\s+', str(text).strip()) if s.strip()]


def _is_closer_sentence(sentence, max_words=CLOSER_MAX_WORDS):
    """True when a sentence is short AND made of nothing but completion words.

    Both halves are required. "Confirmed." qualifies; "Buy milk."  fails on
    vocabulary (buy/milk are not closers); "The migration completed
    successfully after all four services restarted." fails on length even
    though it is about completion -- it is a real sentence carrying real
    information, and length is the cheapest signal that says so.
    """
    words = [w for w in (_WORD_CLEAN_RE.sub("", w.lower()) for w in sentence.split()) if w]
    if not words or len(words) > max_words:
        return False
    content = [w for w in words if w not in _CLOSER_STOPWORDS]
    if not content:
        return False
    return all(w.startswith(_CLOSER_STEMS) for w in content)


def has_closer_run(text, min_run=CLOSER_MIN_RUN, max_words=CLOSER_MAX_WORDS):
    """True when `text` contains a run of `min_run` consecutive closer sentences.

    Used as a degeneracy SIGNAL (see Agent._looks_degenerate), so it looks
    anywhere rather than only at the tail: during streaming generation the
    run is at the tail by construction, and after the fact a run buried
    mid-answer is just as much evidence that the generation collapsed.
    """
    if not text:
        return False
    run = 0
    for s in _sentence_list(text):
        run = run + 1 if _is_closer_sentence(s, max_words) else 0
        if run >= min_run:
            return True
    return False


def trim_closer_tail(text, min_run=CLOSER_MIN_RUN, max_words=CLOSER_MAX_WORDS):
    """Drop a TRAILING run of closer sentences.

    Trimming is restricted to the tail even though detection is not: a run
    buried mid-text has real content after it that cutting there would
    destroy, whereas a run reaching the end of the text has nothing after
    it to preserve. Returns the text unchanged when the trailing run is
    shorter than min_run, and when trimming would leave nothing behind --
    an answer that is ENTIRELY closers is a failure the judge's own
    emptiness checks should see, not something to silently blank out here
    (same reasoning as drop_incomplete_tail).
    """
    if not text:
        return text
    sentences = _sentence_list(text)
    if not sentences:
        return text

    cut = len(sentences)
    while cut > 0 and _is_closer_sentence(sentences[cut - 1], max_words):
        cut -= 1

    if len(sentences) - cut < min_run or cut == 0:
        return text
    return " ".join(sentences[:cut])
