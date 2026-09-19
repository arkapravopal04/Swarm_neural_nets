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


def _join_within_budget(sentences, max_chars, separators=None):
    """
    Joins sentences, keeping only as many WHOLE sentences as fit within
    max_chars -- never slices into the middle of the last one.

    `separators[i]` is the whitespace that followed sentences[i] in the
    original text. Passed, it is put back verbatim, so a bulleted or
    multi-line answer keeps its line breaks and indentation; omitted,
    sentences are joined with a single space, which flattens the text to
    one paragraph. Flattening is the right default for everything that
    goes back into a prompt as one line -- a goal, a fail_reason, a DIE
    line -- and wrong only for text a person reads as a document. See
    dedupe_global_and_cap's keep_line_breaks.

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
    joins = []
    for i, s in enumerate(sentences[1:], start=1):
        # What separated this sentence from the one before it. Budgeted at
        # its real length rather than at 1, so a cap means the same number
        # of characters whether or not the layout was kept.
        sep = separators[i - 1] if separators else " "
        if total + len(sep) + len(s) > max_chars:
            break
        joins.append(sep)
        kept.append(s)
        total += len(sep) + len(s)
    result = kept[0] + "".join(sep + s for sep, s in zip(joins, kept[1:]))
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


def _split_keeping_separators(text):
    """[(sentence, the whitespace that followed it), ...].

    The same boundary every other split in this module uses, with the
    whitespace kept instead of discarded -- which is the whole of what
    separates a bulleted answer from the same answer flattened into one
    paragraph. The last sentence's separator is "".
    """
    parts = re.split(r'(?<=[.!?])(\s+)', text.strip())
    return [(parts[i], parts[i + 1] if i + 1 < len(parts) else "")
            for i in range(0, len(parts), 2)]


def dedupe_global_and_cap(text, max_chars: int = 400, keep_line_breaks: bool = False):
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

    keep_line_breaks puts each sentence back behind the whitespace that
    originally followed the one before it, instead of behind a single
    space. Off by default and deliberately not the only behaviour: most
    callers here are cleaning a string that goes back into a prompt as
    one line, where a preserved newline is noise. It exists for the one
    caller whose output is a document somebody reads -- the synthesizer's
    final answer, which was arriving as a flattened paragraph however
    carefully the decode had laid it out in bullets. Nothing upstream put
    those breaks back, because nothing upstream had them any more.
    """
    if not text:
        return text
    pieces = _split_keeping_separators(text)
    seen = set()
    deduped = []
    for sentence, separator in pieces:
        key = sentence.strip().lower()
        if not key or key not in seen:
            if key:
                seen.add(key)
            deduped.append((sentence, separator))
    return _join_within_budget(
        [s for s, _ in deduped], max_chars,
        # The separator that followed each KEPT sentence, so a dropped
        # duplicate takes its own trailing whitespace with it rather than
        # leaving a gap where it used to be.
        separators=[sep for _, sep in deduped] if keep_line_breaks else None,
    )


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


_OPENING_FENCE_RE = re.compile(
    rf"{_FENCE}[ \t]*(?:{_FENCE_INFO}\b)?", re.IGNORECASE
)
_ANY_FENCE_RE = re.compile(_FENCE)


def cut_at_code_block(text):
    """
    Keeps only the prose of a one-sentence generation, discarding any code
    block the model opened alongside it.

    strip_code_fences removes fence MARKERS at line/text edges, and so
    misses the shape that actually leaked into the goal string: the model
    finishes its sentence and opens a block on the same line --
    "Generate a parser. ```python def extract(text): ..." -- with a
    newline and a code body after. The fence is mid-line and not at the
    end of the text, so no edge pattern matches; both the fence and the
    code behind it survived into the root task description and goal_vector.

    If the text OPENS with a fence, the sentence is inside the block: the
    opener (and its info string) is dropped and the text is cut at the next
    fence. Otherwise it is cut at the first fence. Run strip_code_fences
    afterwards for any leftover edge markers.
    """
    if not text:
        return text
    stripped = text.lstrip()
    opener = _OPENING_FENCE_RE.match(stripped)
    if opener:
        stripped = stripped[opener.end():]
    closer = _ANY_FENCE_RE.search(stripped)
    if closer:
        stripped = stripped[:closer.start()]
    return stripped.strip()


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


# Self-correction markers that never occur in a constraint, wherever they
# sit in the line. "wait" only counts as an interjection ("Wait," / "wait...")
# so "Wait time must be under 2 s" is not self-talk.
_STRONG_REASONING_RE = re.compile(
    r"(?:^|[^\w])(?:wait\s*(?:[,!]|\.{2,}|…)|hmm+\b|let me\b|on second thought\b|"
    r"fix this\b|we need to check\b)",
    re.IGNORECASE,
)
# First-person / filler openers. These DO occur mid-sentence in real
# constraints ("..., okay, and throughput above 1k rps", "the API I can
# call"), so they only count when they open a clause: start of the line or
# right after a sentence terminator.
_CLAUSE_OPENER_REASONING_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:let's\b|i need to\b|i should\b|i think\b|i'll\b|"
    r"i will\b|i'm\b|i am\b|i must\b|i can\b|actually,|okay,|ok,)",
    re.IGNORECASE,
)
# First-person anywhere in the line: not enough on its own, but together
# with excessive length it marks narrated reasoning rather than a clause.
_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|i'll|i've|me|my|let's)\b", re.IGNORECASE)
# A numeric range written with an ellipsis ("1...10", "0 … 255").
_NUMERIC_RANGE_RE = re.compile(r"\d\s*(?:\.{3}|…)\s*[-+]?\d")
REASONING_MAX_WORDS = 40


def reasoning_reason(line):
    """
    Why an extracted "constraint" looks like the model thinking out loud,
    or None if it reads as a constraint. See looks_like_model_reasoning.
    """
    if not line:
        return None
    flat = re.sub(r"\s+", " ", str(line)).strip()
    if _STRONG_REASONING_RE.search(flat):
        return "self-correction marker"
    if _CLAUSE_OPENER_REASONING_RE.search(flat):
        return "first-person/filler clause opener"
    if flat.endswith("?"):
        return "phrased as a question"
    if "..." in _NUMERIC_RANGE_RE.sub("", flat).replace("…", "..."):
        return "trailing-off ellipsis"
    if len(flat.split()) > REASONING_MAX_WORDS and _FIRST_PERSON_RE.search(flat):
        return f"over {REASONING_MAX_WORDS} words of first-person narration"
    return None


def looks_like_model_reasoning(line):
    """
    True when an extracted "constraint" is really a line of the model
    thinking out loud, not a constraint.

    Observed: "Wait, I need to check if there's an explicit constraint on
    scheduling... Let me fix this." came out of the phaser's extractor as a
    bullet, survived final_derived_constraint (it contains "need to", which
    reads as normative), and was threaded into child agents' "Constraints
    you must satisfy" block -- where it could never be satisfied and cost
    the run a hard task abandonment.

    Rejected: self-correction markers anywhere; first-person or filler
    words only where they OPEN a clause (mid-sentence "okay," / "I can" is
    ordinary constraint prose); a question; an ellipsis that is not a
    numeric range; and long lines only when they are also first-person --
    length alone never drops a constraint. Callers log every rejection
    (reasoning_reason gives the why), so a wrongly dropped constraint is
    visible in the run log.
    """
    return reasoning_reason(line) is not None


# Lines that can only be source code: a definition/import statement.
_CODE_STATEMENT_RE = re.compile(
    r"^[ \t]*(?:def\s+\w+\s*\(|class\s+\w+\s*[:(]|import\s+[\w.]+\s*$|"
    r"from\s+[\w.]+\s+import\s+\w|#include\s*<|function\s+\w+\s*\(|"
    r"(?:const|let|var)\s+\w+\s*=)",
    re.MULTILINE,
)
# Weaker signals: each can show up in a goal sentence on its own ("compute
# f(x) = x + 1"), so it takes two of them.
_CODE_SIGNAL_RES = (
    re.compile(r"\breturn\s+[\w\[\(\{\"'-]"),        # return <expr>
    re.compile(r"\w\([^()]*\)\s*:"),                  # f(x):
    re.compile(r"[=!<>]=|:=|->|=>"),                  # comparison / arrows
    re.compile(r";\s*$", re.MULTILINE),               # statement terminator
    re.compile(r"[{}]"),                              # braces
    re.compile(r"\bself\.\w"),                        # attribute access on self
    re.compile(r"^(?: {4}|\t)\S", re.MULTILINE),      # indented body line
    re.compile(r"\b\w+\s*=\s*[\w\[\{\(\"']"),         # assignment
)


def looks_like_source_code(text):
    """
    True when a generated goal "sentence" is actually code.

    cut_at_code_block/strip_code_fences remove fence MARKERS; when the model
    put code (not prose) inside the fence -- "```python def extract(text):
    return 1" -- stripping leaves the code itself as the root task
    description and goal_vector target. A definition/import statement is
    decisive; otherwise two independent code signals are required, so a
    goal that merely mentions "f(x) = x + 1" is not rejected.
    """
    if not text:
        return False
    if _CODE_STATEMENT_RE.search(text):
        return True
    return sum(1 for r in _CODE_SIGNAL_RES if r.search(text)) >= 2


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^\s>]{0,40}\|>")
# Split into its two halves so degeneracy_cut can take one without the
# other, and so neither is written down twice.
#
# The LABEL half is the ambiguous one. Under IGNORECASE, "Action: Book the
# venue by Friday." -- an ordinary labelled line in a finished plan -- is
# indistinguishable from the harness's own "ACTION:" protocol line. That is
# harmless where this pattern started: strip_scaffolding_lines REMOVES the
# matching line and keeps everything around it, so a false positive costs
# one line. It is not harmless in degeneracy_cut, which TRUNCATES from the
# match onward -- there a false positive costs the whole rest of the
# answer, and a plan written with "Action:"/"Payload:" labels lost
# everything after its first bullet. degeneracy_cut uses
# _PROTOCOL_SHOUT_RE instead, which is case-SENSITIVE for exactly this
# reason.
_SCAFFOLD_LABEL_PATTERN = r"^[ \t]*(?:ACTION|PAYLOAD)\s*:.*$"
# The PHRASE half is unambiguous either way: these are sentences lifted
# from the agent prompt itself, and no finished answer says them. Safe to
# truncate on.
_SCAFFOLD_PHRASE_PATTERN = (
    r"^[ \t]*Your next action\s*:?.*$"
    r"|^[ \t]*Available actions\s*:?.*$"
    r"|^[ \t]*Your output must be.*$"
)
_SCAFFOLD_LINE_RE = re.compile(
    _SCAFFOLD_LABEL_PATTERN + "|" + _SCAFFOLD_PHRASE_PATTERN,
    re.IGNORECASE | re.MULTILINE,
)
_SCAFFOLD_PHRASE_RE = re.compile(
    _SCAFFOLD_PHRASE_PATTERN, re.IGNORECASE | re.MULTILINE
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


# Action words the prompt has ever offered, plus the ones agents invent when
# they imitate a menu (RESTART/EXIT/RETRY/ABORT appear nowhere in the real
# prompt). Matched case-SENSITIVELY below: the shouted form is the protocol
# register, so "you must report the result" in ordinary reasoning is left
# alone while "You must RESTART to fix this violation" is not.
_THOUGHT_ACTION_WORDS = r"(?:THINK|SPAWN|TOOL|REPORT|DIE|RESTART|EXIT|RETRY|ABORT)"
# Scaffold-mimicking lines in stored reasoning, found anywhere on the line
# rather than only at its start: a DIE DEBUG dump showed the agent reading
# its own "Your only allowed action is DIE" and "Your next action: RESTART or
# EXIT?" back as if the harness had said them, and obeying.
_THOUGHT_SCAFFOLD_RES = (
    # ACTION:/PAYLOAD: labels, including markdown-decorated ones
    # ("**ACTION:** DIE", "- PAYLOAD: ...") that the anchored label pattern
    # above does not see.
    re.compile(r"^[ \t]*[*_#>`\-]*[ \t]*(?:ACTION|PAYLOAD)[*_`]*[ \t]*:", re.IGNORECASE | re.MULTILINE),
    # Headings and sentences lifted from the agent prompt itself.
    re.compile(
        r"^.*(?:"
        r"OUTPUT FORMAT INSTRUCTIONS"
        r"|Your next action"
        r"|Available actions"
        r"|Real actions that exist"
        r"|Your output must be"
        r"|respond with EXACTLY ONE action block"
        r"|Your Previous Thoughts"
        r"|END OF PREVIOUS THOUGHTS"
        r"|VERDICT FROM REVIEWER"
        r"|END VERDICT"
        r"|THINKING BUDGET EXHAUSTED"
        r"|NO ENERGY FOR NEW SUBTASKS"
        r"|You are an AI agent in a colony"
        r").*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Claims about which actions exist or are allowed.
    re.compile(
        r"^.*\b(?:your|the|my)\s+(?:only\s+)?(?:allowed|available|permitted|valid|remaining|next)\s+actions?\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    # Orders to take an action, in the protocol's shouted register.
    re.compile(
        r"^.*\b(?i:you|i)\s+(?i:must|should|can only|may only|have to|need to)"
        r"(?:\s+(?i:now|only|immediately))?\s+" + _THOUGHT_ACTION_WORDS + r"\b.*$",
        re.MULTILINE,
    ),
    # A menu: two or more shouted action words in a list ("THINK, REPORT,
    # DIE", "RESTART or EXIT").
    re.compile(
        r"^.*\b" + _THOUGHT_ACTION_WORDS + r"\b\s*(?:,|/|\||\bor\b)\s*\b"
        + _THOUGHT_ACTION_WORDS + r"\b.*$",
        re.MULTILINE,
    ),
)
# Lines the harness itself writes into thought_process. Never dropped: a
# child's result or a budget note is exactly what the agent must see.
_HARNESS_NOTE_RE = re.compile(
    r"^[ \t]*\[(?:CHILD RESULT|TOOL RESULT|BUDGET\]|REVIEW\]|earlier choice)"
)
_EXTRA_BLANK_LINES_RE = re.compile(r"\n{3,}")


def sanitize_thought_text(text):
    """
    Stored reasoning with every scaffold-mimicking line removed.

    A superset of strip_scaffolding_lines for text that is fed back to the
    agent as "Your Previous Thoughts": that section is rendered directly
    above the real action menu, so a line the model wrote in the harness's
    own voice -- a fake menu, an "only allowed action", an order to DIE --
    reads as the harness talking. Such lines are dropped whole; the
    reasoning around them is kept. Lines the harness wrote (child results,
    tool results, budget notes) are never touched.
    """
    if not text:
        return text
    kept = []
    for line in text.split("\n"):
        if not _HARNESS_NOTE_RE.match(line) and (
            _SCAFFOLD_LINE_RE.search(line)
            or any(pattern.search(line) for pattern in _THOUGHT_SCAFFOLD_RES)
        ):
            continue
        kept.append(line)
    return _EXTRA_BLANK_LINES_RE.sub("\n\n", "\n".join(kept))


_DECISION_ACTION_LINE_RE = re.compile(
    r"^[ \t]*[*_#>`\-]*[ \t]*ACTION[*_`]*[ \t]*:[*_`]*[ \t]*([A-Za-z]+).*$",
    re.IGNORECASE | re.MULTILINE,
)
_DECISION_PAYLOAD_LABEL_RE = re.compile(
    r"^[ \t]*[*_#>`\-]*[ \t]*PAYLOAD[*_`]*[ \t]*:[*_`]*[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)


def record_decision_text(text):
    """
    decide()'s raw output as it is kept in thought_process.

    decide() generates straight after the prompt's "Your next action:", so
    its output is the most scaffold-shaped text an agent ever writes, and
    it used to be stored raw -- last, so it filled most of the window the
    next decide() reads back. The ACTION: label becomes a past-tense note
    ("[earlier choice: TOOL]") and the PAYLOAD: label is dropped with its
    content kept, so the agent still knows what it tried; everything else
    goes through sanitize_thought_text.

    The choice and its payload are kept on ONE line, capped, so that
    without_decision_records can remove them whole: the cycle-cap
    fallback builds a REPORT out of recent thought_process, and a record
    of an earlier SPAWN payload is not an answer. Only the last ACTION
    block is recorded -- the same block decide()'s parser acts on.
    """
    if not text:
        return text
    matches = list(_DECISION_ACTION_LINE_RE.finditer(text))
    if not matches:
        return sanitize_thought_text(text)
    prose = sanitize_thought_text(text[:matches[0].start()]).strip()
    last = matches[-1]
    choice = last.group(1).upper()
    payload = _DECISION_PAYLOAD_LABEL_RE.sub("", text[last.end():])
    if choice == "THINK":
        # A THINK payload IS reasoning: kept as prose, uncapped, so the
        # cycle-cap fallback may still use it. Only the choice is a record.
        payload = sanitize_thought_text(payload).strip()
        return "\n".join(part for part in (prose, "[earlier choice: THINK]", payload) if part)
    payload = " ".join(sanitize_thought_text(payload).split())
    if len(payload) > _DECISION_RECORD_MAX_CHARS:
        payload = payload[:_DECISION_RECORD_MAX_CHARS] + "..."
    record = f"[earlier choice: {choice}] {payload}".rstrip()
    return f"{prose}\n{record}" if prose else record


_DECISION_RECORD_MAX_CHARS = 200
_DECISION_RECORD_LINE_RE = re.compile(r"^[ \t]*\[earlier choice:.*(?:\n|$)", re.MULTILINE)


def without_decision_records(text):
    """thought_process with record_decision_text's one-line records removed,
    for callers that treat recent thoughts as answer text."""
    if not text:
        return text
    return _DECISION_RECORD_LINE_RE.sub("", text)


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

    Returns a PREFIX of the input rather than a rejoin of the sentences it
    kept, the same way degeneracy_cut does. Rejoining flattened a bulleted
    answer into one paragraph as the price of removing its trailing
    "Done. Submitted. Confirmed." -- a cut that changes the layout of the
    part it kept is doing more than it says.
    """
    if not text:
        return text
    text = str(text)
    spans = _sentence_spans(text)
    if not spans:
        return text

    cut = len(spans)
    while cut > 0 and _is_closer_sentence(text[spans[cut - 1][0]:spans[cut - 1][1]].strip(),
                                          max_words):
        cut -= 1

    if len(spans) - cut < min_run or cut == 0:
        return text
    return text[:spans[cut - 1][1]].rstrip()


# ---------------------------------------------------------------------------
# Echo tails: closer-cycling's cousin.
#
# Observed at the end of a promoted REPORT:
#
#   "...Nothing else seems necessary at this stage. Exactly five. Done. Five."
#
# Only one of those fragments is a closer, so trim_closer_tail's run never
# starts, and none of them repeats a sentence, so the dedupe helpers pass
# it through. What the others have in common is that they say nothing new:
# "Five." re-uses a word the answer already used. The model has finished
# and is muttering its answer back to itself.
#
# Same precision-first stance as the closer trimmer. A fragment only counts
# when it is at most two words and either opens on a confirmation word
# ("Exactly five.") or has every content word be a closer, a confirmation
# word, or a word used elsewhere in the text. The
# run must include at least one real closer, which is what separates
# "Five. Done." from the tail of a short list. The run is left alone when
# the sentence before it is a question, since a one-word tail there may be
# the answer to it: "Is it feasible? Yes. Done."
#
# min_run is 2, not trim_closer_tail's 4, because this runs after the
# report-trim has cut a REPORT down to three sentences. The observed
# four-fragment tail arrives here as "Exactly five. Done.".
# ---------------------------------------------------------------------------

ECHO_MIN_RUN = 2
ECHO_MAX_WORDS = 2

_ECHO_CONFIRM_STEMS = ("exact", "correct", "indeed")

# A whole sentence that only says the answer is over: "Nothing else seems
# necessary at this stage." It is too long to be a fragment, but it carries
# nothing, so at the tail it joins the run and counts as its closer. A run
# holding one is cut even when it is shorter than min_run, since the
# report-trim can leave the sign-off as the last sentence on its own.
# Whole-sentence matches only, so "Nothing else is needed for the pump to
# start once the valve is open." is real content and stays. "No" needs a
# "further/other/..." after it: "No permit is required." is an answer.
_SIGNOFF_RE = re.compile(
    r"^(?:"
    r"(?:nothing\s+(?:else|more|further)|no\s+(?:further|other|additional|more))"
    r"\s+(?:\w+\s+)?(?:is|are|seems?|appears?)\s+(?:to\s+be\s+)?"
    r"(?:needed|necessary|required)"
    r"|that(?:'s|\s+is)\s+(?:all|it)"
    r"|this\s+(?:completes|concludes)\s+(?:the|this|my)\s+(?:task|answer|report)"
    r")(?:\s+(?:at|for)\s+(?:this|the)\s+(?:stage|point|time|moment))?[.!]?$",
    re.IGNORECASE,
)

_QUESTION_END_RE = re.compile(r"\?[\"'\)\]]*$")


def _letter_words(sentence):
    """Lowercased letters-only words, the same cleaning _is_closer_sentence uses."""
    return [w for w in (_WORD_CLEAN_RE.sub("", w.lower()) for w in sentence.split()) if w]


def trim_echo_tail(text, min_run=ECHO_MIN_RUN, max_words=ECHO_MAX_WORDS):
    """Drop a trailing run of fragments that only restate the answer or sign off.

    Returns a prefix of the input, like trim_closer_tail, so the layout of
    the kept part is unchanged. Returns the text unchanged when the run is
    shorter than min_run (and holds no sign-off sentence), has no closer in
    it, follows a question, or is the whole text.
    """
    if not text:
        return text
    text = str(text)
    spans = _sentence_spans(text)
    if not spans:
        return text

    sentences = [text[start:end].strip() for start, end in spans]
    per_sentence = [_letter_words(s) for s in sentences]
    totals = {}
    for words in per_sentence:
        for w in words:
            totals[w] = totals.get(w, 0) + 1

    cut = len(spans)
    saw_closer = saw_signoff = False
    while cut > 0:
        if _SIGNOFF_RE.match(sentences[cut - 1]):
            saw_closer = saw_signoff = True
            cut -= 1
            continue
        words = per_sentence[cut - 1]
        content = [w for w in words if w not in _CLOSER_STOPWORDS]
        if not content or len(words) > max_words:
            break
        own = {}
        for w in words:
            own[w] = own.get(w, 0) + 1
        # A word counts as an echo when it also appears in another sentence.
        # A fragment that opens on a confirmation word ("Exactly five.") is
        # a restatement whatever follows, which matters once the report-trim
        # has removed the later "Five." that the word would have echoed.
        confirms = content[0].startswith(_ECHO_CONFIRM_STEMS)
        if not confirms and not all(w.startswith(_CLOSER_STEMS + _ECHO_CONFIRM_STEMS)
                                    or totals[w] > own[w]
                                    for w in content):
            break
        if _is_closer_sentence(sentences[cut - 1], max_words):
            saw_closer = True
        cut -= 1

    if cut == 0 or not saw_closer or (len(spans) - cut < min_run and not saw_signoff):
        return text
    if _QUESTION_END_RE.search(sentences[cut - 1]):
        return text
    return text[:spans[cut - 1][1]].rstrip()


# ---------------------------------------------------------------------------
# Degeneracy: one detector, and where to cut.
#
# looks_degenerate is Agent._looks_degenerate, moved here body-and-all.
# agent_node still gates generation with it (think()'s early stop and
# decide()'s retry loop both ask it whether the model has collapsed), and
# now the final synthesis can ask the same question of the same text shape
# without a second copy of the rules -- a copy would be free to drift away
# from the one whose behaviour the run logs describe.
#
# degeneracy_cut answers the other half: not "is this collapsed" but "where
# did it collapse", so a caller holding the last text in the run can keep
# the part written before the wheels came off instead of taking or dropping
# the whole thing.

_DEGENERATE_NOISE_CHARS = "{}[]<>`"
# Whole-text check: brackets/backticks in the last 120 chars, budget 40.
_NOISE_TAIL_WINDOW = 120
_NOISE_TAIL_BUDGET = 40
# Per-sentence check: the same one-third ratio rather than a second
# invented threshold, plus a floor so a short sentence carrying one
# bracketed aside is not read as symbol-soup.
_NOISE_SENTENCE_RATIO = _NOISE_TAIL_BUDGET / _NOISE_TAIL_WINDOW
_NOISE_SENTENCE_FLOOR = 8

# A sentence long enough that one repeat is already a collapse, and the
# shortest sentence worth counting at all.
_REPEAT_LONG_SENTENCE_CHARS = 60
_REPEAT_MIN_SENTENCE_CHARS = 15


def _repeat_threshold(sentence):
    """How many occurrences of `sentence` mean the generation has collapsed.

    A flat threshold of 3 let a long block (e.g. a whole multi-sentence
    "Verified..." paragraph) repeat twice, burn roughly half of a 400-token
    budget on the duplicate, and trail off mid-sentence WITHOUT being
    flagged. A long sentence (>60 chars) repeating even once more is a much
    stronger signal than a short filler phrase repeating -- "Understood."
    three times is probably fine; a 20-word clause twice almost never is.
    """
    return 2 if len(sentence) > _REPEAT_LONG_SENTENCE_CHARS else 3


def looks_degenerate(text) -> bool:
    """Cheap heuristic check for a collapsed generation: a repeated
    sentence, a long tail of pure bracket/backtick noise, or closer-cycling.

    Closer-cycling is the shape neither of the other two sees -- every
    sentence is unique, so the repeat counter never climbs past 1, and it is
    plain prose, so the noise count stays at 0. A REPORT that trailed off
    into "Ready. Finalized. Deploying. Deployment. Done. Submitted."
    was therefore scored as a perfectly healthy generation.
    """
    if not text:
        return False
    tail = text[-_NOISE_TAIL_WINDOW:]
    noise_chars = sum(1 for c in tail if c in _DEGENERATE_NOISE_CHARS)
    if noise_chars > _NOISE_TAIL_BUDGET:
        return True

    if has_closer_run(text):
        return True

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.strip()) > _REPEAT_MIN_SENTENCE_CHARS]
    counts = {}
    for s in sentences:
        counts[s] = counts.get(s, 0) + 1
        if counts[s] >= _repeat_threshold(s):
            return True
    return False


# A prompt placeholder echoed back into prose: "<independent part A of YOUR
# task>". agent_node's SPAWN exemplars are all this shape by construction
# (deliberately domain-free placeholders, so that a copy is unmistakable).
# Matched on shape as well as on the exemplar strings themselves, since a
# placeholder the model half-rewrote ("<part B of the task>") is the same
# failure and matches no exemplar exactly.
#
# Words only, and at least two of them. "<[^<>\n]+>" -- anything between
# angle brackets -- reads ordinary prose as a placeholder and throws the
# answer away over it: "keep the tank between <5 and >10 degrees" and "send
# it to Smith <smith@example.com>" both matched it. A leading letter rules
# out the inequality, the restricted character class rules out the address,
# and requiring a space rules out "<br>".
_PLACEHOLDER_ECHO_RE = re.compile(r"<[A-Za-z][A-Za-z\- ]*\s[A-Za-z][A-Za-z\- ]*>")


# The ACTION:/PAYLOAD: labels as the harness itself writes them -- bare, or
# shouted with a word of padding. This is the ONLY scaffolding-label pattern
# degeneracy_cut consults; _SCAFFOLD_LABEL_PATTERN above is deliberately not,
# because it is IGNORECASE and a truncating caller cannot afford that (see
# the note there).
#
# Case-SENSITIVE and all-caps on purpose. "Action required: book the venue."
# and "Payload: 200 attendees" are ordinary English in a finished answer;
# "ACTION REQUESTED: RUN OR ABORT?" -- how one real run signed off its
# user-facing answer -- is the protocol talking. The optional [A-Z]+ group
# is what separates them, and it is also why the bare "ACTION: REPORT" is
# still caught: the padding word is optional, the capitals are not.
_PROTOCOL_SHOUT_RE = re.compile(
    r"^[ \t>*_#-]*(?:ACTION|PAYLOAD)(?:\s+[A-Z]+)?\s*:", re.MULTILINE
)


def _sentence_spans(text):
    """(start, end) offsets of each sentence, blanks dropped.

    The same split as _sentence_list, keeping offsets: degeneracy_cut
    returns a SLICE of the original text rather than a rejoin of the
    sentences it kept, so a bulleted or multi-line answer keeps its line
    breaks instead of being flattened into one paragraph on the way out.
    """
    spans = []
    start = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+", text):
        if text[start:boundary.start()].strip():
            spans.append((start, boundary.start()))
        start = boundary.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def normalize_words(text):
    """Casefolded, punctuation-free, whitespace-collapsed view of a string --
    so "Pick a color palette for the newsletter." and "pick a colour  palette
    for the newsletter" compare as near-identical rather than as unrelated."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


_PHRASE_LOOP_COPIES = 3
_PHRASE_LOOP_MIN_WORDS = 3
_PHRASE_LOOP_MAX_WORDS = 12
_LOOP_WORD_RE = re.compile(r"[a-z0-9']+")


def _has_phrase_loop(sentence, copies=_PHRASE_LOOP_COPIES,
                     min_words=_PHRASE_LOOP_MIN_WORDS, max_words=_PHRASE_LOOP_MAX_WORDS):
    """True when one sentence holds `copies` back-to-back copies of the same
    run of min_words..max_words words: "Crews rotate hourly, crews rotate
    hourly, crews rotate hourly." The sentence-level repeat counter sees one
    sentence, so a loop that never reaches a full stop was promoted whole.
    Three copies of three or more words, so "very, very, very" and a list
    with a shared stem ("buy milk, buy eggs") are left alone."""
    words = _LOOP_WORD_RE.findall(sentence.lower())
    for size in range(min_words, max_words + 1):
        if copies * size > len(words):
            break
        for i in range(len(words) - copies * size + 1):
            block = words[i:i + size]
            if all(words[i + k * size:i + (k + 1) * size] == block for k in range(1, copies)):
                return True
    return False


def degeneracy_cut(text, exemplars=(), repeat_threshold=None):
    """
    `text` cut at the first point where the generation stopped answering,
    as (kept_text, reason). (text, None) when it never did.

    repeat_threshold overrides _repeat_threshold's per-sentence count when
    given. The default is tuned for text still being generated, where a
    short sentence recurring twice across several paragraphs can be
    legitimate. A result that has already been trimmed to a few sentences
    (scrub_result's input) has no room for that: one repeat there is the
    loop, so scrub_result passes 2.

    "The first sign of it" is the whole point: once a decode locks onto its
    own output, or starts reciting the harness's scaffolding back, the text
    AFTER that point is the failure, not content to be salvaged and stitched
    into something that reads finished. Deduping such a tail rather than
    cutting it produces exactly that -- a plausible-looking answer assembled
    out of the model's collapse.

    Five shapes, all of them "everything after this is noise":

      * a sentence repeated up to _repeat_threshold (the loop itself),
      * a phrase looped inside one sentence (_has_phrase_loop), the same
        loop before the model reached a full stop,
      * a sentence that is mostly bracket/backtick noise,
      * an ACTION:/PAYLOAD:-style scaffolding line echoed back -- one real
        run ended its user-facing answer on "ACTION REQUESTED: RUN OR
        ABORT?", which is the harness's protocol talking, not an answer,
      * a prompt exemplar or placeholder echoed back.

    WHERE it cuts is as load-bearing as whether it cuts. The scaffolding
    patterns are line-anchored, so the cut goes at the start of the
    offending LINE, not at the start of the sentence containing it: a final
    answer that is a bulleted plan carries no sentence terminator before its
    trailing "ACTION REQUESTED:" line, which makes the whole answer one
    "sentence" -- cutting at its start threw the entire answer away to
    remove one line of scaffolding, which is a worse failure than the one
    this guard exists to fix. The other four shapes have no sub-sentence
    position to speak of and cut at the sentence boundary.

    Closer-cycling is deliberately NOT cut here: it is a degeneracy signal
    (looks_degenerate reports it anywhere) but trim_closer_tail already
    handles it where cutting is safe, i.e. at the tail, because a closer run
    buried mid-text has real content after it that cutting would destroy.

    Cutting at offset 0 returns "" -- the caller decides what an
    entirely-degenerate text means, since blanking it silently would hide
    the failure rather than report it.
    """
    if not text:
        return text, None
    spans = _sentence_spans(text)
    if not spans:
        return text, None

    exemplar_norms = [n for n in (normalize_words(e) for e in exemplars or ()) if n]

    counts = {}
    for sentence_start, sentence_end in spans:
        sentence = text[sentence_start:sentence_end]
        stripped = sentence.strip()
        cut = None
        reason = None

        if len(stripped) > _REPEAT_MIN_SENTENCE_CHARS:
            counts[stripped] = counts.get(stripped, 0) + 1
            threshold = (repeat_threshold if repeat_threshold is not None
                         else _repeat_threshold(stripped))
            if counts[stripped] >= threshold:
                cut, reason = sentence_start, "a sentence repeated"

        if reason is None and _has_phrase_loop(stripped):
            cut, reason = sentence_start, "a phrase looped within a sentence"

        if reason is None:
            noise = sum(1 for c in stripped if c in _DEGENERATE_NOISE_CHARS)
            if (noise >= _NOISE_SENTENCE_FLOOR
                    and noise > len(stripped) * _NOISE_SENTENCE_RATIO):
                cut, reason = sentence_start, "bracket/backtick noise"

        if reason is None:
            # _SCAFFOLD_PHRASE_RE, not _SCAFFOLD_LINE_RE: the label half of
            # that pattern is IGNORECASE, and cutting on it cost a plan
            # written with "Action:"/"Payload:" labels everything after its
            # first one. _PROTOCOL_SHOUT_RE covers the labels this actually
            # needs to catch, without reading ordinary English as protocol.
            echoed = (_SCAFFOLD_PHRASE_RE.search(sentence)
                      or _PROTOCOL_SHOUT_RE.search(sentence))
            if echoed is not None:
                cut = sentence_start + echoed.start()
                reason = "harness scaffolding echoed back"

        if reason is None and _PLACEHOLDER_ECHO_RE.search(sentence):
            cut, reason = sentence_start, "a prompt placeholder echoed back"

        if reason is None and exemplar_norms:
            normalized = normalize_words(stripped)
            if normalized and any(e in normalized for e in exemplar_norms):
                cut, reason = sentence_start, "a prompt exemplar echoed back"

        if reason is not None:
            return text[:cut].rstrip(), reason

    return text, None


# ---------------------------------------------------------------------------
# Status-report tails.
#
# Observed ending a synthesized FINAL ANSWER (twice), and single REPORTs
# (agent_83dcb89a, agent_d496677d):
#
#   "...Ready to deploy. Done. Go. Deployment timestamp: 2023-11-03T14:23:10Z.
#    Metrics report: accuracy=1.000000, consistency=1.000000... Final state:
#    COMPLETED... Report end."
#
# The answer is over and the model has switched register into a deploy log:
# a sign-off loop, a timestamp nobody gave it, metrics it made up, a status
# line and an end marker. No sentence repeats, "Go." breaks the closer run,
# and the log lines are too long to be closers, so every trimmer above let
# it through, and the synthesizer's global dedupe then collapsed a repeated
# "Ready to deploy. Done. Go." into a single copy that looked clean.
#
# Cut as a trailing run, like the closer and echo tails. The run is walked
# over SEGMENTS (sentences, and lines, since log lines often have no
# terminator). A segment belongs to it when it is:
#   * a closer, a sign-off or a go-word ("Go.", "Proceed."),
#   * an end marker ("Report end.", "End of report."),
#   * a status line: "<label with a log word>: <SHOUTED STATE>"
#     ("Final state: COMPLETED"),
#   * telemetry: "<label with a log word>: ..." carrying a clock timestamp or
#     key=number metrics, or a bare key=number list, whose figures the
#     grounding text does not contain.
# It is cut only when it holds an end marker, a status line or telemetry, or
# a go-word next to at least one other closer (the "ready/done/go" loop). A
# run of plain closers stays trim_closer_tail's call, with its own min_run.
# Never cut right after a question.
#
# Labels need a log word and states must be SHOUTED, so "Decision: APPROVED"
# and "Status: approved" stay: those are answers. Grounding keeps a
# timestamp or metric the problem or the results actually gave.
# ---------------------------------------------------------------------------

_SEGMENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")

# No "result", "outcome", "verdict" or "score" among the labels, and no PASS,
# APPROVED or VERIFIED among the states: a verifier's "Result: PASS." is its
# answer, not a log line.
_LOG_LABEL_WORDS = (
    r"state|status|build|deploy(?:ment|ed)?|run|job|pipeline|system|report|"
    r"exit|timestamp|time|generated|completed|finished|metrics?|log|checksum|"
    r"hash|version|uptime|latency"
)
_LOG_LABEL_RE = re.compile(
    r"^[-*•>\s]*(?P<label>[A-Za-z][A-Za-z _/-]{0,40}?)\s*:\s*(?P<value>.+)$"
)
_LOG_WORD_RE = re.compile(rf"\b(?:{_LOG_LABEL_WORDS})\b", re.IGNORECASE)
_SHOUTED_STATE_RE = re.compile(
    r"^(?:COMPLETED?|SUCCESS(?:FUL)?|SUCCEEDED|DONE|READY|OK|"
    r"FINISHED|FINALI[SZ]ED|DEPLOYED|CLOSED|TERMINATED|GO)"
    r"[\W_]*$"
)
_END_MARKER_RE = re.compile(
    r"^[-*•>\s]*(?:"
    r"(?:report|log|output|transmission|message|session|response)\s+"
    r"(?:end|ends|ended|complete|completed|closed)"
    r"|end\s+of\s+(?:report|log|output|transmission|message|session|response)"
    r"|eof|over\s+and\s+out"
    r")[\W_]*$",
    re.IGNORECASE,
)
_GO_WORD_RE = re.compile(
    r"^[-*•>\s]*(?:go|go\s+go(?:\s+go)?|proceed|ready\s+to\s+go|good\s+to\s+go)[\W_]*$",
    re.IGNORECASE,
)
# A clock timestamp, date AND time: a bare date is usually content ("the
# first meeting is 2024-03-05"), a date with a time of day in a log line is
# a log stamp.
_CLOCK_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_METRIC_ASSIGN_RE = re.compile(r"\b[A-Za-z_][\w ]{0,30}?\s*=\s*(\d+(?:\.\d+)?%?)")
_BARE_METRIC_LIST_RE = re.compile(
    r"^[-*•>\s]*(?:[A-Za-z_][\w ]{0,30}?\s*=\s*\d+(?:\.\d+)?%?[,;\s]*)+[\W_]*$"
)


def _segment_spans(text):
    """(start, end) of each sentence OR line. _sentence_spans keeps a run of
    unterminated lines together, which is exactly the shape of a log block."""
    spans = []
    start = 0
    for boundary in _SEGMENT_BOUNDARY_RE.finditer(text):
        if text[start:boundary.start()].strip():
            spans.append((start, boundary.start()))
        start = boundary.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _figures(segment):
    """The timestamps and metric values a log segment asserts."""
    return (_CLOCK_TIMESTAMP_RE.findall(segment)
            + [m.group(1) for m in _METRIC_ASSIGN_RE.finditer(segment)])


def _ungrounded(figures, grounding):
    """True when any figure is absent from the grounding text, matched as a
    whole number ("1" is not grounded by a "10" somewhere in the results)."""
    if not figures:
        return False
    if grounding is None:
        return True
    return any(
        re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", grounding) is None
        for f in figures
    )


def is_telemetry_segment(segment, grounding=None):
    """A log line asserting a clock timestamp or key=number metrics that the
    grounding text does not contain."""
    segment = segment.strip()
    if _BARE_METRIC_LIST_RE.match(segment):
        return _ungrounded(_figures(segment), grounding)
    match = _LOG_LABEL_RE.match(segment)
    if match is None or not _LOG_WORD_RE.search(match.group("label")):
        return False
    return _ungrounded(_figures(match.group("value")), grounding)


def ungrounded_telemetry(text, grounding=None):
    """Every log-line segment in `text` asserting a timestamp or metric the
    grounding does not contain. For reporting: a line mid-answer has real
    content after it, so it is flagged rather than cut."""
    if not text or not str(text).strip():
        return []
    text = str(text)
    return [text[s:e].strip() for s, e in _segment_spans(text)
            if is_telemetry_segment(text[s:e], grounding)]


def _status_segment_kind(segment, grounding):
    """"signal" for a segment that marks the log register on its own, "go"
    for a go-word, "closer" for a closer or sign-off, None otherwise."""
    segment = segment.strip()
    if _END_MARKER_RE.match(segment):
        return "signal"
    match = _LOG_LABEL_RE.match(segment)
    if (match is not None and _LOG_WORD_RE.search(match.group("label"))
            and _SHOUTED_STATE_RE.match(match.group("value").strip())):
        return "signal"
    if is_telemetry_segment(segment, grounding):
        return "signal"
    if _GO_WORD_RE.match(segment):
        return "go"
    if _is_closer_sentence(segment) or _SIGNOFF_RE.match(segment):
        return "closer"
    return None


def trim_status_tail(text, grounding=None):
    """Cut a trailing status-report/deploy-log run, as (kept, reason).

    (text, None) when there is none. ("", reason) when the WHOLE text is one,
    the degeneracy_cut contract: what an all-log answer means is the
    caller's decision. `grounding` is the text the answer was allowed to
    draw on (the problem, the subtask results); None treats every timestamp
    or metric in a log line as made up.
    """
    if not text or not str(text).strip():
        return text, None
    text = str(text)
    spans = _segment_spans(text)
    if not spans:
        return text, None

    cut = len(spans)
    kinds = []
    while cut > 0:
        kind = _status_segment_kind(text[spans[cut - 1][0]:spans[cut - 1][1]], grounding)
        if kind is None:
            break
        kinds.append(kind)
        cut -= 1

    fires = "signal" in kinds or ("go" in kinds and len(kinds) >= 2)
    if not fires:
        return text, None
    if cut > 0 and _QUESTION_END_RE.search(text[spans[cut - 1][0]:spans[cut - 1][1]].strip()):
        return text, None
    return text[:spans[cut - 1][1]].rstrip() if cut > 0 else "", "status-report tail"


def cut_adjacent_repeat(text, max_block=4, min_chars=_REPEAT_MIN_SENTENCE_CHARS):
    """Cut at the second copy of a block repeated back to back, as
    (kept, reason). (text, None) when there is none.

    A block of 1..max_block segments followed immediately by itself is the
    decode looping. degeneracy_cut counts a short sentence three times
    before calling it a loop, which is right across a long answer, but a
    back-to-back copy is the loop with nothing in between to excuse it. A
    global dedupe afterwards only collapses the copies, keeping one and
    everything after it, and what comes after a loop is more of the
    collapse ("...Done. Go. Done. Go. Deployment timestamp: ...").
    """
    if not text or not str(text).strip():
        return text, None
    text = str(text)
    spans = _segment_spans(text)
    norms = [re.sub(r"\s+", " ", text[s:e]).strip().rstrip(".!?… ").casefold()
             for s, e in spans]
    best = None
    for size in range(1, max_block + 1):
        for i in range(len(spans) - 2 * size + 1):
            block = norms[i:i + size]
            if block != norms[i + size:i + 2 * size]:
                continue
            if sum(len(n) for n in block) < min_chars:
                continue
            if best is None or i + size < best:
                best = i + size
            break
    if best is None:
        return text, None
    return text[:spans[best][0]].rstrip(), "a block repeated back to back"


def trim_degenerate_tails(text, grounding=None):
    """The three tail trimmers (status report, closer run, restating
    fragments), repeated until none applies, as (kept, reasons).

    Repeated because one can expose another: "A. Final state: COMPLETED.
    Exactly five. Done." loses its echo tail first, which leaves a status
    tail. Every step cuts a prefix, so the fixed point is what makes a
    second call a no-op. Returns "" only when the whole text is a status
    report (see trim_status_tail).
    """
    if not text or not str(text).strip():
        return text, []
    text = str(text)
    reasons = []
    while True:
        before = text
        kept, reason = trim_status_tail(text, grounding)
        if reason is not None:
            reasons.append(reason)
            text = kept
            if not text.strip():
                return "", reasons
        kept = trim_closer_tail(text)
        if kept != text:
            reasons.append("closer-cycling tail")
            text = kept
        kept = trim_echo_tail(text)
        if kept != text:
            reasons.append("restating tail")
            text = kept
        if text == before:
            return text, reasons


def scrub_result(text, exemplars=(), grounding=None):
    """
    One subtask result with its degenerate parts removed, as
    (kept_text, reasons). reasons is empty when nothing was removed.

    A REPORT that passed the judge is stored, sent to its parent, cached,
    and handed to any sibling that depends on it. None of those consumers
    cleaned it, so a tail like "Exactly five. Done." or a sentence the
    model looped on travelled with it into other agents' prompts, where a
    small model tends to continue it. This is run once, where the result
    is stored, and again at the points it is injected into a prompt.

    Two steps, in order:
      1. degeneracy_cut with repeat_threshold=2. The result has already been
         trimmed to about three sentences, so a sentence appearing twice is
         the loop starting. Everything from its second copy is cut.
      2. trim_degenerate_tails: a status-report/deploy-log tail ("Final
         state: COMPLETED. Report end.", checked against `grounding` for
         made-up timestamps and metrics), a closer run from REPORT paths
         that did not trim it at the source, and the restating fragments
         the closer trimmer does not catch.

    Every step cuts a prefix and never rewrites what it keeps, so the
    function is idempotent. That is what makes the second run at injection
    free on text that was already scrubbed.

    Can return "" when the text is degenerate from its first sentence (an
    echoed exemplar, say, or nothing but a status report). What an empty
    result means is the caller's decision, the same contract as
    degeneracy_cut.

    Prose only. Callers exempt source code, since sentence boundaries mean
    nothing there.
    """
    if not text or not str(text).strip():
        return text, []
    text = str(text)
    reasons = []

    kept, reason = degeneracy_cut(text, exemplars=exemplars, repeat_threshold=2)
    if reason is not None:
        reasons.append(reason)
        text = kept
        if not text.strip():
            return "", reasons

    kept, tail_reasons = trim_degenerate_tails(text, grounding)
    return kept, reasons + tail_reasons


# ---------------------------------------------------------------------------
# Software framing on projects that are not about software.
#
# A 4B instruction-tuned model reads "implement", "logic", "algorithm" and
# "mechanism" as a request for code, and once one agent words a subtask that
# way every agent downstream inherits it through its task text and ghost
# context. Result: a question about how a group should settle disagreements
# comes back as "scheduler.py, scorer.py, resolver.py", a made-up checksum,
# and `def select_book(preferences):`. Swapping the SPAWN exemplars for
# domain-free placeholders did not change that, so the wording is handled
# here, in code, for any project whose own request never asked for software.

# Words in the USER'S request that mean software really is on the table.
# Errs toward matching: a false hit only switches the guard off for the run
# (the old behaviour), while a miss would reword a real coding task.
_SOFTWARE_REQUEST_RE = re.compile(
    # Lookarounds rather than \b, so "C++" (ends on a non-word char) matches.
    r"(?<!\w)(?:code|codes|coding|coder|codebase|script|scripts|scripting|"
    r"software|programming|programmer|computer program|algorithms?|"
    r"pseudo-?code|python|javascript|typescript|java|c\+\+|golang|sql|"
    r"api|apis|endpoint|database|backend|frontend|website|web ?app|app|apps|"
    r"regex|compiler?|debug|repo|repository|github|docker|kubernetes|json|"
    r"csv|dataframe|dataset|cli|html|css)(?!\w)",
    re.IGNORECASE,
)
# Everyday words that also have a software sense ("library", "package",
# "function", "program", "notebook", "source") are deliberately absent:
# "a book club that meets at the library" is not a software request.

# A file name with a source-code extension: "votetally.py", "resolver.ts".
_CODE_FILENAME_RE = re.compile(
    r"\b[\w\-]+\.(?:py|ipynb|js|jsx|ts|tsx|java|cpp|cc|hpp|rb|go|rs|sh|ps1|"
    r"sql|php|cs|kt|swift|scala|lua|pl)\b",
    re.IGNORECASE,
)
# snake_case identifier called like a function: "select_book(preferences)".
_SNAKE_CALL_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\s*\(")
_SOFTWARE_ARTIFACT_WORD_RE = re.compile(
    r"\b(?:pseudo-?code|checksums?|source code|unit tests?|code snippet)\b",
    re.IGNORECASE,
)


def asks_for_software(text):
    """True when a user's request (or goal) itself asks for software. Empty
    text counts as True, so the guard below fails open."""
    if not text or not str(text).strip():
        return True
    text = str(text)
    return bool(
        _SOFTWARE_REQUEST_RE.search(text)
        or _CODE_FILENAME_RE.search(text)
        or "```" in text
        or looks_like_source_code(text)
    )


def software_artifact_reason(text):
    """
    Why this subtask/answer is shaped like a software deliverable, or None.

    Only unmistakable markers count -- a code fence, a definition line, a
    source-file name, a snake_case call, pseudocode, a checksum. Plain words
    such as "implement" are NOT reasons to reject anything (they are reworded
    instead, see plain_register): "implement the seating plan" is ordinary
    English, and rejecting it would cost a respawn for nothing.
    """
    if not text:
        return None
    text = str(text)
    if "```" in text:
        return "contains a code block"
    match = _CODE_FILENAME_RE.search(text)
    if match:
        return f"names a source-code file ({match.group(0)!r})"
    if _CODE_STATEMENT_RE.search(text):
        return "contains a code definition"
    match = _SNAKE_CALL_RE.search(text)
    if match:
        return f"contains a function call ({match.group(0).strip()!r})"
    match = _SOFTWARE_ARTIFACT_WORD_RE.search(text)
    if match:
        return f"describes a software artifact ({match.group(0)!r})"
    if looks_like_source_code(text):
        return "is written as code"
    return None


# Code-coded word -> plain word. Ordered: longer phrases first.
_PLAIN_REGISTER_SUBS = (
    # "a conflict resolver" -> "a way to resolve conflict". The \1 in the
    # template (the noun) is filled in by plain_register below.
    (re.compile(r"\b([A-Za-z\-]+) (resolvers?)\b", re.IGNORECASE), r"way to resolve \1"),
    (re.compile(r"\bimplementation of\b", re.IGNORECASE), "details of"),
    (re.compile(r"\bimplementations\b", re.IGNORECASE), "plans"),
    (re.compile(r"\bimplementation\b", re.IGNORECASE), "plan"),
    (re.compile(r"\bimplementing\b", re.IGNORECASE), "working out"),
    (re.compile(r"\bimplemented\b", re.IGNORECASE), "worked out"),
    (re.compile(r"\bimplements\b", re.IGNORECASE), "works out"),
    (re.compile(r"\bimplement\b", re.IGNORECASE), "work out"),
    (re.compile(r"\balgorithms\b", re.IGNORECASE), "methods"),
    (re.compile(r"\balgorithm\b", re.IGNORECASE), "method"),
    (re.compile(r"\bmechanisms\b", re.IGNORECASE), "methods"),
    (re.compile(r"\bmechanism\b", re.IGNORECASE), "method"),
    (re.compile(r"\blogic\b", re.IGNORECASE), "rules"),
    (re.compile(r"\bpipeline\b", re.IGNORECASE), "process"),
    (re.compile(r"\bcodify\b", re.IGNORECASE), "write down"),
)


def _match_case(source, replacement):
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def plain_register(text):
    """
    Rewords code-coded vocabulary in a task description into plain English:
    "Implement the voting logic" -> "Work out the voting rules".

    Returns (new_text, [original words replaced]). Callers apply this ONLY
    when the project did not ask for software (see asks_for_software); on a
    software project these words mean exactly what they say.
    """
    if not text:
        return text, []
    replaced = []
    new_text = str(text)
    for pattern, plain in _PLAIN_REGISTER_SUBS:
        def _sub(match, plain=plain):
            if match.re.groups:
                # Only the code-coded word is recorded, not the noun it took.
                replaced.append(match.group(match.re.groups))
                filled = plain.replace(r"\1", match.group(1).lower())
                return _match_case(match.group(0), filled)
            replaced.append(match.group(0))
            return _match_case(match.group(0), plain)
        new_text = pattern.sub(_sub, new_text)
    return new_text, replaced
