'''
the actual agent sits here
the agent can
think(): runs latent forward passes, updates hidden state, returns nothing
decide(): decodes once, returns one action token
execute(): reads the action, calls the right method


role_caps = {
    "decomposer": 512, # root agent thinks longer
    "executor": 256, # workers think less
    "verifier": 128, # verifier just checks
} - maybe double them
'''


from event_queue import Event, Messenger
from colony_state import AgentNode
from tools import ToolRegistry
from text_utils import (
    dedupe_and_cap as _dedupe_repeated_sentences,
    has_closer_run,
    normalize_identifier,
    strip_special_tokens,
    strip_scaffolding_lines,
    trim_closer_tail,
)
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import ast
import re
import json
from collections import deque


# ---------------------------------------------------------------------------
# ACTION: line parsing -- one definition, used by BOTH the streaming stop
# criterion and decide()'s own post-hoc read, which are required to agree.
#
# FIX (label-parsed-as-action): the old pattern was r"ACTION:\s*([A-Za-z]+)",
# which captures whatever word follows the FIRST colon. When the model emits
# the label twice -- "ACTION: ACTION: SPAWN", a common restatement of the
# format block -- that word is "ACTION" itself. "ACTION" is not in
# action_tokens and is too far from any of them to fuzzy-match, so a
# perfectly readable SPAWN was thrown away and downgraded to a wasted THINK
# cycle (three of them on the root, before any work started, in one run).
# Any run of repeated ACTION:/PAYLOAD: labels is now skipped, so the
# captured group is the token AFTER the label rather than the label itself.
#
# Anchored to the start of a line (after optional markdown decoration such
# as "**" or "- "). Unanchored, the label matched anywhere -- including
# inside a PAYLOAD -- and once decide() started taking the LAST block,
# prose like "PAYLOAD: The recommended action: DIE if pressure > 5 bar."
# was parsed as ACTION: DIE. A real label always opens its own line.
_ACTION_LINE_RE = re.compile(
    r"^[ \t>*_#-]*ACTION\s*:\s*(?:(?:ACTION|PAYLOAD)\b\s*:?\s*)*([A-Za-z]+)",
    re.IGNORECASE | re.MULTILINE,
)
_ACTION_TOKENS = ("THINK", "SPAWN", "TOOL", "REPORT", "DIE")
# A bare keyword opening a line, no "ACTION:" label.
_BARE_ACTION_RE = re.compile(
    r"^[ \t]*(THINK|SPAWN|TOOL|REPORT|DIE)\b", re.IGNORECASE | re.MULTILINE
)
_PAYLOAD_RE = re.compile(r"PAYLOAD:\s*(.*)", re.DOTALL | re.IGNORECASE)


def _last_action_match(text):
    """
    The action block decide() acts on: the LAST one in the generation, not
    the first.

    A model that drafts a block and then corrects itself ("ACTION: REPORT
    ... wait, this needs splitting ... ACTION: SPAWN") means the later
    block; first-match acted on the abandoned draft.

    Labelled lines win over bare keywords. A label only counts at the start
    of a line (see _ACTION_LINE_RE), so payload prose ("the recommended
    action: DIE if ...") is never a candidate. Among labelled lines, the
    last one whose token is a real action is preferred -- only if no
    labelled token is a real action does the last labelled match stand
    (decide() then fuzzy-matches or THINKs it).

    Bare-keyword fallback: the last line opening with an UPPERCASE keyword,
    or the text's own leading word in any case (the pre-existing recovery
    for "Spawn\\n{...}"). Mid-text lines need uppercase because "Think
    about..." / "Report the..." open ordinary prose lines constantly.
    """
    labelled = list(_ACTION_LINE_RE.finditer(text))
    if labelled:
        real = [m for m in labelled if m.group(1).strip().upper() in _ACTION_TOKENS]
        return (real or labelled)[-1]
    bare = [
        m for m in _BARE_ACTION_RE.finditer(text)
        if m.group(1).isupper() or not text[:m.start(1)].strip()
    ]
    return bare[-1] if bare else None


def role_may_use_tools(role):
    """Single source of truth for TOOL access by role: the prompt menu
    (Agent._role_may_tool) and the orchestrator's enforcement both read it."""
    return role != "decomposer"


def _payload_match_for(text, action_match):
    """PAYLOAD: belonging to action_match -- the first one AFTER it. A
    PAYLOAD: earlier in the text belongs to an earlier (discarded) block.
    With no action at all, the last PAYLOAD: in the text."""
    if action_match is not None:
        return _PAYLOAD_RE.search(text, action_match.end())
    last = None
    for last in re.finditer(r"PAYLOAD:", text, re.IGNORECASE):
        pass
    return _PAYLOAD_RE.search(text, last.start()) if last else None


# ---------------------------------------------------------------------------
# Every task string that appears inside a SPAWN example in the prompt below.
#
# These exist in ONE place and are interpolated into the examples rather
# than written out inline, because orchestrator.handle_spawn rejects any
# spawned subtask whose description near-matches one of them. Three separate
# attempts to fix this by rewording the prompt have failed: a decomposer
# under load copies the example's task text verbatim into its own SPAWN
# payload, and the colony then spends real agents on a subtask that belongs
# to the EXAMPLE's problem rather than its own. The guard is in code now, and
# a guard that can drift out of sync with the prompt it guards is worse than
# no guard -- so the prompt reads its strings from here too.
#
# Deliberately domain-free placeholders, not realistic tasks. The previous
# exemplars (a newsletter template, a turbine blade's material/coating,
# coolant channel geometry) were concrete enough to be lifted into a real
# plan, and concrete enough to drag an unrelated colony's vocabulary toward
# that domain even when not copied verbatim. A placeholder only teaches the
# SHAPE of a batch; copied literally it is unmistakable, and the guard
# rejects it.
#
# Add a task string to a prompt example ONLY by adding it here.
_EX_ONE_PIECE = "<one focused piece of YOUR task>"
_EX_PART_A = "<independent part A of YOUR task>"
_EX_PART_B = "<independent part B of YOUR task>"
_EX_NEEDS_A = "<step that needs the result of part A>"
_EX_FINAL_ABC = "<final step that needs the results of parts A, B and C>"

EXEMPLAR_SUBTASK_DESCRIPTIONS = (
    _EX_ONE_PIECE,
    _EX_PART_A,
    _EX_PART_B,
    _EX_NEEDS_A,
    _EX_FINAL_ABC,
)


class _ActionPayloadStop(StoppingCriteria):
    """
    N3: real early-stop for decide()'s generation, checked every step.

    Without this, decide() always burns its full max_new_tokens budget --
    confirmed via real decide() calls discarding ~1000-1300 trailing chars
    (roughly two-thirds of the 400-token budget) every single time, all of
    it generated then thrown away by the post-hoc extraction below ("N3-lite":
    _extract_first_balanced_object trims the wasted tail AFTER the fact,
    it never stops generation from producing that tail in the first place).
    This stops the model the moment its own output already contains a
    complete ACTION:/PAYLOAD: block, so the wasted tokens are never
    generated at all.

    Only decodes the NEWLY generated suffix (input_ids sliced at
    prompt_len) each step -- decoding the whole sequence repeatedly would
    re-scan the entire prompt (which can itself contain "ACTION:"/
    "PAYLOAD:" as instructional text) on every one of the ~400 steps.
    """

    # Per-action budget for a REPORT. Everything else in this class waits
    # for a STRUCTURAL closer -- a balanced "}" for SPAWN/TOOL, a blank
    # line for the rest -- but a free-text REPORT has no closing token at
    # all, so when the model does not happen to emit a blank line the
    # criterion never fires and generation runs to the full 400. That is
    # the common case, not the edge case: a colony whose subtasks are
    # "list the waste types" does not need 400 tokens to answer one, and
    # the surplus is where the restatement-and-rambling that gets the
    # REPORT rejected comes from.
    #
    # Raised from 80 (roughly 60 words). 80 was chosen as the length these
    # answers want to BE, which is the right target only if the answer
    # starts at token 1 -- and it did not: a preamble sentence ("The
    # icebreaker activity prompt and timing guide are structured as
    # follows:") consumed the entire allowance on its own, and the cap then
    # fired before a single word of the actual answer was generated. The
    # prompt now tells REPORT to lead with the answer, and the budget is
    # wide enough to survive a preamble when it appears anyway.
    #
    # The cap still fires mid-sentence -- a free-text REPORT has no closing
    # token to wait for -- so the resulting stump is walked back to the last
    # complete sentence by text_utils.drop_incomplete_tail before the judge
    # reads it. The budget is a ceiling on generation, not on what is kept.
    REPORT_MAX_NEW_TOKENS = 200

    def __init__(self, tokenizer, prompt_len, extract_balanced_object, min_new_tokens=10,
                 report_max_new_tokens=None):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.extract_balanced_object = extract_balanced_object
        self.min_new_tokens = min_new_tokens
        self.report_max_new_tokens = (
            self.REPORT_MAX_NEW_TOKENS if report_max_new_tokens is None
            else report_max_new_tokens
        )

    def _parsed_action(self, text):
        """The action decide() would parse out of this partial generation.

        Mirrors decide()'s own two-step read: a labelled "ACTION: X" line
        first, and failing that a bare leading keyword -- decide()
        recovers from a missing label that way, so the budget below has
        to recognize the same generations decide() will.
        """
        match = _last_action_match(text)
        return match.group(1).strip().upper() if match else None

    def __call__(self, input_ids, scores, **kwargs):
        new_ids = input_ids[0][self.prompt_len:]
        if len(new_ids) < self.min_new_tokens:
            return False

        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        action = self._parsed_action(text)
        if action is None:
            return False

        # REPORT's own budget, checked before the structural tests below:
        # those can only ever fire on a blank line for this action, and
        # waiting for one is exactly what burns the other 320 tokens.
        if action == "REPORT" and len(new_ids) >= self.report_max_new_tokens:
            return True

        action_match = _last_action_match(text)
        if action_match is None or action_match.re is not _ACTION_LINE_RE:
            return False
        # The payload of the LAST block, same as decide() reads -- a
        # complete earlier block no longer ends generation on its own if
        # the model has already opened a newer one.
        payload_match = _payload_match_for(text, action_match)
        if not payload_match:
            return False
        payload_raw = payload_match.group(1).strip()
        if not payload_raw:
            return False

        if payload_raw.startswith("{"):
            # JSON-shaped payload (SPAWN/TOOL) -- only complete once the
            # object actually balances; mirrors decide()'s own parser so
            # this never stops mid-object.
            return self.extract_balanced_object(payload_raw) is not None

        # Plain-text payload (REPORT/THINK/DIE) -- no structural closer to
        # wait for, so a blank line after real content is treated as "the
        # model considers this done" (matches how format_example_str's own
        # examples are paragraph-separated).
        return text.endswith("\n\n")


class Agent:
    # think()'s sampling profile. think() ran pure-greedy argmax while
    # decide() sampled on retry, which made think() -- the loop that
    # generates 256 tokens per cycle and feeds every one of them into
    # thought_process -- the one place with no escape from a locked-in
    # trajectory. Greedy decoding there is what produces the 256-token
    # repetition loops and the verbatim regurgitation of the prompt's own
    # context back into the agent's reasoning.
    THINK_TEMPERATURE = 0.7
    THINK_TOP_P = 0.9

    # Ported from decide()'s generate() call, where it has been carrying
    # repetition control on its own since no_repeat_ngram_size was
    # reverted. think() got the sampling half of that profile
    # (temperature/top_p above) but not the anti-repetition half, which is
    # the gap that let a room-booking THINK cycle emit the same sentence
    # eight times verbatim at temperature 0.7 -- sampling alone does not
    # break a loop once the model has locked onto a clause, it just picks
    # the same high-probability continuation again from a slightly wider
    # set. Same value as decide() so the two paths degrade identically.
    THINK_REPETITION_PENALTY = 1.15

    # How far back the penalty looks. model.generate() penalizes against
    # the entire context; here the context is a KV cache that survives
    # across cycles and can reach MAX_TOTAL_THINK_TOKENS, and penalizing
    # every token an agent has ever produced would tax the task's own
    # vocabulary (the words it legitimately needs to repeat) as hard as
    # the loop. A sliding window penalizes the loop -- which is local by
    # construction -- and leaves long-range reuse alone. Prompt tokens are
    # deliberately excluded for the same reason.
    THINK_REPETITION_WINDOW = 256

    # Degeneracy check cadence for think()'s early stop. _looks_degenerate
    # re-splits the whole chunk on sentence boundaries, so it is not worth
    # running per-token; every 16 tokens bounds the wasted generation at
    # 15 tokens past the repeat while keeping the regex work off the hot
    # path. The minimum exists because two sentences have to EXIST before
    # one can repeat.
    THINK_DEGEN_CHECK_EVERY = 16
    THINK_DEGEN_MIN_TOKENS = 48

    MAX_TOTAL_THINK_TOKENS = 4096
    THINK_CYCLES_BEFORE_DECIDE = 1

    # Hard ceiling on run() cycles that end without finishing or handing off.
    # Every run() past the first spends a full think() generation AND a
    # decide() generation, so an agent whose decide() keeps answering THINK
    # never ends -- one executor looped THINK seven-plus times without a
    # REPORT or DIE and was most of that run's think_tick spend on its own.
    # MAX_TOTAL_THINK_TOKENS does not catch it: it only stops think() from
    # generating, and decide() still offers THINK every cycle.
    #
    # Counts any cycle whose action is not REPORT/DIE and that left
    # `awaiting` unset -- not only THINK. A SPAWN or TOOL the agent itself
    # rejects (malformed JSON, missing keys) returns without setting
    # `awaiting`, so the next tick is another think()+decide() on the same
    # task: the same loop, and counting THINK alone never saw it. Includes
    # think()'s automatic first cycle, so 4 means three more chosen by
    # decide(). Once reached, run() skips think() and decide() is offered
    # only final_actions.
    MAX_NON_TERMINAL_CYCLES = 4

    def __init__(self, tokeniser, model, message: Messenger, node: AgentNode,
                 KV_Cache=None, last_hidden_state=None):
        self.node = node
        self.KV_Cache = KV_Cache
        self.last_hidden_state = last_hidden_state
        self.message = message
        self.tokeniser = tokeniser
        self.model = model
        self.thought_process = ""
        self.last_tool_result = None
        # FIX (retry loop): a single-slot `last_tool_call` only ever caught a
        # call identical to the *immediately* preceding one, and compared the
        # args byte-for-byte. Both holes fired together in practice: an agent
        # resubmitting "print(x)" then a newline-prefixed copy of it and
        # then "print(x)" again
        # matched neither the previous slot nor the exact-string compare, so
        # all seven identical-in-substance calls sailed through to the 15-call
        # ceiling. A short window of whitespace-normalized signatures catches
        # that pattern on call two.
        self.recent_tool_calls = deque(maxlen=self.TOOL_CALL_HISTORY)
        # Statuses of the most recent tool results, for the consecutive-failure
        # circuit breaker. A repeated *failing* call is worse than a repeated
        # successful one and gets a much tighter bound than the 15-call ceiling.
        self.recent_tool_errors = deque(maxlen=self.MAX_CONSECUTIVE_TOOL_FAILURES)
        # The error text behind those failures, so the circuit breaker can
        # hand the agent what actually went wrong instead of just a count.
        self.recent_tool_error_text = deque(maxlen=self.MAX_CONSECUTIVE_TOOL_FAILURES)
        # Set when request_tool had to repair a mangled tool_name, so the
        # agent is told what its spelling actually resolved to.
        self.pending_tool_name_correction = None
        # Count of TOOL calls blocked purely because tools are disabled for
        # this run (see MAX_DISABLED_TOOL_ATTEMPTS) -- kept separate from
        # tool_call_count, which a blocked-for-this-reason call never
        # increments (see request_tool()).
        self.disabled_tool_attempts = 0
        self.last_token_id = None
        self._total_generated = 0
        # Sliding window of recently generated token ids, used to apply
        # THINK_REPETITION_PENALTY by hand in think(). Lives on the
        # instance, not the loop, because the KV cache it mirrors survives
        # across think() cycles -- a loop that straddles a cycle boundary
        # has to stay penalized.
        self._think_recent_ids = deque(maxlen=self.THINK_REPETITION_WINDOW)
        # Cycles so far that neither finished nor handed off, against
        # MAX_NON_TERMINAL_CYCLES.
        self.non_terminal_cycles = 0
        # Set by decide() when the cap forced a different action than the
        # model chose; the orchestrator reads it after run() to tally it.
        self.cap_coerced_last_run = False
        # actions
        self.action_tokens = ["THINK", "SPAWN", "TOOL", "REPORT", "DIE"]
        # Set by run() to whatever available_tools it was actually called
        # with. None means "no restriction" (direct request_tool() calls,
        # e.g. in tests, bypass run() entirely and keep working against the
        # real registry). run() setting this to [] is what makes an
        # available_tools=[] argument actually block TOOL at execute() time
        # instead of only changing prompt wording -- see request_tool().
        self._run_available_tools = None

    @property
    def agent_id(self):
        return self.node.agent_id

    @property
    def generation(self):
        return self.node.generation

    @property
    def role(self):
        return self.node.role

    @property
    def cycles_capped(self):
        return self.non_terminal_cycles >= self.MAX_NON_TERMINAL_CYCLES

    @property
    def _decomposer_awaiting_first_spawn(self):
        return self.role == "decomposer" and not getattr(self.node, "has_spawned", False)

    @property
    def final_actions(self):
        """The whole menu once cycles_capped. A decomposer that has not
        spawned yet finishes by SPAWNing -- a REPORT from it is either
        structurally rejected (root) or its own planning notes judged as a
        subtask's answer (non-root). Once its children exist the
        orchestrator only runs it after all of them finish, so REPORT is
        then the real roll-up."""
        if self._decomposer_awaiting_first_spawn:
            return ("SPAWN", "DIE")
        return ("REPORT", "DIE")

    @property
    def _role_may_tool(self):
        # Decomposers plan and never work the problem themselves, so TOOL
        # is left off their menu entirely -- and the orchestrator refuses
        # their tool requests (Orchestrator.handle_tool_request), both via
        # role_may_use_tools so prompt and enforcement cannot drift apart.
        return role_may_use_tools(self.role)

    @property
    def task(self):
        return self.node.task

    @task.setter
    def task(self, value):
        self.node.task = value

    @property
    def task_id(self):
        return self.node.task_id

    @property
    def requirements(self):
        return self.node.requirements

    @property
    def parent_id(self):
        return self.node.parent_id

    @property
    def ghost_context(self):
        return self.node.ghost_context

    @ghost_context.setter
    def ghost_context(self, value):
        self.node.ghost_context = value

    @property
    def fail_reason(self):
        return self.node.fail_reason

    @fail_reason.setter
    def fail_reason(self, value):
        self.node.fail_reason = value

    @property
    def think_cycle(self):
        return self.node.think_cycle

    @think_cycle.setter
    def think_cycle(self, value):
        self.node.think_cycle = value

    @property
    def warning_count(self):
        return self.node.warning_count

    @warning_count.setter
    def warning_count(self, value):
        self.node.warning_count = value

    @property
    def awaiting(self):
        return self.node.awaiting

    @awaiting.setter
    def awaiting(self, value):
        self.node.awaiting = value


    def _get_role_cap(self):
        """Per-role ceiling on think()'s generation....add more when ever

        Halved (decomposer 128->64, executor 256->128, verifier 128->64).
        These were sized as "how long may a reasoning cycle run", but
        think() has no natural stop -- it is a hand-rolled forward-pass
        loop with no EOS check, so it ALWAYS runs the full cap, every
        cycle, for every agent. The budget was therefore not a ceiling
        that occasionally bound; it was the exact length of every single
        generation, and the back half of it was filler: restatement of
        the seed prompt, then the same sentence looping until the counter
        ran out.

        The cap truncates mid-sentence either way (nothing here waits for
        a sentence boundary), so spending fewer tokens to reach the same
        truncation is strictly cheaper. The degeneracy check in think()
        now usually stops the loop before even this lower cap is reached.
        """
        if self.role == "decomposer":
            return 64
        if self.role == "executor":
            return 128
        if self.role == "verifier":
            return 64
        return 128  # safety default for any role not yet in this map

    def _default_available_tools(self):
        try:
            return ToolRegistry.list_tools()
        except Exception:
            return []

    def _get_format_example(self, available_tools=None, final_only=False):
        examples = {
            "decomposer": (
                'ACTION: SPAWN\n'
                'PAYLOAD: {"role": "executor", "task": "' + _EX_ONE_PIECE + '"}'
            ),
            "verifier": (
                'ACTION: REPORT\n'
                'PAYLOAD: The implementation is correct and handles edge cases.'
            ),
            "executor": (
                'ACTION: TOOL\n'
                'PAYLOAD: {"tool_name": "run_code", "args": {"code_string": "print(\'hello\')"}}'
            ),
        }
        tools_blocked = self._tools_blocked(available_tools)

        positive = examples.get(self.role, examples["executor"])
        if tools_blocked and positive.startswith("ACTION: TOOL"):
            # Showing a TOOL example while TOOL is refused is the single
            # strongest nudge back into the loop we are trying to break.
            # Widened from (circuit_open or disabled_closed) to the shared
            # gate: with tools disabled for the run, EVERY executor was
            # still being shown a run_code call as its one worked example
            # of a valid response, which is why so many of them opened
            # with one.
            positive = examples["verifier"]

        if final_only:
            # Cycle cap reached with a REPORT/DIE menu, so the only worked
            # example is a REPORT -- no SPAWN batches, no tool reference.
            return (
                f"Example of a VALID response:\n{examples['verifier']}\n\n"
                f"Example of an INVALID response (do NOT do this -- free-form "
                f"reasoning with no ACTION: line is never an acceptable output):\n"
                f"<free-form reasoning with no ACTION: line -- placeholder, not a task>\n\n"
            )

        decomposer_batch_example = (
            'Every subtask you SPAWN in a batch MUST include an explicit '
            '"dependencies" field -- treat this as producing a topological '
            'sort of the work, not a flat list of ideas. There is no '
            '"leave it out for independent tasks" option: independent '
            'tasks get "dependencies": [] explicitly, not an omitted key. '
            'Use "label" on every subtask so later ones can reference it.\n\n'
            'Example of a batch with two INDEPENDENT subtasks (neither '
            'needs the other\'s result, so both get an explicit empty '
            'list, not a missing field):\n'
            'ACTION: SPAWN\n'
            'PAYLOAD: {"subtasks": [\n'
            '  {"label": "a", "role": "executor", "task": "' + _EX_PART_A + '", "dependencies": []},\n'
            '  {"label": "b", "role": "executor", "task": "' + _EX_PART_B + '", "dependencies": []}\n'
            ']}\n\n'
            'Example of a batch with a real SEQUENTIAL dependency (one '
            'piece genuinely cannot start until another\'s result exists) '
            '-- give the piece being depended on a "label" and list that '
            'label in the dependent piece\'s "dependencies":\n'
            'ACTION: SPAWN\n'
            'PAYLOAD: {"subtasks": [\n'
            '  {"label": "a", "role": "executor", "task": "' + _EX_PART_A + '", "dependencies": []},\n'
            '  {"role": "executor", "task": "' + _EX_NEEDS_A + '", "dependencies": ["a"]}\n'
            ']}\n\n'
            'Example combining all three shapes in ONE batch -- some '
            'subtasks start immediately (empty dependencies), one needs '
            'another\'s result (sequential), and one final subtask needs '
            'several earlier ones done first (terminal):\n'
            'ACTION: SPAWN\n'
            'PAYLOAD: {"subtasks": [\n'
            '  {"label": "a", "role": "executor", "task": "' + _EX_PART_A + '", "dependencies": []},\n'
            '  {"label": "b", "role": "executor", "task": "' + _EX_PART_B + '", "dependencies": []},\n'
            '  {"label": "c", "role": "executor", "task": "' + _EX_NEEDS_A + '", "dependencies": ["a"]},\n'
            '  {"role": "executor", "task": "' + _EX_FINAL_ABC + '", "dependencies": ["a", "b", "c"]}\n'
            ']}\n\n'
            'Watch for this mistake: a subtask worded "the proposed/chosen/'
            'selected X" implies another subtask produces X first -- make '
            'sure that producing subtask exists in this batch AND is listed '
            'in the dependent one\'s "dependencies".\n\n'
            'SAFETY NET (do not rely on this -- it exists only for a subtask '
            'you genuinely forgot to annotate): if "dependencies" is missing '
            'entirely from one subtask, the system will default it to '
            'depending on the subtask listed immediately before it, which is '
            'almost always the WRONG choice for independent work and wastes '
            'real parallel execution time. Declaring "dependencies" '
            'explicitly on every subtask, every time, avoids this.\n\n'
        )


        # This string is shown to the model as an example of BAD output.
        # It must not contain anything that can be lifted out and read as
        # a task: a real-sounding sentence here gets copied verbatim into
        # a SPAWN payload (confirmed -- one agent spawned "Implement the
        # binomial probability mass function..." into an unrelated
        # engineering colony straight out of the previous wording here,
        # killing four downstream agents on the contradiction). The
        # example only has to demonstrate SHAPE, so it carries no domain
        # content at all.
        negative = "<free-form reasoning with no ACTION: line -- placeholder, not a task>"

        # Dropped wholesale when TOOL cannot fire. This table is five
        # lines of exact-key JSON for tools the agent is not permitted to
        # call -- it is both the largest tool-shaped block in the prompt
        # and an implicit claim that calling them is on the table.
        tool_arg_reference = "" if tools_blocked or not self._role_may_tool else (
            "Tool argument reference (use the EXACT keys shown for each "
            "tool -- they are NOT interchangeable):\n"
            '- run_code: {"tool_name": "run_code", "args": {"code_string": "..."}}\n'
            '- verify_math: {"tool_name": "verify_math", "args": {"expression": "...", "mode": "computational"}}\n'
            '- safe_read_file: {"tool_name": "safe_read_file", "args": {"filepath": "..."}}\n'
            '- write_file: {"tool_name": "write_file", "args": {"filepath": "...", "content": "..."}}\n'
            '- query_dataframe: {"tool_name": "query_dataframe", "args": {"filepath": "...", "action": "summary"}}\n'
        )

        # SPAWN examples go to decomposers only. A SPAWN from any other role
        # is refused by orchestrator._reject_spawn_from_non_decomposer, so
        # showing executors a worked SPAWN (they used to get a single-child
        # example AND the whole batch set) only taught a move that bounces.
        batch_alternative = decomposer_batch_example if self.role == "decomposer" else ""

        return (
            f"Example of a VALID response:\n{positive}\n\n"
            f"{batch_alternative}"
            f"Example of an INVALID response (do NOT do this -- free-form "
            f"reasoning with no ACTION: line is never an acceptable output):\n"
            f"{negative}\n\n"
            f"{tool_arg_reference}"
        )

    def _get_role_constraint_str(self, available_tools=None):
        # Same gate as the prompt builders, taking the caller's
        # effective tool set rather than re-deriving one from
        # _run_available_tools, which a direct decide()/think() call
        # that skipped run() would never have set.
        tools_blocked = self._tools_blocked(available_tools)
        if self.role == "decomposer":
            base_rule = (
                "CRITICAL RULE: You are FORBIDDEN from solving the problem "
                "directly. Your ONLY job is to break this into subtasks and "
                "SPAWN specialized agents. If you find yourself writing code "
                "or working out the solution, STOP and SPAWN instead.\n"
                "IMPORTANT: any constraints listed below apply to the "
                "PROJECT as a whole, not to you individually -- they will "
                "be satisfied collectively by the different sub-agents you "
                "spawn (e.g. a hardware constraint goes to one child, a "
                "software/algorithm constraint goes to another). Seeing "
                "constraints that look incompatible with EACH OTHER is "
                "normal and expected -- that is a reason to split the work "
                "across multiple specialized children, not a reason to "
                "DIE. Only DIE if the task itself is impossible regardless "
                "of how it's decomposed.\n"
                "IMPORTANT: if you can already identify SEVERAL independent "
                "pieces of work up front (parts of the task that can each "
                "be worked out without waiting for another's result), "
                "SPAWN all of them in ONE action "
                "using the \"subtasks\" list format shown below, instead of "
                "spawning one, waiting for it to finish, then spawning the "
                "next. Independent work should run in parallel, not "
                "queued one at a time. Only spawn a single child at a time "
                "when the next piece of work genuinely depends on the "
                "previous one's result.\n"
            )
            if self.generation == 0:
                tier_rule = (
                    "You are the ROOT decomposer (generation 0). SPAWN ONLY "
                    "\"decomposer\" children here, one per large independent "
                    "chunk of the problem -- do NOT spawn "
                    "\"executor\" children directly yourself. Each decomposer "
                    "child you create will handle breaking its chunk down "
                    "further into small, concrete pieces of work.\n"
                    "STAGE your children as a real dependency graph, not one "
                    "flat independent batch and not one single linear chain. "
                    "EVERY child you list MUST have an explicit "
                    "\"dependencies\" field -- never omit it, even for a "
                    "chunk with no prerequisites (give it \"dependencies\": "
                    "[] explicitly instead of leaving the key out). A "
                    "typical problem has THREE kinds of "
                    "structure, and you should use whichever apply:\n"
                    "  1. PARALLEL stage: chunks with no dependency on each "
                    "other (they can all start immediately) -- give these "
                    "\"dependencies\": [] explicitly.\n"
                    "  2. SEQUENTIAL stage: a chunk that genuinely needs "
                    "another chunk's actual computed result as an input "
                    "(it cannot even start without that value) -- give "
                    "it \"dependencies\": [\"<the label of the chunk it "
                    "needs>\"]. Its agent will automatically receive that "
                    "prerequisite's real result once it's done -- do not "
                    "invent placeholder numbers for something a dependency "
                    "will actually compute.\n"
                    "  3. TERMINAL stage: a final chunk that can only run "
                    "once several earlier chunks are ALL done (it combines "
                    "or checks their results) -- "
                    "list every one of those chunks in its \"dependencies\".\n"
                    "Use \"label\" on every chunk so later chunks can "
                    "reference it by name in their own \"dependencies\".\n"
                )
            else:
                tier_rule = (
                    "You are a NON-ROOT decomposer. SPAWN ONLY \"executor\" "
                    "children here, and make each one's task GRANULAR -- "
                    "small enough that it should take an executor no more "
                    + ("than 2-3 think cycles to "
                       if tools_blocked else
                       "than 2-3 think cycles and at most one TOOL call to ") +
                    "finish (e.g. \"calculate X given these inputs\", "
                    "\"look up the density of Y\", \"print the result of "
                    "this formula\") -- never a whole sub-project. If you "
                    "can't picture an "
                    "executor finishing it almost immediately, break it "
                    "down further into more, smaller pieces instead.\n"
                )
            return base_rule + tier_rule
        # Executors only. handle_failure re-plans a "TASK TOO LARGE:" DIE as
        # a decomposer only when it comes from an executor, so offering the
        # move to a verifier taught it a DIE that just fails its task.
        too_large_rule = (
            "IMPORTANT: your task may be larger than a single agent should "
            "handle directly. If it genuinely contains multiple substantial, "
            "independent pieces of work, you cannot split it yourself. "
            "Instead, ACTION: DIE with a PAYLOAD "
            "that starts with the exact phrase \"TASK TOO LARGE:\" followed "
            "by why. You will be replaced by a decomposer that breaks your "
            "task down properly. "
            + ("Use REPORT when you can produce the answer yourself in one "
               "pass" if tools_blocked else
               "Use TOOL/REPORT when you can produce the answer yourself in "
               "one pass") +
            " -- reserve this DIE for when the "
            "task is genuinely several tasks wearing one description.\n"
        ) if self.role == "executor" else ""
        return (
            "IMPORTANT: the constraints listed below apply to the PROJECT "
            "as a whole -- not every constraint necessarily applies to "
            "YOUR specific task. If a constraint is clearly outside what "
            "you were asked to do (e.g. a hardware/analog constraint for a "
            "pure software task), it belongs to a different specialized "
            "agent. Do your best on what's relevant to your task, and note "
            "any out-of-scope constraints as open items in your REPORT "
            "rather than treating them as a reason to DIE.\n"
            + too_large_rule +
            "IMPORTANT: if a task asks you to estimate, specify, or "
            "calculate a real-world value (a material property, a physical "
            "constant, a typical engineering figure) and you don't have an "
            "exact experimental lookup available, use your general "
            "engineering/scientific knowledge to give a reasonable, clearly "
            "labeled ESTIMATE or typical reference value instead. This is "
            "normal, expected engineering practice, not a reason to DIE -- "
            "reserve DIE for tasks that are conceptually impossible given "
            "your role, not for 'I don't have an exact measured number.'\n"
        )

    def _build_thinking_seed(self, requirements=None, available_tools=None):
        if requirements is None:
            requirements = self.requirements or []
        if available_tools is None:
            available_tools = self._default_available_tools()
        role_constraint_str = self._get_role_constraint_str(available_tools)
        requirements_str = (
            "Constraints you must satisfy:\n" + "\n".join(f"- {r}" for r in requirements) + "\n"
            if requirements else ""
        )
        tools_blocked = self._tools_blocked(available_tools)
        tools_str = (
            f"Tools actually available to you: {available_tools}\n"
            if available_tools and not tools_blocked and self._role_may_tool else ""
        )
        if self.role == "decomposer":
            # Role-scoped menu: a decomposer plans and never runs tools
            # (its role rule forbids working the problem itself), so the
            # tool-state branches below do not apply to it at all.
            actions_str = "Real actions that exist for your role: THINK, SPAWN, REPORT, DIE (no others exist).\n"
        elif self.tool_circuit_open:
            actions_str = (
                "Real actions that exist for your role: THINK, REPORT, DIE "
                "(TOOL is closed to you after repeated tool failures -- do "
                "not attempt it).\n"
            )
        elif self.disabled_tool_closed:
            actions_str = (
                "Real actions that exist for your role: THINK, REPORT, DIE "
                "(TOOL is closed to you for the rest of this run -- do not "
                "attempt it).\n"
            )
        elif tools_blocked:
            # Tools disabled for the whole run. This branch did not
            # exist: an empty available_tools silenced tools_str above
            # (so the agent was never told WHICH tools it had) and then
            # fell through to the line below, which advertises TOOL as a
            # real action anyway. The agent therefore knew TOOL existed,
            # knew nothing about what it could call, and guessed --
            # run_code and write_file, the two it had just seen in the
            # format example. Every guess cost a full think() cycle to
            # compose and another to recover from the rejection.
            actions_str = (
                "Real actions that exist for your role: THINK, REPORT, DIE. "
                "There are NO tools in this run -- TOOL is not an "
                "available action and any tool call will be refused. "
                "Produce the answer from your own reasoning.\n"
            )
        else:
            actions_str = "Real actions that exist for your role: THINK, TOOL, REPORT, DIE (no others exist).\n"
        ghost_str = f"Ghost Context: {self.ghost_context}\n" if self.ghost_context else ""
        # Delimited and labelled as an external verdict, not narrated as a
        # sentence. "WARNING - Previous Action Failed: <prose>" read as one
        # more line of the agent's own commentary, especially with the
        # thoughts section rendered in the same register directly below;
        # the model continued it instead of acting on it.
        fail_str = (
            f"[VERDICT FROM REVIEWER -- this is not your reasoning, it is a "
            f"ruling on your last attempt]\n{self.fail_reason}\n"
            f"[END VERDICT]\n"
            if self.fail_reason else ""
        )
        tool_result_str = (
            f"Result of your most recent TOOL call:\n{self.last_tool_result}\n"
            if self.last_tool_result else ""
        )
        return (
            f"You are an AI agent in a colony of agents working together to solve problems.\n"
            f"Your Role: {self.role}\n"
            f"{role_constraint_str}Your Task: {self.task}\n\n"
            f"{requirements_str}{tools_str}{actions_str}{ghost_str}{fail_str}{tool_result_str}"
            f"Think through how to approach this task."
        )

    def _build_prompt(self, available_roles=["decomposer", "executor", "verifier"],
                       available_tools=None, requirements=None):
        if available_tools is None:
            available_tools = self._default_available_tools()
        if requirements is None:
            requirements = self.requirements or []

        ghost_str = f"Ghost Context: {self.ghost_context}\n" if self.ghost_context else ""
        # Delimited and labelled as an external verdict, not narrated as a
        # sentence. "WARNING - Previous Action Failed: <prose>" read as one
        # more line of the agent's own commentary, especially with the
        # thoughts section rendered in the same register directly below;
        # the model continued it instead of acting on it.
        fail_str = (
            f"[VERDICT FROM REVIEWER -- this is not your reasoning, it is a "
            f"ruling on your last attempt]\n{self.fail_reason}\n"
            f"[END VERDICT]\n"
            if self.fail_reason else ""
        )
        tool_result_str = (
            f"Result of your most recent TOOL call:\n{self.last_tool_result}\n"
            if self.last_tool_result else ""
        )
        _tail = self.thought_process[-500:]
        _last_newline = _tail.rfind("\n")
        if _last_newline > 0:
            _tail = _tail[:_last_newline]
        _tail = strip_special_tokens(_tail)
        thoughts_str = (
            f"Your Previous Thoughts (most recent):\n...{_tail}\n"
            if self.thought_process else ""
        )
        requirements_str = (
            "Constraints you must satisfy:\n" + "\n".join(f"- {r}" for r in requirements) + "\n"
            if requirements else ""
        )
        format_example_str = self._get_format_example(available_tools)
        role_constraint_str = self._get_role_constraint_str(available_tools)

        # Circuit breaker open: TOOL is refused by request_tool anyway, so
        # advertising it here only invites another wasted decide() cycle that
        # gets bounced. Drop it from the menu and say why. Same logic applies
        # when the caller passed available_tools=[] for this run (e.g. the
        # tools-disabled isolation experiment): TOOL is enforced-blocked at
        # request_tool() now (see _run_available_tools), so don't advertise
        # it as an option here either -- drop the action and its PAYLOAD
        # format line entirely rather than showing "Available tools: []".
        if not self._role_may_tool:
            # Decomposers plan; they never call tools. Not "UNAVAILABLE"
            # -- simply not on this role's menu.
            tool_action_line = ""
            tool_format_line = ""
        elif self.tool_circuit_open:
            tool_action_line = (
                "- TOOL   — UNAVAILABLE. Your last "
                f"{self.MAX_CONSECUTIVE_TOOL_FAILURES} tool calls all failed; "
                "further TOOL actions are refused. Do not attempt one.\n"
            )
            tool_format_line = ""
        elif self.disabled_tool_closed:
            tool_action_line = (
                "- TOOL   — UNAVAILABLE. You have already tried "
                f"{self.disabled_tool_attempts} tool call(s) that aren't in "
                "your available set; TOOL is closed for the rest of this "
                "run. Do not attempt one.\n"
            )
            tool_format_line = ""
        elif self._tools_blocked(available_tools):
            # Same gate the thinking seed and the format example use now,
            # so a run with tools disabled produces a consistent prompt at
            # all three sites instead of one that drops TOOL here while
            # still advertising it during think().
            tool_action_line = ""
            tool_format_line = ""
        else:
            tool_action_line = (
                f"- TOOL   — call an external tool. Available tools: {available_tools}.\n"
            )
            tool_format_line = '- If TOOL: Provide a JSON object: {"tool_name": "name", "args": {...}}\n'

        # Role-scoped menu. Only a decomposer may SPAWN (the orchestrator
        # refuses anyone else's), so executors and verifiers are not shown
        # it; and a decomposer's REPORT is the roll-up of its children's
        # results, not the "give the answer itself" line written for the
        # agents that actually do the work.
        if self.role == "decomposer":
            spawn_action_line = (
                f"- SPAWN  — create sub-agents to handle sub-tasks. Available roles: {available_roles}.\n"
            )
            spawn_format_line = (
                '- If SPAWN: Provide a JSON object: {"role": "chosen_role", "task": "specific task definition"}, '
                'or a {"subtasks": [...]} batch as shown below\n'
            )
            report_action_line = (
                "- REPORT — your children have reported back; submit their combined result to your parent.\n"
            )
            report_format_line = (
                "- If REPORT: Combine your children's results into one plain-text result. "
                "Do not solve anything they did not.\n"
            )
        else:
            spawn_action_line = ""
            spawn_format_line = ""
            report_action_line = (
                "- REPORT — your task is complete, submit your final result to your parent.\n"
            )
            report_format_line = (
                "- If REPORT: Provide the final answer or result in plain text. "
                "Give the answer itself first. Do not introduce it.\n"
            )

        think_action_line = (
            "- THINK  — continue reasoning before acting (Use this to plan your next move).\n"
        )
        think_format_line = "- If THINK: Provide your reasoning in plain text.\n"
        think_cap_str = ""
        if self.cycles_capped:
            # Cycle budget spent: final_actions is the whole menu, and every
            # other line of the prompt has to agree with it -- the role rule
            # and the example included, not just the action list.
            think_action_line = think_format_line = ""
            tool_action_line = tool_format_line = ""
            if self._decomposer_awaiting_first_spawn:
                report_action_line = report_format_line = ""
                think_cap_str = (
                    f"[THINKING BUDGET EXHAUSTED -- you have already used "
                    f"{self.non_terminal_cycles} cycles on this task without "
                    f"spawning. You must act now: SPAWN your subtasks, or DIE "
                    f"if this task cannot be decomposed.]\n"
                )
                # The normal decomposer example is already SPAWN-only.
            else:
                spawn_action_line = spawn_format_line = ""
                if self.role == "decomposer":
                    role_constraint_str = (
                        "CRITICAL RULE: your subtasks are finished and their "
                        "results are in your previous thoughts. Your ONLY job "
                        "now is to combine those results for your parent. Do "
                        "not create new subtasks and do not solve anything "
                        "your children did not.\n"
                    )
                think_cap_str = (
                    f"[THINKING BUDGET EXHAUSTED -- you have already used "
                    f"{self.non_terminal_cycles} cycles on this task. You must "
                    f"finish now: REPORT your best result (state plainly anything "
                    f"you could not work out), or DIE if you have nothing usable.]\n"
                )
                format_example_str = self._get_format_example(available_tools, final_only=True)

        prompt = f"""You are an AI agent in a colony of agents working together to solve problems.
Your Agent ID: {self.agent_id}
Your Role: {self.role}
{role_constraint_str}Your Task: {self.task}

{requirements_str}{ghost_str}{fail_str}{tool_result_str}{thoughts_str}{think_cap_str}
Available actions:
{think_action_line}{spawn_action_line}{tool_action_line}{report_action_line}- DIE    — you cannot complete this task, signal failure to your parent.

OUTPUT FORMAT INSTRUCTIONS:
You must respond with EXACTLY ONE action block in the following format:

ACTION: <One of the available actions>
PAYLOAD: <Depends on the action>
{think_format_line}{spawn_format_line}{tool_format_line}{report_format_line}- If DIE: Provide the reason you cannot proceed.

{format_example_str}
Your next action:"""
        
        return prompt
    

    def think(self, available_roles=None, available_tools=None, requirements=None):
        """
        See class-level docstring/comments in the previous revision for the
        full history of fixes to this method -- unchanged in this pass.
        """
        max_tokens = self._get_role_cap() 
        self.think_cycle += 1

        if self.KV_Cache is None:
            prompt = self._build_thinking_seed(requirements, available_tools)
            inputs = self.tokeniser(prompt, return_tensors="pt").to(self.model.device)
            input_ids = inputs["input_ids"]
            self._total_generated = 0  # fresh reasoning chain -- reset lifetime counter
            self._think_recent_ids.clear()  # ...and the repetition window it shares a context with

        else:
            if self.last_token_id is not None:
                input_ids = torch.tensor([[self.last_token_id]], device=self.model.device)
            else:
                raise ValueError("KV_Cache is present, but missing last_token_id to continue generation.")

        hit_token_ceiling = False
        hit_degenerate = False
        generated_chunk = ""

        with torch.no_grad():
            for step in range(max_tokens):
                if self._total_generated >= self.MAX_TOTAL_THINK_TOKENS:
                    hit_token_ceiling = True
                    break

                if step % 25 == 0:
                    # rep= reports the penalty actually in force and how many
                    # distinct token ids it is currently suppressing. The
                    # early-stop below only prints when it FIRES, so a clean
                    # run produced no think()-side evidence at all that either
                    # mechanism was wired in -- this makes the penalty visible
                    # on every heartbeat instead of only on failure.
                    print(f"  [think() heartbeat] agent={self.agent_id} role={self.role} "
                          f"step={step}/{max_tokens} total_generated={self._total_generated} "
                          f"rep={self.THINK_REPETITION_PENALTY}/{len(set(self._think_recent_ids))}")

                outputs = self.model(
                input_ids=input_ids,
                past_key_values=self.KV_Cache,
                use_cache=True
                )

                self.last_hidden_state = None

                self.KV_Cache = outputs.past_key_values
                logits = outputs.logits

                # Nucleus sample rather than argmax. Done by hand because
                # this is a manual forward-pass loop, not model.generate():
                # temperature-scale, keep the smallest set of tokens whose
                # cumulative probability reaches THINK_TOP_P, renormalize
                # over just those, draw one.
                # Cloned: the penalty below writes in place, and the raw
                # slice is a view onto outputs.logits.
                next_token_logits = logits[:, -1, :].clone()

                # Repetition penalty, applied to the raw logits BEFORE the
                # temperature divide -- same order model.generate() uses, so
                # a given penalty value means the same thing here as it does
                # in decide(). Divide a positive score / multiply a negative
                # one, which pushes a token that has already appeared toward
                # -inf from either side (a flat subtraction would invert the
                # ranking of negative scores).
                if self._think_recent_ids:
                    recent = torch.tensor(
                        list(set(self._think_recent_ids)),
                        device=next_token_logits.device,
                        dtype=torch.long,
                    )
                    seen = next_token_logits[:, recent]
                    next_token_logits[:, recent] = torch.where(
                        seen > 0,
                        seen / self.THINK_REPETITION_PENALTY,
                        seen * self.THINK_REPETITION_PENALTY,
                    )

                next_token_logits = next_token_logits / self.THINK_TEMPERATURE
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                # Subtracting each token's own probability keeps the token
                # that CROSSES the threshold, and guarantees the top-1 token
                # is never masked (its cumulative-minus-self is 0).
                cutoff = sorted_probs.cumsum(dim=-1) - sorted_probs > self.THINK_TOP_P
                sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
                choice = torch.multinomial(
                    torch.softmax(sorted_logits, dim=-1), num_samples=1
                )
                next_token_tensor = sorted_indices.gather(-1, choice).squeeze(-1)
                self.last_token_id = next_token_tensor.item()
                self._total_generated += 1
                self._think_recent_ids.append(self.last_token_id)

                token_text = self.tokeniser.decode([self.last_token_id], skip_special_tokens=True)
                generated_chunk += token_text

                input_ids = next_token_tensor.unsqueeze(0)

                # Early stop on a collapsed generation. The penalty above
                # makes a verbatim loop much less likely; it does not make
                # it impossible, and when one does start there is nothing
                # else in this loop that can end it -- there is no EOS check
                # here, so a degenerate chain otherwise runs out the full
                # role cap and files every repeat into thought_process.
                # _looks_degenerate is the same check decide() retries on
                # (repeated sentence, or a tail of bracket/backtick noise);
                # here there is no retry to fall back on, so the cycle just
                # ends and the accumulated text is kept as-is.
                if (self._total_generated >= self.THINK_DEGEN_MIN_TOKENS
                        and step % self.THINK_DEGEN_CHECK_EVERY == 0
                        and self._looks_degenerate(generated_chunk)):
                    hit_degenerate = True
                    print(f"  [think() early-stop] agent={self.agent_id} role={self.role} "
                          f"generation looked degenerate at step={step}/{max_tokens} "
                          f"-- ending cycle instead of spending the remaining budget.")
                    break

        # Sanitize the whole cycle's output before storing it -- scaffolding
        # lines (ACTION:/PAYLOAD:/etc.) can only be recognized reliably once
        # a full line has accumulated, not token-by-token mid-line.
        self.thought_process += strip_scaffolding_lines(strip_special_tokens(generated_chunk))

        # A cycle that collapsed is treated like the token ceiling: another
        # THINK continues the same KV cache that just produced the loop, so
        # it would most likely reproduce it. Hand control to decide()
        # instead, which has its own retry loop and can re-roll.
        if (self.think_cycle <= self.THINK_CYCLES_BEFORE_DECIDE
                and not hit_token_ceiling and not hit_degenerate):
            return "THINK"
        else:
            return "FORCE_DECIDE"

    @staticmethod
    def _extract_first_balanced_object(text):
        if not text.startswith("{"):
            return None
        depth = 0
        in_string = False
        string_char = None
        escape = False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == string_char:
                    in_string = False
            else:
                if ch in ("'", '"'):
                    in_string = True
                    string_char = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[:i + 1]
        return None

    # Payload keys that make a recovered object usable by the action that
    # was actually parsed. A recovered object that satisfies none of these
    # is not this action's payload, and handing it over would be worse than
    # admitting the parse failed.
    _PAYLOAD_SHAPE = {
        "SPAWN": ("subtasks", "task", "description"),
        "TOOL": ("tool_name",),
    }

    @classmethod
    def _recover_unlabelled_payload(cls, action, text):
        """
        The JSON payload of a SPAWN/TOOL that never emitted a "PAYLOAD:"
        label, or emitted it somewhere the PAYLOAD: regex could not see.

        FIX (SPAWN mislabel, second half): the first half of this fix
        stopped a payload-less SPAWN from being relabelled REPORT -- which
        ended the task on a parse failure -- and degraded it to THINK
        instead. Safe, but still wrong in the case that actually happens:
        the model wrote the object, just without the label in front of it
        ("ACTION: SPAWN" then a bare {...} on the next line). The action is
        known, the object is right there, and the agent was still burning a
        cycle re-deriving it -- reliably the root's opening cycles. So look
        for a well-formed object of the right SHAPE anywhere in the
        generation before falling back to THINK.

        Returns the dict, or None if nothing of the right shape is present
        (in which case the caller degrades to THINK exactly as before).
        """
        required = cls._PAYLOAD_SHAPE.get(action)
        if not required or not text:
            return None

        search_from = 0
        while True:
            start = text.find("{", search_from)
            if start == -1:
                return None
            extracted = cls._extract_first_balanced_object(text[start:])
            search_from = start + 1
            if extracted is None:
                continue
            candidate = None
            try:
                candidate = json.loads(extracted)
            except json.JSONDecodeError:
                try:
                    candidate = ast.literal_eval(extracted)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(candidate, dict) and any(k in candidate for k in required):
                return candidate

    @staticmethod
    def _looks_degenerate(text: str) -> bool:
        """Cheap heuristic check for a collapsed generation: either a
        repeated sentence, or a long tail of pure bracket/backtick noise.

        FIX: confirmed via a real run -- the old flat threshold of 3
        occurrences let a long block (e.g. a whole multi-sentence
        "Verified..." paragraph) repeat twice, burn roughly half of a
        400-token budget on the duplicate, and trail off mid-sentence
        WITHOUT ever being flagged, since it never reached a third
        repetition. A long sentence (>60 chars) repeating even once more
        is already a much stronger degeneracy signal than a short filler
        phrase repeating -- e.g. "Understood." repeating 3 times is
        probably fine; a 20-word clause repeating twice almost never is.
        Scaling the threshold by sentence length catches this earlier
        without over-triggering on short, legitimately-repeated phrases.
        """
        tail = text[-120:]
        noise_chars = sum(1 for c in tail if c in "{}[]<>`")
        if noise_chars > 40:
            return True

        # Third shape: closer-cycling. Neither check below sees it -- every
        # sentence is unique, so the repeated-sentence counter never climbs
        # past 1, and it is plain prose, so the bracket-noise count stays at
        # 0. A REPORT that trailed off into "Ready. Finalized. Deploying.
        # Deployment. Done. Submitted. Confirmed. Completed." was therefore
        # scored as a perfectly healthy generation by this function, which
        # is why decide() never retried it and think() never cut it short.
        if has_closer_run(text):
            return True
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
        counts = {}
        for s in sentences:
            counts[s] = counts.get(s, 0) + 1
            threshold = 2 if len(s) > 60 else 3
            if counts[s] >= threshold:
                return True
        return False

    def decide(self, available_roles=None, available_tools=None, requirements=None):
        if available_roles is None:
            available_roles = ["decomposer", "executor", "verifier"]
        if available_tools is None:
            available_tools = self._default_available_tools()
        if requirements is None:
            requirements = self.requirements or []

        prompt = self._build_prompt(available_roles, available_tools, requirements)
        inputs = self.tokeniser(prompt , return_tensors = "pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # N3: stop the moment a complete ACTION:/PAYLOAD: block exists,
        # instead of always running the full max_new_tokens budget and
        # discarding the wasted tail after the fact. Stateless (only reads
        # input_ids each call), so the same instance is reused across the
        # retry loop below.
        stopping_criteria = StoppingCriteriaList([
            _ActionPayloadStop(self.tokeniser, prompt_len, self._extract_first_balanced_object)
        ])

        generated_text = None
        for attempt in range(3):
            # Sampled from attempt 0, not just on retry. Greedy-first meant
            # the FIRST decide() of every agent was the one draw with no
            # escape from a locked-in trajectory -- and a first attempt that
            # degenerates has already cost a full 400-token generation by the
            # time _looks_degenerate catches it. The retry loop below stays
            # as a second net: each attempt is an independent draw, so a
            # degenerate one is now genuinely re-rolled rather than switching
            # decoding strategy mid-loop.
            gen_kwargs = dict(
                max_new_tokens=400,
                min_new_tokens=10,
                pad_token_id=self.tokeniser.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                stopping_criteria=stopping_criteria,
                # REVERTED (no_repeat_ngram_size=4 was here): the model
                # routes around a blocked n-gram rather than obeying it --
                # a blocked 4-gram comes back as a misspelling of itself
                # ("thermodynamicaly" -> "thermodynamally" ->
                # "thermodynamilly"), which is a fresh 4-gram every time, and
                # one agent degenerated all the way into Cyrillic symbol-soup
                # ("декор / деформа / деоторон"). Same failure mode as the
                # original revert, and the sampling retry below did not
                # rescue it. Repetition control is repetition_penalty alone
                # again; clause-level repeats get cleaned downstream in
                # _dedupe_repeated_sentences instead of by narrowing the
                # vocabulary the model is allowed to spell correctly.
            )
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            generated_tokens = outputs[0][prompt_len:]
            generated_text = self.tokeniser.decode(generated_tokens, skip_special_tokens=True).strip()

            if not self._looks_degenerate(generated_text):
                break
            print(
                f"  [decide() retry] attempt {attempt + 1} generation looked "
                f"degenerate (repeated sentence or bracket noise) -- "
                f"drawing a fresh sample."
            )

        self.thought_process += f"\n{strip_special_tokens(generated_text)}\n"

        action = "REPORT"  # Safe default fallback
        # Last block wins -- see _last_action_match.
        action_match = _last_action_match(generated_text)

        if action_match is not None and action_match.re is _ACTION_LINE_RE:
            print(f"  [decide() action-match] raw='{action_match.group(1)}' "
                  f"parsed='{action_match.group(1).strip().upper()}'")
            if action_match.group(1).strip().upper() == "DIE":
                print("  [decide() DIE DEBUG] Full prompt that produced this DIE:")
                print("  " + "-" * 60)
                print(prompt)
                print("  " + "-" * 60)
        else:
            print(f"  [decide() action-match] NO 'ACTION:' LINE FOUND. "
                  f"Tail of generated_text: {generated_text[-200:]!r}")
            if action_match is not None:
                print(
                    f"  [decide() action-match RECOVERED] no 'ACTION:' label, "
                    f"but generated_text has a line opening with bare keyword "
                    f"'{action_match.group(1).upper()}' -- treating the last "
                    f"such line as the intended action."
                )

        if action_match:
            parsed_action = action_match.group(1).strip().upper()
            if parsed_action in self.action_tokens:
                action = parsed_action
            else:
                # Minimum-length guard before fuzzy matching. difflib at
                # cutoff 0.75 happily turns a 3-character fragment into a
                # full action token -- 'SPA' matched 'SPAWN', and the
                # resulting spawn carried whatever half-formed text followed
                # it. A token this short is evidence the generation was cut
                # off or garbled, not evidence of which action was meant, so
                # it is not fuzzy-matched at all.
                normalized_action = None
                if len(parsed_action) >= 4:
                    normalized_action = normalize_identifier(
                        parsed_action, self.action_tokens, cutoff=0.75
                    )
                else:
                    print(
                        f"  [decide() action-match] action token "
                        f"'{parsed_action}' is shorter than 4 characters -- "
                        f"too short to fuzzy-match safely, not attempting."
                    )

                if normalized_action:
                    action = normalized_action
                else:
                    # Falls back to THINK, NOT REPORT. A parse failure is the
                    # one moment we know least about what the agent intended,
                    # and REPORT is the single action that ENDS the task --
                    # a garbled 'SPAWNSUBTASKS' becoming a REPORT is what
                    # triggered a root structural reject on a live run.
                    # THINK costs one cycle and discards nothing.
                    action = "THINK"
                    print(
                        f"  [decide() action-match] unrecognized action "
                        f"'{parsed_action}' -- no exact or fuzzy match against "
                        f"{self.action_tokens}; defaulting to THINK (one wasted "
                        f"cycle) rather than REPORT (ends the task on a parse "
                        f"failure)."
                    )

        payload = ""
        payload_match = _payload_match_for(generated_text, action_match)

        if payload_match:
            payload_raw = payload_match.group(1).strip()
            
            if payload_raw.startswith("{") and payload_raw.endswith("}"):
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    recovered = False
                    trimmed = payload_raw
                    for _ in range(2):
                        if not trimmed.endswith("}"):
                            break
                        trimmed = trimmed[:-1].rstrip()
                        try:
                            payload = json.loads(trimmed)
                            recovered = True
                            print(
                                f"  [decide() payload RECOVERED] stripped "
                                f"{len(payload_raw) - len(trimmed)} trailing "
                                f"char(s) to parse successfully."
                            )
                            break
                        except json.JSONDecodeError:
                            continue

                    if not recovered:
                        try:
                            candidate = ast.literal_eval(payload_raw)
                            if isinstance(candidate, dict):
                                payload = candidate
                                recovered = True
                                print(
                                    "  [decide() payload RECOVERED] parsed as "
                                    "Python dict literal (single/mixed quotes) "
                                    "instead of strict JSON."
                                )
                        except (ValueError, SyntaxError):
                            pass

                    if not recovered:
                        self.fail_reason = "JSONDecodeError: Payload was malformed."
                        print(f"  [decide() payload JSONDecodeError] raw={payload_raw[-200:]!r}")
                        payload = payload_raw
            elif payload_raw.startswith("{"):
                extracted = self._extract_first_balanced_object(payload_raw)
                recovered = False
                if extracted is not None:
                    try:
                        payload = json.loads(extracted)
                        recovered = True
                    except json.JSONDecodeError:
                        try:
                            candidate = ast.literal_eval(extracted)
                            if isinstance(candidate, dict):
                                payload = candidate
                                recovered = True
                        except (ValueError, SyntaxError):
                            pass
                    if recovered:
                        print(
                            f"  [decide() payload RECOVERED] extracted first "
                            f"complete object, discarding "
                            f"{len(payload_raw) - len(extracted)} trailing "
                            f"char(s) of extra commentary/rambling."
                        )

                if not recovered:
                    self.fail_reason = (
                        "Payload appears TRUNCATED (starts with '{' but never "
                        "reaches a balanced closing '}') -- likely ran out of "
                        "generation budget before finishing."
                    )
                    print(f"  [decide() payload TRUNCATED] raw={payload_raw[-200:]!r}")
                    payload = payload_raw
            else:
                # FIX (repetition-collapse bug): a plain-text REPORT/THINK
                # payload can itself degenerate into a repeated block
                # (confirmed via a real transcript -- "Verified: ..."
                # paragraph repeated near-verbatim before trailing off).
                # die() already deduped its own payload before this pass;
                # the plain-text path here never got the same treatment,
                # meaning the duplicated block would ride all the way into
                # a REPORT's result untouched. This is the source-side fix
                # complementing the orchestrator's own dedupe-before-judging
                # pass on handle_completion's result.
                # trim_closer_tail runs FIRST, on the raw text: it works
                # on sentence boundaries, and the dedupe's length cap can
                # append an ellipsis that changes where the last sentence
                # appears to end. Removes the "Ready. Done. Submitted."
                # tail that _looks_degenerate now also flags upstream --
                # both are needed, since the flag only triggers a retry and
                # a retry can come back with filler of its own.
                payload = _dedupe_repeated_sentences(
                    trim_closer_tail(payload_raw), max_chars=2000
                )

        if not payload and not (isinstance(payload, dict)):
            if generated_text.strip():
                payload = generated_text.strip()
                if action in ("SPAWN", "TOOL"):
                    # FIX (SPAWN/TOOL -> REPORT mislabel): this used to
                    # silently relabel action as REPORT here, handing the
                    # parent a "final result" that was actually just the raw
                    # ACTION:/PAYLOAD scaffolding of a decomposition the
                    # agent never got to make -- the task branch ends right
                    # there with that scaffolding text as its answer, and it
                    # can ride all the way into the synthesized final answer.
                    #
                    # Second half of the same fix: before degrading at all,
                    # try to find the object the model actually wrote but
                    # failed to label (see _recover_unlabelled_payload). Only
                    # if there is nothing of the right shape to recover is
                    # this a real parse failure -- same category as the
                    # unrecognized-action case above, and it gets the same
                    # treatment: THINK (one wasted cycle, discards nothing),
                    # not REPORT (which ends the task on a parse failure).
                    # Searched only after the chosen block, so an object
                    # from an earlier, abandoned draft is never recovered.
                    recovered_payload = self._recover_unlabelled_payload(
                        action,
                        generated_text[action_match.end():] if action_match else generated_text,
                    )
                    if recovered_payload is not None:
                        print(
                            f"  [decide() payload RECOVERED] no 'PAYLOAD:' "
                            f"label after 'ACTION: {action}', but a "
                            f"well-formed {action} payload object was found "
                            f"in the generation -- running the {action} "
                            f"instead of burning a THINK cycle."
                        )
                        payload = recovered_payload
                        self.fail_reason = None
                    else:
                        self.fail_reason = (
                            f"{action} Action Failed: no PAYLOAD was found after "
                            f"'ACTION: {action}'. Provide the required JSON "
                            f"payload on your next attempt."
                        )
                        action = "THINK"
            else:
                payload = "[No content generated -- decide() produced an empty response.]"

        self.cap_coerced_last_run = False
        if self.cycles_capped:
            final = self.final_actions
            if action not in final or (action == "SPAWN" and not self._is_spawn_payload(payload)):
                action, payload = self._coerce_final_action(action, payload)
                self.cap_coerced_last_run = True

        return action, payload

    @staticmethod
    def _is_spawn_payload(payload):
        """True only if the orchestrator would start at least one subtask
        from this payload. Shape alone is not enough at the cap: a subtasks
        list of strings or task-less dicts passes a shape check, starts
        nothing, and would be this agent's last action. Key spelling is not
        checked: the orchestrator fuzzy-repairs keys like "taask", and if
        nothing starts anyway it routes a capped spawner to failure
        (_release_spawner_if_nothing_started)."""
        def has_task(item):
            return isinstance(item, dict) and any(
                isinstance(value, str) and value.strip()
                for key, value in item.items() if key not in ("role", "label")
            )

        if not isinstance(payload, dict):
            return False
        subtasks = payload.get("subtasks")
        if isinstance(subtasks, list):
            return any(has_task(sub) for sub in subtasks)
        return "role" in payload and has_task(payload)

    def _coerce_final_action(self, action, payload):
        """
        Cycle cap reached but decide() still produced an action outside
        final_actions (or a parse failure, which lands on THINK above).
        Leaving the menu out of the prompt does not stop the model writing
        the word, and every fallback in decide() degrades to THINK -- so
        without this the cap is advisory and the loop continues.

        REPORT/DIE menu: converted to a REPORT of whatever prose there is,
        so the tier checks judge it; DIE only when there is no text at all.
        SPAWN/DIE menu (decomposer, no children yet): anything but a usable
        SPAWN is a DIE. Its prose is planning notes, never an answer.
        """
        # Whatever decide() said about the action it just discarded ("SPAWN
        # Action Failed ... on your next attempt") no longer applies, and a
        # WARN that keeps this agent running would otherwise show it a
        # verdict telling it to retry an action it does not have.
        self.fail_reason = None

        if "REPORT" in self.final_actions:
            text = payload if isinstance(payload, str) else ""
            text = strip_scaffolding_lines(strip_special_tokens(text)).strip()
            if not text:
                text = strip_scaffolding_lines(
                    strip_special_tokens(self.thought_process[-2000:])
                ).strip()
        else:
            text = ""
        forced = "REPORT" if text else "DIE"
        print(
            f"  [decide() CYCLE CAP] agent={self.agent_id} role={self.role} "
            f"cycles={self.non_terminal_cycles} menu={self.final_actions} -- "
            f"decide() chose {action}, which is closed or unusable; forcing {forced}."
        )
        if text:
            return "REPORT", _dedupe_repeated_sentences(trim_closer_tail(text), max_chars=2000)
        if self._decomposer_awaiting_first_spawn:
            reason = "without producing a usable SPAWN"
        else:
            reason = "with no usable result to report"
        return "DIE", (
            f"Cycle cap reached ({self.non_terminal_cycles} cycles) {reason}."
        )
        
    def execute(self, action, payload):
        '''just makes it to teh handler methods'''
        if action == "THINK":
            print(f"  [execute()] agent={self.agent_id} role={self.role} action=THINK "
                  f"(thought_process len={len(payload) if payload else 0}, not printed in full)")
        else:
            print(f"  [execute()] agent={self.agent_id} role={self.role} action={action!r} payload={payload!r}")
        if action == "THINK":
            pass
        elif action == "SPAWN":
            self.request_spawn(payload)
        elif action == "TOOL":
            self.request_tool(payload)
        elif action == "REPORT":
            self.report(payload)
        elif action == "DIE":
            self.die(payload)
        else:
            self.fail_reason = f"Execution failed: Unknown action '{action}'"

    def request_spawn(self, payload):
        """Creates a sub-agent to handle a sub-task, or several at once via
        a "subtasks" list."""
        if not isinstance(payload, dict):
            self.fail_reason = "SPAWN Action Failed: Payload must be a JSON object."
            print(f"  [request_spawn() REJECTED] payload={payload!r}")
            return

        subtasks = payload.get("subtasks")
        has_single = "role" in payload and "task" in payload

        if not subtasks and not has_single:
            self.fail_reason = (
                "SPAWN Action Failed: Payload must be a JSON object with "
                "'role' and 'task', OR a 'subtasks' list of such objects."
            )
            print(f"  [request_spawn() REJECTED] payload={payload!r}")
            return

        self.fail_reason = None

        event = Event(type="spawn_request", from_agent=self.agent_id)
        event.payload.update({
            "parent_id": self.agent_id,
            "role": payload.get("role"),
            "task_id": payload.get("task"),
            "subtasks": subtasks,
        })
        
        self.message.push(event)

        self.awaiting = "children"

    MAX_TOOL_ATTEMPTS_PER_AGENT = 15
    # How many recent tool-call signatures to remember for duplicate
    # detection. Long enough to catch an A -> B -> A alternation, short
    # enough that a genuinely iterative agent (edit, run, edit, run) is
    # not blocked from revisiting a call several steps later.
    TOOL_CALL_HISTORY = 5
    # Consecutive error-returning tool calls tolerated before TOOL is
    # refused outright. Deliberately far below MAX_TOOL_ATTEMPTS_PER_AGENT:
    # a call that keeps failing is not converging, and letting it run to 15
    # just burns energy producing the same SyntaxError fifteen times.
    MAX_CONSECUTIVE_TOOL_FAILURES = 3
    # Blocked-because-tools-are-disabled-this-run calls tolerated before TOOL
    # is refused outright. This is not a failing call -- it's a call that
    # was never going to reach a tool at all, so it needs its own, much
    # tighter bound rather than sharing MAX_TOOL_ATTEMPTS_PER_AGENT. Nothing
    # changes between attempt 1 and attempt 15 of the same deterministically
    # blocked call; two is enough to establish the agent is repeating it.
    MAX_DISABLED_TOOL_ATTEMPTS = 2

    @staticmethod
    def _normalize_args_for_signature(value):
        r"""
        Whitespace-insensitive view of a tool's args, for duplicate detection.

        The model regenerates the *same* call with cosmetically different
        whitespace constantly -- a leading newline, a trailing one, an extra
        blank line between statements. Byte-comparing the JSON treats those
        as distinct calls; normalizing them away treats them as what they
        are: the same call again.

        LEADING INDENTATION IS PRESERVED, deliberately. The obvious
        implementation -- re.sub(r"\s+", " ", value) -- collapses
        indentation too, and in Python indentation *is* the program:

            "if x:\n    print(1)\n    print(2)"   (both prints in the branch)
            "if x:\n    print(1)\nprint(2)"       (second print unconditional)

        both flatten to "if x: print(1) print(2)". Those are two different
        programs, and an agent that just fixed an IndentationError by
        re-indenting a block would have its corrected call rejected as a
        duplicate of the broken one -- turning this guard into the very
        loop it exists to break. So: per line, expand tabs, collapse runs
        of intra-line whitespace, drop trailing whitespace and blank lines,
        keep the indent.
        """
        if isinstance(value, str):
            normalized_lines = []
            for line in value.expandtabs(4).splitlines():
                stripped = line.strip()
                if not stripped:
                    continue  # blank lines never change a call's meaning
                indent = line[:len(line) - len(line.lstrip())]
                normalized_lines.append(indent + re.sub(r"\s+", " ", stripped))
            return "\n".join(normalized_lines)
        if isinstance(value, dict):
            return {k: Agent._normalize_args_for_signature(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Agent._normalize_args_for_signature(v) for v in value]
        return value

    @classmethod
    def _tool_call_signature(cls, tool_name, args):
        """(canonical_tool_name, whitespace-normalized args) as a hashable key."""
        normalized_args = cls._normalize_args_for_signature(args)
        try:
            args_key = json.dumps(normalized_args, sort_keys=True, default=str)
        except TypeError:
            args_key = str(normalized_args)
        return (tool_name, args_key)

    def _canonicalize_tool_name(self, raw_name):
        """
        Resolve a possibly-mangled tool name to its canonical spelling HERE,
        at the agent, rather than only inside ToolRegistry.execute().

        ToolRegistry already repairs "__run_code__" -> "run_code" before
        dispatch, but it repairs it privately: the agent's own duplicate
        history, the orchestrator's log line and the echoed result all kept
        carrying whatever the agent typed. So an agent alternating between
        two spellings of one tool saw two "different" calls both succeed,
        with nothing anywhere telling it its spelling was wrong. Canonical
        name in, canonical name stored/logged/echoed back out.

        Returns (canonical_name, correction_note_or_None).
        """
        available = self._default_available_tools()
        canonical = normalize_identifier(raw_name, available, cutoff=0.75)
        if canonical and canonical != raw_name:
            return canonical, (
                f"NOTE: you wrote the tool name as {raw_name!r}; it was "
                f"interpreted as '{canonical}'. Use '{canonical}' exactly "
                f"from now on."
            )
        # No confident match: leave it alone and let ToolRegistry.execute()
        # produce its "not found / did you mean" error, which is more
        # informative than anything that can be said here.
        return (canonical or raw_name), None

    @property
    def tool_circuit_open(self):
        """
        True once MAX_CONSECUTIVE_TOOL_FAILURES tool calls in a row have come
        back with an error. While open, no further TOOL action is accepted --
        the agent's only remaining moves are REPORT (with whatever it has) or
        DIE.
        """
        return (
            len(self.recent_tool_errors) >= self.MAX_CONSECUTIVE_TOOL_FAILURES
            and all(self.recent_tool_errors)
        )

    @property
    def disabled_tool_closed(self):
        """
        True once MAX_DISABLED_TOOL_ATTEMPTS TOOL calls in a row have been
        blocked because the requested tool wasn't in this run's available
        set. FIX (unwired cap): request_tool() used to count these and, on
        reaching the cap, tell the agent TOOL was "closed to you for the
        rest of this run" -- but nothing actually closed it. The prompt
        kept advertising TOOL every cycle exactly like tool_circuit_open's
        case does when it's wired in, so the agent kept re-selecting TOOL,
        the counter kept climbing past the cap, and the agent just re-read
        the same "closed" verdict forever. Mirrors tool_circuit_open so the
        cap is enforced the same way: once True, the prompt stops offering
        TOOL at all.
        """
        return self.disabled_tool_attempts >= self.MAX_DISABLED_TOOL_ATTEMPTS

    def _tools_blocked(self, available_tools=None):
        """True when no TOOL call this agent makes can possibly succeed.

        Three separate gates all end in the same place -- request_tool()
        refusing the call -- and every one of them used to be handled at a
        different subset of the prompt-building sites:

          - tool_circuit_open      (too many consecutive tool errors)
          - disabled_tool_closed   (too many out-of-set tool attempts)
          - run(available_tools=[])  tools disabled for the whole run

        The third was the one nothing checked consistently. An empty list
        is enforced at request_tool() via _run_available_tools, but the
        thinking seed still listed TOOL among "real actions that exist",
        and the format example still handed every executor a run_code
        call to copy. So agents did exactly what the prompt showed them:
        spend a full think() cycle composing a run_code/write_file call,
        get it bounced, then spend another full cycle recovering from the
        rejection. At least five agents did this in one run and one did it
        three times -- pure dead weight in the loop that is 85% of spend.

        Note the None/[] distinction, which is load-bearing: `None` means
        "no restriction" (the default, and what direct decide()/think()
        calls fall back to), `[]` means "explicitly nothing". Only the
        latter blocks.
        """
        if self.tool_circuit_open or self.disabled_tool_closed:
            return True
        if available_tools is None:
            available_tools = self._run_available_tools
        return available_tools is not None and len(available_tools) == 0

    def _accumulated_tool_errors_text(self):
        """The error text behind an open circuit, oldest first."""
        if not self.recent_tool_error_text:
            return "  (no error text was captured)"
        return "\n".join(
            f"  ({i}) {err}" for i, err in enumerate(self.recent_tool_error_text, 1)
        )

    def _push_tool_budget_failure(self):
        event = Event(type="failure_request", from_agent=self.agent_id)
        event.payload.update({
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "task_id": self.task_id,
            "role": self.role,
            "result": str(self.fail_reason),
        })
        self.message.push(event)

    def request_tool(self, payload):
        """Calls an external tool."""
        if not isinstance(payload, dict) or "tool_name" not in payload or "args" not in payload:
            self.fail_reason = "TOOL Action Failed: Payload must be a JSON object with 'tool_name' and 'args'."
            print(f"  [request_tool() REJECTED] payload={payload!r}")
            return

        tool_name, correction_note = self._canonicalize_tool_name(payload["tool_name"])
        args = payload["args"]

        # Checked, and returned from, BEFORE tool_call_count is touched: this
        # call was never going to reach a tool -- no tools exist for this run
        # at all -- so it costs the agent nothing toward
        # MAX_TOOL_ATTEMPTS_PER_AGENT. Closes TOOL itself after
        # MAX_DISABLED_TOOL_ATTEMPTS identical-in-kind blocks instead of
        # letting the agent grind all the way to the 15-call ceiling
        # re-discovering the same "no tools available" fact each time.
        if self._run_available_tools is not None and tool_name not in self._run_available_tools:
            self.disabled_tool_attempts += 1
            if self.disabled_tool_attempts >= self.MAX_DISABLED_TOOL_ATTEMPTS:
                self.fail_reason = (
                    f"TOOL is closed to you for the rest of this run: tools "
                    f"are disabled for this run and you have already tried "
                    f"{self.disabled_tool_attempts} times. Choose REPORT "
                    f"(submit your best result) or DIE (explain why this "
                    f"task cannot be completed without a tool)."
                )
            else:
                self.fail_reason = (
                    "TOOL is disabled for this run (no tools were made available "
                    f"to you). Requested tool: '{tool_name}'. Choose REPORT "
                    "(submit your best result) or DIE (explain why this task "
                    "cannot be completed without a tool)."
                )
            print(f"  [request_tool() BLOCKED - tools disabled for this run] "
                  f"{tool_name} (attempt {self.disabled_tool_attempts}/"
                  f"{self.MAX_DISABLED_TOOL_ATTEMPTS})")
            return

        self.node.tool_call_count = getattr(self.node, "tool_call_count", 0) + 1

        if self.tool_circuit_open:
            self.fail_reason = (
                f"TOOL is no longer available to you: your last "
                f"{self.MAX_CONSECUTIVE_TOOL_FAILURES} tool calls ALL failed:\n"
                f"{self._accumulated_tool_errors_text()}\n"
                f"Stop calling tools. Choose REPORT (submit your best "
                f"partial result and state plainly what you could not "
                f"verify) or DIE (explain why this task cannot be "
                f"completed). Those are your only two options now."
            )
            print(f"  [request_tool() BLOCKED - circuit breaker after "
                  f"{self.MAX_CONSECUTIVE_TOOL_FAILURES} consecutive failures] {tool_name}")
            return

        call_signature = self._tool_call_signature(tool_name, args)

        if call_signature in self.recent_tool_calls:
            self.fail_reason = (
                f"You already submitted this exact '{tool_name}' call -- same "
                f"arguments, ignoring whitespace -- within your last "
                f"{self.TOOL_CALL_HISTORY} tool calls, and already have its "
                f"result above (see 'Result of your most recent TOOL call'). "
                f"Reformatting the same call (adding a newline, re-indenting) "
                f"does not make it a new call. Use that result, try genuinely "
                f"different arguments, or move to REPORT/DIE."
            )
            print(f"  [request_tool() BLOCKED - duplicate within last "
                  f"{self.TOOL_CALL_HISTORY}] {call_signature}")
            if self.node.tool_call_count > self.MAX_TOOL_ATTEMPTS_PER_AGENT:
                self.fail_reason = (
                    f"Exceeded {self.MAX_TOOL_ATTEMPTS_PER_AGENT} TOOL attempts "
                    f"(including blocked duplicate ones) without "
                    f"converging to REPORT. Stop iterating and report your "
                    f"best result so far."
                )
                self._push_tool_budget_failure()
            return

        if self.node.tool_call_count > self.MAX_TOOL_ATTEMPTS_PER_AGENT:
            self.fail_reason = (
                f"Exceeded {self.MAX_TOOL_ATTEMPTS_PER_AGENT} TOOL attempts "
                f"(including malformed ones) without converging to REPORT. "
                f"Stop iterating and report your best result so far."
            )
            self._push_tool_budget_failure()
            return

        self.fail_reason = None
        self.recent_tool_calls.append(call_signature)
        self.pending_tool_name_correction = correction_note
        if correction_note:
            print(f"  [request_tool() NAME REPAIRED] {payload['tool_name']!r} -> {tool_name!r}")

        event = Event(type="tool_request", from_agent=self.agent_id)
        event.payload.update({
            "agent_id": self.agent_id,
            # Canonical name, so the orchestrator logs it, ToolRegistry sees
            # it already-resolved, and receive_tool_result echoes it back.
            "tool_name": tool_name,
            "args": args
        })

        self.message.push(event)

    def report(self, payload):
        """Submits the final result to the parent agent."""
        self.KV_Cache = None
        self.last_hidden_state = None
        
        event = Event(type="completion_request", from_agent=self.agent_id)
        event.payload.update({
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "task_id": self.task_id,
            "result": str(payload)
        })
        
        self.message.push(event)

    def die(self, payload):
        """Signals catastrophic failure to the parent."""
        self.KV_Cache = None
        self.last_hidden_state = None

        self.fail_reason = f"Previous attempt DIED with reason: {_dedupe_repeated_sentences(str(payload), max_chars=300)}"

        event = Event(type="failure_request", from_agent=self.agent_id)
        event.payload.update({
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "task_id": self.task_id,
            "role": self.role,
            "result": str(payload)  # Using result for the death reason
        })
        
        self.message.push(event)

    def receive_tool_result(self, tool_name: str, result_summary: str, success: bool):
        """
        Injects a TOOL action's outcome back into this agent's own context so
        its next think()/decide() cycle actually sees what happened.

        `tool_name` is the canonical name request_tool resolved and sent --
        not whatever the agent originally typed -- so a mangled spelling is
        visibly corrected here rather than silently accepted.

        Also feeds the consecutive-failure circuit breaker: this is the only
        place that learns whether a call actually worked.
        """
        result_summary = str(result_summary)
        if self.pending_tool_name_correction:
            result_summary = f"{self.pending_tool_name_correction}\n{result_summary}"
            self.pending_tool_name_correction = None

        self.thought_process += f"\n[TOOL RESULT - {tool_name}]: {result_summary}\n"
        self.last_tool_result = f"[TOOL RESULT - {tool_name}]: {result_summary}"

        self.recent_tool_errors.append(not success)

        if not success:
            self.recent_tool_error_text.append(
                _dedupe_repeated_sentences(f"{tool_name}: {result_summary}", max_chars=300)
            )
            self.fail_reason = f"Tool call to '{tool_name}' failed: {result_summary}"
            if self.tool_circuit_open:
                # Nth consecutive failure: say so now, in the same context the
                # agent reads before its next decide(), rather than waiting for
                # it to attempt an (N+1)th call and get bounced by request_tool.
                self.fail_reason = (
                    f"{self.MAX_CONSECUTIVE_TOOL_FAILURES} tool calls in a row "
                    f"have now failed:\n{self._accumulated_tool_errors_text()}\n"
                    f"TOOL is closed to you. REPORT your best partial result "
                    f"(saying what you could not verify) or DIE."
                )
                print(f"  [receive_tool_result() CIRCUIT OPEN] agent={self.agent_id} "
                      f"after {self.MAX_CONSECUTIVE_TOOL_FAILURES} consecutive tool failures")
        else:
            self.recent_tool_error_text.clear()
            self.fail_reason = None

    def run(self, available_roles=None, available_tools=None, requirements=None):
        """The core heartbeat coordinator for a single execution cycle."""
        # Record what this cycle was actually told is available so
        # request_tool() can enforce it below, not just advertise it in the
        # prompt. `None` here (the default) means "unrestricted" -- decide()
        # and think() already fall back to _default_available_tools() in
        # that case, so leave enforcement off to match.
        self._run_available_tools = available_tools

        self.cap_coerced_last_run = False
        if self.cycles_capped:
            # No think() once the cap is hit: its generation is the bulk of
            # a tick's think_tick cost, and it can only feed a THINK that
            # decide() is no longer allowed to return.
            action, payload = self.decide(available_roles, available_tools, requirements)
        else:
            action = self.think(available_roles, available_tools, requirements)

            if action == "FORCE_DECIDE" or (action in self.action_tokens and action != "THINK"):
                action, payload = self.decide(available_roles, available_tools, requirements)
            else:
                payload = self.thought_process

        self.execute(action, payload)

        # After execute(), not before: whether a SPAWN/TOOL actually handed
        # off is only known once its handler has accepted or rejected it. A
        # dispatched TOOL sets no `awaiting` (its result arrives before the
        # next tick) but clears fail_reason, while every refusal sets one; it
        # is not charged here because MAX_TOOL_ATTEMPTS_PER_AGENT and the
        # duplicate-call window already bound it, and charging it would
        # leave an executor three real tool calls.
        handed_off = self.awaiting is not None or (action == "TOOL" and self.fail_reason is None)
        if action not in ("REPORT", "DIE") and not handed_off:
            self.non_terminal_cycles += 1
        
        return action