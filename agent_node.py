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
    normalize_identifier,
    strip_special_tokens,
    strip_scaffolding_lines,
)
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import ast
import re
import json
from collections import deque


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

    def __init__(self, tokenizer, prompt_len, extract_balanced_object, min_new_tokens=10):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.extract_balanced_object = extract_balanced_object
        self.min_new_tokens = min_new_tokens

    def __call__(self, input_ids, scores, **kwargs):
        new_ids = input_ids[0][self.prompt_len:]
        if len(new_ids) < self.min_new_tokens:
            return False

        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        if not re.search(r"ACTION:\s*[A-Za-z]+", text, re.IGNORECASE):
            return False
        payload_match = re.search(r"PAYLOAD:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
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
    MAX_TOTAL_THINK_TOKENS = 4096
    THINK_CYCLES_BEFORE_DECIDE = 1

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
        self.last_token_id = None
        self._total_generated = 0
        # actions
        self.action_tokens = ["THINK", "SPAWN", "TOOL", "REPORT", "DIE"]

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
        """Actual cap of number of tokens....add more when ever"""
        if self.role == "decomposer":
            return 128
        if self.role == "executor":
            return 256
        if self.role == "verifier":
            return 128
        return 256  # safety default for any role not yet in this map

    def _default_available_tools(self):
        try:
            return ToolRegistry.list_tools()
        except Exception:
            return []

    def _get_format_example(self):
        examples = {
            "decomposer": (
                'ACTION: SPAWN\n'
                'PAYLOAD: {"role": "executor", "task": "Implement the core algorithm"}'
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
        positive = examples.get(self.role, examples["executor"])
        if self.tool_circuit_open and positive.startswith("ACTION: TOOL"):
            # Showing a TOOL example while TOOL is refused is the single
            # strongest nudge back into the loop we are trying to break.
            positive = examples["verifier"]

        executor_spawn_example = (
            'Example of a valid response when YOUR task turns out to be too '
            'large for one agent (several independent sub-parts, not just '
            'one focused piece of work):\n'
            'ACTION: SPAWN\n'
            'PAYLOAD: {"role": "executor", "task": "Analyze coolant channel geometry independently of the flow rate calculation"}\n\n'
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
            '  {"label": "copy", "role": "executor", "task": "Draft the welcome email copy for new newsletter subscribers", "dependencies": []},\n'
            '  {"label": "palette", "role": "executor", "task": "Pick a color palette for the newsletter template", "dependencies": []}\n'
            ']}\n\n'
            'Example of a batch with a real SEQUENTIAL dependency (one '
            'piece genuinely cannot start until another\'s result exists) '
            '-- give the piece being depended on a "label" and list that '
            'label in the dependent piece\'s "dependencies":\n'
            'ACTION: SPAWN\n'
            'PAYLOAD: {"subtasks": [\n'
            '  {"label": "palette", "role": "executor", "task": "Pick a color palette for the newsletter template", "dependencies": []},\n'
            '  {"label": "copy", "role": "executor", "task": "Draft the welcome email copy", "dependencies": []},\n'
            '  {"role": "executor", "task": "Assemble the final template using the chosen palette and copy", "dependencies": ["palette", "copy"]}\n'
            ']}\n\n'
            'Example combining all three shapes in ONE batch -- some '
            'subtasks start immediately (empty dependencies), one needs '
            'another\'s result (sequential), and one final subtask needs '
            'several earlier ones done first (terminal):\n'
            'ACTION: SPAWN\n'
            'PAYLOAD: {"subtasks": [\n'
            '  {"label": "material", "role": "executor", "task": "Select the base material", "dependencies": []},\n'
            '  {"label": "geometry", "role": "executor", "task": "Define the geometry bounds", "dependencies": []},\n'
            '  {"label": "coating", "role": "executor", "task": "Size the protective coating using the selected material\'s properties", "dependencies": ["material"]},\n'
            '  {"role": "executor", "task": "Run the final combined check using material, geometry, and coating results", "dependencies": ["material", "geometry", "coating"]}\n'
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
        # turbine-blade colony straight out of the previous wording here,
        # killing four downstream agents on the contradiction). The
        # example only has to demonstrate SHAPE, so it carries no domain
        # content at all.
        negative = "<free-form reasoning with no ACTION: line -- placeholder, not a task>"

        tool_arg_reference = (
            "Tool argument reference (use the EXACT keys shown for each "
            "tool -- they are NOT interchangeable):\n"
            '- run_code: {"tool_name": "run_code", "args": {"code_string": "..."}}\n'
            '- verify_math: {"tool_name": "verify_math", "args": {"expression": "...", "mode": "computational"}}\n'
            '- safe_read_file: {"tool_name": "safe_read_file", "args": {"filepath": "..."}}\n'
            '- write_file: {"tool_name": "write_file", "args": {"filepath": "...", "content": "..."}}\n'
            '- query_dataframe: {"tool_name": "query_dataframe", "args": {"filepath": "...", "action": "summary"}}\n'
        )

        spawn_alternative = executor_spawn_example if self.role == "executor" else ""
        batch_alternative = decomposer_batch_example if self.role in ("decomposer", "executor") else ""

        return (
            f"Example of a VALID response:\n{positive}\n\n"
            f"{spawn_alternative}{batch_alternative}"
            f"Example of an INVALID response (do NOT do this -- free-form "
            f"reasoning with no ACTION: line is never an acceptable output):\n"
            f"{negative}\n\n"
            f"{tool_arg_reference}"
        )

    def _get_role_constraint_str(self):
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
                "pieces of work up front (e.g. a turbine blade problem "
                "obviously splits into material selection, structural/"
                "stress analysis, and cooling design -- none of these "
                "depend on each other), SPAWN all of them in ONE action "
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
                    "chunk of the problem (e.g. \"material selection\", "
                    "\"cooling design\", \"fatigue analysis\") -- do NOT spawn "
                    "\"executor\" children directly yourself. Each decomposer "
                    "child you create will handle breaking its chunk down "
                    "further into small, concrete pieces of work.\n"
                    "STAGE your children as a real dependency graph, not one "
                    "flat independent batch and not one single linear chain. "
                    "EVERY child you list MUST have an explicit "
                    "\"dependencies\" field -- never omit it, even for a "
                    "chunk with no prerequisites (give it \"dependencies\": "
                    "[] explicitly instead of leaving the key out). A "
                    "typical engineering problem has THREE kinds of "
                    "structure, and you should use whichever apply:\n"
                    "  1. PARALLEL stage: chunks with no dependency on each "
                    "other (e.g. \"material selection\" and \"geometry "
                    "bounds\" can both start immediately) -- give these "
                    "\"dependencies\": [] explicitly.\n"
                    "  2. SEQUENTIAL stage: a chunk that genuinely needs "
                    "another chunk's actual computed result as an input "
                    "(e.g. \"internal cooling layout\" needs the heat load "
                    "left over after \"TBC thickness\" is decided) -- give "
                    "it \"dependencies\": [\"<the label of the chunk it "
                    "needs>\"]. Its agent will automatically receive that "
                    "prerequisite's real result once it's done -- do not "
                    "invent placeholder numbers for something a dependency "
                    "will actually compute.\n"
                    "  3. TERMINAL stage: a final chunk that can only run "
                    "once several earlier chunks are ALL done (e.g. "
                    "\"fatigue life\" needs the compiled geometry, material "
                    "constants, AND thermal stresses from earlier stages) -- "
                    "list every one of those chunks in its \"dependencies\".\n"
                    "Use \"label\" on every chunk so later chunks can "
                    "reference it by name in their own \"dependencies\".\n"
                )
            else:
                tier_rule = (
                    "You are a NON-ROOT decomposer. SPAWN ONLY \"executor\" "
                    "children here, and make each one's task GRANULAR -- "
                    "small enough that it should take an executor no more "
                    "than 2-3 think cycles and at most one TOOL call to "
                    "finish (e.g. \"calculate X given these inputs\", "
                    "\"look up the density of Y\", \"print the result of "
                    "this formula\") -- never a whole sub-project like "
                    "\"design the cooling system\". If you can't picture an "
                    "executor finishing it almost immediately, break it "
                    "down further into more, smaller pieces instead.\n"
                )
            return base_rule + tier_rule
        return (
            "IMPORTANT: the constraints listed below apply to the PROJECT "
            "as a whole -- not every constraint necessarily applies to "
            "YOUR specific task. If a constraint is clearly outside what "
            "you were asked to do (e.g. a hardware/analog constraint for a "
            "pure software task), it belongs to a different specialized "
            "agent. Do your best on what's relevant to your task, and note "
            "any out-of-scope constraints as open items in your REPORT "
            "rather than treating them as a reason to DIE.\n"
            "IMPORTANT: your task may be larger than a single agent should "
            "handle directly. If it genuinely contains multiple substantial, "
            "independent pieces of work (e.g. \"design the cooling system\" "
            "actually requires separately analyzing channel geometry, "
            "coolant flow, AND manufacturing tolerances), do NOT try to "
            "SPAWN children yourself -- instead, ACTION: DIE with a PAYLOAD "
            "that starts with the exact phrase \"TASK TOO LARGE:\" followed "
            "by why. You will be replaced by a decomposer that breaks your "
            "task down properly. Use TOOL/REPORT when you can produce the "
            "answer yourself in one pass -- reserve this DIE for when the "
            "task is genuinely several tasks wearing one description.\n"
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
        role_constraint_str = self._get_role_constraint_str()
        requirements_str = (
            "Constraints you must satisfy:\n" + "\n".join(f"- {r}" for r in requirements) + "\n"
            if requirements else ""
        )
        tools_str = (
            f"Tools actually available to you: {available_tools}\n"
            if available_tools and not self.tool_circuit_open else ""
        )
        actions_str = (
            "Real actions that exist: THINK, SPAWN, REPORT, DIE (TOOL is "
            "closed to you after repeated tool failures -- do not attempt "
            "it).\n"
            if self.tool_circuit_open else
            "Real actions that exist: THINK, SPAWN, TOOL, REPORT, DIE (no others exist).\n"
        )
        ghost_str = f"Ghost Context: {self.ghost_context}\n" if self.ghost_context else ""
        fail_str = f"WARNING - Previous Action Failed: {self.fail_reason}\n" if self.fail_reason else ""
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
        fail_str = f"WARNING - Previous Action Failed: {self.fail_reason}\n" if self.fail_reason else ""
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
        format_example_str = self._get_format_example()
        role_constraint_str = self._get_role_constraint_str()

        # Circuit breaker open: TOOL is refused by request_tool anyway, so
        # advertising it here only invites another wasted decide() cycle that
        # gets bounced. Drop it from the menu and say why.
        if self.tool_circuit_open:
            tool_action_line = (
                "- TOOL   — UNAVAILABLE. Your last "
                f"{self.MAX_CONSECUTIVE_TOOL_FAILURES} tool calls all failed; "
                "further TOOL actions are refused. Do not attempt one.\n"
            )
        else:
            tool_action_line = (
                f"- TOOL   — call an external tool. Available tools: {available_tools}.\n"
            )

        prompt = f"""You are an AI agent in a colony of agents working together to solve problems.
Your Agent ID: {self.agent_id}
Your Role: {self.role}
{role_constraint_str}Your Task: {self.task}

{requirements_str}{ghost_str}{fail_str}{tool_result_str}{thoughts_str}
Available actions:
- THINK  — continue reasoning before acting (Use this to plan your next move).
- SPAWN  — create a sub-agent to handle a sub-task. Available roles: {available_roles}.
{tool_action_line}- REPORT — your task is complete, submit your final result to your parent.
- DIE    — you cannot complete this task, signal failure to your parent.

OUTPUT FORMAT INSTRUCTIONS:
You must respond with EXACTLY ONE action block in the following format:

ACTION: <One of the available actions>
PAYLOAD: <Depends on the action>
- If THINK: Provide your reasoning in plain text.
- If SPAWN: Provide a JSON object: {{"role": "chosen_role", "task": "specific task definition"}}
- If TOOL: Provide a JSON object: {{"tool_name": "name", "args": {{...}}}}
- If REPORT: Provide the final answer or result in plain text.
- If DIE: Provide the reason you cannot proceed.

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

        else:
            if self.last_token_id is not None:
                input_ids = torch.tensor([[self.last_token_id]], device=self.model.device)
            else:
                raise ValueError("KV_Cache is present, but missing last_token_id to continue generation.")

        hit_token_ceiling = False
        generated_chunk = ""

        with torch.no_grad():
            for step in range(max_tokens):
                if self._total_generated >= self.MAX_TOTAL_THINK_TOKENS:
                    hit_token_ceiling = True
                    break

                if step % 25 == 0:
                    print(f"  [think() heartbeat] agent={self.agent_id} role={self.role} "
                          f"step={step}/{max_tokens} total_generated={self._total_generated}")

                outputs = self.model(
                input_ids=input_ids,
                past_key_values=self.KV_Cache,
                use_cache=True
                )

                self.last_hidden_state = None

                self.KV_Cache = outputs.past_key_values
                logits = outputs.logits

                next_token_tensor = torch.argmax(logits[:, -1, :], dim=-1)
                self.last_token_id = next_token_tensor.item()
                self._total_generated += 1

                token_text = self.tokeniser.decode([self.last_token_id], skip_special_tokens=True)
                generated_chunk += token_text

                input_ids = next_token_tensor.unsqueeze(0)

        # Sanitize the whole cycle's output before storing it -- scaffolding
        # lines (ACTION:/PAYLOAD:/etc.) can only be recognized reliably once
        # a full line has accumulated, not token-by-token mid-line.
        self.thought_process += strip_scaffolding_lines(strip_special_tokens(generated_chunk))

        if self.think_cycle <= self.THINK_CYCLES_BEFORE_DECIDE and not hit_token_ceiling:
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
            use_sampling = attempt > 0
            gen_kwargs = dict(
                max_new_tokens=400,
                min_new_tokens=10,
                pad_token_id=self.tokeniser.eos_token_id,
                do_sample=use_sampling,
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
            if use_sampling:
                gen_kwargs["temperature"] = 0.7
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            generated_tokens = outputs[0][prompt_len:]
            generated_text = self.tokeniser.decode(generated_tokens, skip_special_tokens=True).strip()

            if not self._looks_degenerate(generated_text):
                break
            print(
                f"  [decide() retry] attempt {attempt + 1} generation looked "
                f"degenerate (repeated sentence or bracket noise) -- "
                f"retrying with sampling."
            )

        self.thought_process += f"\n{strip_special_tokens(generated_text)}\n"

        action = "REPORT"  # Safe default fallback
        action_match = re.search(r"ACTION:\s*([A-Za-z]+)", generated_text, re.IGNORECASE)

        if action_match:
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
            bare_action_match = re.match(
                r"\s*(THINK|SPAWN|TOOL|REPORT|DIE)\b", generated_text, re.IGNORECASE
            )
            if bare_action_match:
                action_match = bare_action_match
                print(
                    f"  [decide() action-match RECOVERED] no 'ACTION:' label, "
                    f"but generated_text leads with bare keyword "
                    f"'{bare_action_match.group(1).upper()}' -- treating as "
                    f"the intended action."
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
        payload_match = re.search(r"PAYLOAD:\s*(.*)", generated_text, re.DOTALL | re.IGNORECASE)
        
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
                payload = _dedupe_repeated_sentences(payload_raw, max_chars=2000)

        if not payload and not (isinstance(payload, dict)):
            if generated_text.strip():
                payload = generated_text.strip()
                if action in ("SPAWN", "TOOL"):
                    action = "REPORT"
            else:
                payload = "[No content generated -- decide() produced an empty response.]"

        return action, payload
        
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

        self.node.tool_call_count = getattr(self.node, "tool_call_count", 0) + 1

        tool_name, correction_note = self._canonicalize_tool_name(payload["tool_name"])
        args = payload["args"]

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
        action = self.think(available_roles, available_tools, requirements)
        
        if action == "FORCE_DECIDE" or (action in self.action_tokens and action != "THINK"):
            action, payload = self.decide(available_roles, available_tools, requirements)
        else:
            payload = self.thought_process

        self.execute(action, payload)
        
        return action