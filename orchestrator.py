"""
main brain of the project,

functions this should perform:
SPAWN: agent requests children, orchestrator creates them
PROMOTE: agent finished, orchestrator routes result to parent
EXECUTE: agent failed, orchestrator kills it, extracts ghost, respawns smarter
MERGE: energy low or deadlock, orchestrator consolidates agents
WATCHDOG: monitors agent response latency and resolves systemic lockups
RUN & TERMINATE: manages system lifespan, captures run traces, and retrieves execution specs
"""

import uuid
import time
import re
import traceback
from typing import Optional, Dict, Any, List


# # Assuming these are your custom modules
from colony_state import ColonyState, AgentNode
from task_graph import TaskGraph, TaskNode
from event_queue import Messenger, Event
from agent_node import Agent, _dedupe_repeated_sentences
from tools import ToolRegistry
from text_utils import normalize_identifier
import ghost_extractor

class Orchestrator:
    def __init__(self, colony_state: ColonyState, task_graph: TaskGraph, messenger: Messenger,
                 phaser=None, judge=None, memory_store=None, synthesizer=None,
                 model=None, tokeniser=None, embed_model=None):
        self.colony = colony_state
        self.task_graph = task_graph
        self.messenger = messenger
        self.agent_counter = 0 
        self.root_task_id: Optional[str] = None
        self.running = True
        self.run_trace: Optional[Dict[str, Any]] = None

        # Injected shared components -- NONE of these were wired in previously.
        # phaser: Problem_Phaser instance (turns raw text into a spec + budget)
        # judge: Judge instance (tiered verification before promotion)
        # memory_store: MemoryStore instance (ghost + success indices)
        # synthesizer: Synthesizer instance (final English decode)
        # model/tokeniser: shared LLM, needed to actually construct live Agent
        #   objects -- previously the orchestrator only ever touched AgentNode
        #   (data), never the Agent (reasoning) class, so nothing ever ticked.
        # embed_model: shared SentenceTransformer, used to embed agent outputs
        #   before handing them to judge.decide() for semantic_check.
        self.phaser = phaser
        self.judge = judge
        self.memory_store = memory_store
        self.synthesizer = synthesizer
        self.model = model
        self.tokeniser = tokeniser
        self.embed_model = embed_model

        # Populated by initialize_colony() once Problem_Phaser has run.
        self.spec: Optional[Dict[str, Any]] = None

        # A1 observation only: task_id -> number of times _kill_and_respawn
        # has respawned an agent for that task. No cap is enforced here --
        # a task_id sitting at 14 respawns with no completion is the
        # signature of the respawn loop, and seeing that number is the
        # whole point before any control flow changes to react to it.
        self.respawn_counts: Dict[str, int] = {}

        # High-water mark of len(self.live_agents), sampled once per tick.
        self.peak_live_agents = 0

        # Live Agent objects (the think/decide/execute reasoning wrapper),
        # keyed by agent_id. Distinct from ColonyState.agents, which only
        # holds the AgentNode data records. Previously nothing populated
        # this at all -- the colony could spawn task nodes and agent
        # records forever with no reasoning loop ever actually running.
        self.live_agents: Dict[str, Agent] = {}

        # Hyperparameters
        self.energy_when_new_by_role = {
            "executor": 2,
            "verifier": 2,
            "decomposer": 4,
        }
        self.energy_when_new = 4  # fallback default for an unrecognized role
        self.energy_threshold_stress = 10
        self.energy_threshold_death = 5
        self.timeout_threshold = 30.0  # Seconds an agent can remain silent before being killed/probed

        # A crashing agent.run() is debited the same flat cost as spawning a
        # fresh agent of its role (energy_when_new_by_role) -- crashing isn't
        # free just because no tokens were produced. After this many
        # *consecutive* crashes (reset on any successful tick), the agent is
        # routed through the normal kill/respawn path (failure_request)
        # instead of being retried forever on the same broken state.
        self.MAX_CONSECUTIVE_CRASHES = 3

        self.SHORT_ANSWER_WORD_THRESHOLD = 12

        self.MAX_SUBTASKS_ROOT = 3
        self.MAX_SUBTASKS_NON_ROOT = 2

        # A decomposer's SPAWN batch over the fan-out cap used to get its
        # overflow crammed into one bundled child -- which, being several
        # unrelated items forced into one agent, reliably DIEd as "TASK TOO
        # LARGE" and got re-split anyway (a wasted respawn cycle, twice per
        # run at cap=2/3). Now the overflow beyond `cap` is parked here
        # (keyed by the decomposer's agent_id, already fully resolved --
        # see handle_spawn) and drained one item at a time in
        # _drain_pending_overflow, called whenever one of that decomposer's
        # children genuinely completes and frees a slot. Keeps the same
        # "at most `cap` concurrently in-flight children" invariant the cap
        # exists for, without discarding or force-merging anything.
        self.pending_overflow: Dict[str, list] = {}

        self.energy_map = {
            0: "fine",
            1: "stressed",
            2: "death"
        }

    def _generate_id(self) -> str:
        """Generates a unique 8-character ID for new agents."""
        return f"agent_{uuid.uuid4().hex[:8]}"

    def _generate_task_id(self) -> str:
        """Generates a unique ID for new task graph nodes."""
        return f"task_{uuid.uuid4().hex[:8]}"

    def _check_energy(self) -> int:
        """Returns the current energy budget of the colony."""
        return getattr(self.colony, 'budget_remaining', 0)
    
    def _get_energy_status(self, rem_energy: int) -> str:
        """Evaluates the energy levels safely against the thresholds."""
        if rem_energy <= self.energy_threshold_death:
            return "death"
        elif rem_energy <= self.energy_threshold_stress:
            return "stressed"
        return "fine"
    
    def _route_events(self, events: List[Event]):
        """Routes incoming events with their payloads and updates agent activity trackers."""
        now = time.time()
        for e in events:
            # Maintain active pulse tracking on every agent dispatching events
            if e.from_agent and e.from_agent != "orchestrator":
                agent = self.colony.get_agent(e.from_agent)
                if agent:
                    agent.last_active = now

            if e.type == "spawn_request":
                self.handle_spawn(e)
            elif e.type == "completion_request":
                self.handle_completion(e)
            elif e.type == "failure_request":
                self.handle_failure(e)
            elif e.type == "tool_request":     
                self.handle_tool_request(e)
            elif e.type == "parent_notification":
                self.handle_parent_notification(e)
            # "tool_response" is an orchestrator -> nowhere-in-particular
            # broadcast event; nothing needs to route it further since
            # handle_tool_request already delivers the result straight to the
            # live Agent via receive_tool_result().

    def _process_unblocked_tasks(self):
        """Finds tasks whose dependencies are met and spawns agents for them."""
        ready_tasks = self.task_graph.get_ready_tasks()
        for task in ready_tasks:
            # Spawn a new worker if the ready task doesn't have an agent assigned yet
            if getattr(task, 'agent_id', None) is None:
                self.spawn_agent(role=task.required_role, task_id=task.task_id)

    def _run_live_agents(self):
        """
        The missing heartbeat: drives Agent.think()/decide()/execute() for
        every agent whose task is currently running.
        """
        available_roles = ["decomposer", "executor", "verifier"]
        available_tools = ToolRegistry.list_tools()

        for agent_id, live_agent in list(self.live_agents.items()):
            task_id = live_agent.task_id
            task_node = self.task_graph.tasks.get(task_id)
            if task_node is None or task_node.status != 1:
                # Only tick agents actively bound to a running task. This
                # also naturally covers a task waiting on unmet dependencies
                # (status stays 0 until TaskGraph.assign_agent lets it become
                # 1 -- see the fix there) -- such an agent already exists in
                # live_agents (spawn_agent creates it eagerly) but is simply
                # never ticked until its task is actually ready to run.
                continue

            if live_agent.awaiting is not None:
                continue

            agent_node = self.colony.get_agent(agent_id)
            prev_len = len(live_agent.thought_process)
            try:
                live_agent.run(available_roles, available_tools, requirements=None)
            except Exception as e:
                print(f"Agent {agent_id} crashed during run(): {e}")
                print(traceback.format_exc())

                fail_reason = f"CRASH in run(): {e}"
                give_up = True

                if agent_node:
                    # A crash still counts as activity -- without this the
                    # deadlock watchdog sees a stale last_active and piles a
                    # second, conflicting kill/respawn on top of this one.
                    agent_node.last_active = time.time()

                    crash_cost = self.energy_when_new_by_role.get(live_agent.role, self.energy_when_new)
                    self.colony.debit_energy(agent_id, crash_cost, category="crash")

                    agent_node.fail_reason = fail_reason
                    agent_node.crash_count += 1
                    give_up = agent_node.crash_count >= self.MAX_CONSECUTIVE_CRASHES
                    crash_label = f"{agent_node.crash_count} times in a row"
                else:
                    # No colony node: there's nowhere to keep a crash
                    # count (the debit itself is no longer free -- it would
                    # book under "orphaned" -- but an uncounted crash-loop
                    # would still retry forever). Route it out on the first
                    # crash.
                    crash_label = "with no registered colony node"

                if give_up:
                    print(f"Agent {agent_id} crashed {crash_label} -- "
                          f"routing through kill/respawn instead of retrying every tick.")
                    self.messenger.push_event(
                        "failure_request",
                        agent_id,
                        {
                            "task_id": task_id,
                            "role": live_agent.role,
                            "parent_id": live_agent.parent_id,
                            "result": fail_reason,
                        }
                    )
                continue

            if agent_node:
                agent_node.last_active = time.time()
                agent_node.crash_count = 0

            new_chars = len(live_agent.thought_process) - prev_len
            cost = max(1, new_chars // 100)
            self.colony.debit_energy(agent_id, cost, category="think_tick")

    def initialize_colony(self, problem_spec: str):
        """System entry point. Bootstraps the first task and the root agent."""
        self.root_task_id = "root_task_0"

        if self.phaser is not None:
            spec = self.phaser.parse_problem(problem_spec)
            spec = self.phaser.estimate_complexity(spec)
        else:
            spec = {
                "raw_text": problem_spec,
                "goal": problem_spec,
                "goal_vector": None,
                "requirement": [],
                "domain": "General Discourse",
                "colony_budget": self.colony.budget_remaining or 100,
            }
        self.spec = spec

        self.colony.budget_remaining = spec.get("colony_budget", self.colony.budget_remaining)
        # The phaser's budget replaces whatever ColonyState was constructed
        # with, so the ledger's reconciliation baseline has to move with it.
        # Safe here: no debit has landed yet (the root spawn is below).
        self.colony.starting_budget = self.colony.budget_remaining
        self.colony.goal_embedding = spec.get("goal_vector")

        goal_text = spec.get("goal", problem_spec)
        print(f"[initialize_colony] goal ({len(goal_text)} chars): {goal_text}")

        root_node = TaskNode(
            task_id=self.root_task_id,
            description=goal_text,
            required_role="decomposer",
            requirements=spec.get("requirement", []),
        )
        self.task_graph.add_task(root_node)
        # Problem_Phaser dedupes/caps the goal before returning it in spec --
        # the root task description (what the judge prints on WARN) must be
        # that exact same string, not a re-derived one, since goal_vector
        # was embedded from it. A mismatch here means a future edit split
        # the description assignment from the phaser's cleaned goal again.
        assert self.task_graph.tasks[self.root_task_id].description == goal_text

        print(f"Colony initialized. Bootstrapping root task: {self.root_task_id}")
        self.spawn_agent(role="decomposer", task_id=self.root_task_id)

    def spawn_agent(self, role: str, task_id: str, parent_id: Optional[str] = None, ghost_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """SPAWN: Creates an agent, assigns it to a task, and deducts energy.

        NOTE on dependency gating: this still eagerly creates a live Agent
        and debits spawn energy even if the task isn't ready yet (unmet
        dependencies). That's an intentional, smaller-footprint choice --
        see TaskGraph.assign_agent's fix -- rather than deferring agent
        creation itself. _run_live_agents already refuses to tick an agent
        whose task_node.status != 1, so a dependency-gated agent simply sits
        idle in self.live_agents until TaskGraph flips its task to running.
        """
        spawn_cost = self.energy_when_new_by_role.get(role, self.energy_when_new)
        if not self.colony.can_spawn(spawn_cost):
            print(
                f"Cannot spawn {role} for task {task_id}: insufficient energy "
                f"(budget={self.colony.budget_remaining}, cost={spawn_cost})."
            )
            return None

        agent_id = self._generate_id()

        task_node = self.task_graph.tasks.get(task_id)
        description = task_node.description if task_node else task_id
        requirements = task_node.requirements if task_node else []

        if ghost_context is None and self.spec:
            overall_goal = self.spec.get("goal") or self.spec.get("raw_text")
            if overall_goal:
                overall_goal = _dedupe_repeated_sentences(overall_goal, max_chars=300)
                ghost_context = f"[Project goal] {overall_goal}"

        parent_node = self.colony.get_agent(parent_id) if parent_id else None
        generation = (parent_node.generation + 1) if parent_node else 0

        new_agent = AgentNode(
            agent_id=agent_id,
            role=role,
            status="running",
            parent_id=parent_id,
            task=description,
            task_id=task_id,
            ghost_context=ghost_context,
            requirements=requirements,
            generation=generation,
        )
        
        self.colony.register_agent(new_agent)
        self.task_graph.assign_agent(task_id, agent_id)
        
        # Debit the initialization energy cost from the colony budget
        self.colony.debit_energy(agent_id, spawn_cost, category="spawn")
        self.agent_counter += 1

        if self.model is not None and self.tokeniser is not None:
            live_agent = Agent(self.tokeniser, self.model, self.messenger, new_agent)
            self.live_agents[agent_id] = live_agent

        return agent_id

    _SUBTASK_KEY_SYNONYMS = {
        "depends_on": "dependencies",
        "deps": "dependencies",
        "desc": "description",
    }
    _SUBTASK_VALID_KEYS = ("role", "task", "label", "dependencies", "description")

    @classmethod
    def _normalize_subtask_keys(cls, sub: dict) -> dict:
        """Returns a copy of sub with keys cleaned up: whitespace stripped,
        known synonyms mapped to their canonical name, and anything left
        over (a typo like "rolle"/"taask") fuzzy-matched against the valid
        key set."""
        normalized = {}
        for key, value in sub.items():
            clean_key = key.strip() if isinstance(key, str) else key
            if clean_key in cls._SUBTASK_KEY_SYNONYMS:
                # Exact synonyms are checked first and win outright -- a
                # known synonym like "depends_on" must never fall through
                # to the fuzzy pass below and risk resolving to some other
                # valid key instead of its intended canonical name.
                normalized[cls._SUBTASK_KEY_SYNONYMS[clean_key]] = value
            elif isinstance(clean_key, str) and clean_key not in cls._SUBTASK_VALID_KEYS:
                fuzzy_key = normalize_identifier(clean_key, cls._SUBTASK_VALID_KEYS, cutoff=0.75)
                normalized[fuzzy_key or clean_key] = value
            else:
                normalized[clean_key] = value
        return normalized

    _REQ_FILTER_STOPWORDS = frozenset({
        "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at",
        "must", "should", "with", "without", "against", "as", "is", "are",
        "be", "by", "that", "this", "it", "its", "into", "than", "not",
        "specify", "design", "estimate", "under", "since", "base", "cannot",
        "survive", "remain", "via", "show", "just", "but", "final",
    })

    @classmethod
    def _significant_words(cls, text: str) -> set:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
        return {w for w in words if w not in cls._REQ_FILTER_STOPWORDS}

    def _filter_requirements_for_task(self, description: str, requirements: list) -> list:
        if not requirements:
            return []
        task_words = self._significant_words(description)
        if not task_words:
            return list(requirements)
        kept = [r for r in requirements if task_words & self._significant_words(r)]
        return kept if kept else list(requirements)

    _VALID_ROLES = ("decomposer", "executor", "verifier")

    def _enforce_child_role(self, requested_role: str, parent_id: Optional[str]) -> str:
        normalized_role = normalize_identifier(requested_role, self._VALID_ROLES, cutoff=0.75)
        if normalized_role:
            requested_role = normalized_role

        parent_node = self.colony.get_agent(parent_id) if parent_id else None
        if parent_node is None or parent_node.role != "decomposer":
            return requested_role
        if requested_role == "verifier":
            return requested_role
        return "decomposer" if parent_node.generation == 0 else "executor"

    def _build_dependency_context(self, dependencies: Optional[list]) -> str:
        if not dependencies:
            return ""
        lines = []
        for dep_id in dependencies:
            dep_task = self.task_graph.tasks.get(dep_id)
            if dep_task is None:
                continue
            dep_result = dep_task.result
            if dep_result is None:
                dep_result = self.colony.results.get(dep_id)
            if dep_result is None:
                continue
            snippet = str(dep_result).strip()
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            lines.append(f'- From completed step "{dep_task.description}": {snippet}')
        if not lines:
            return ""
        return (
            "Shared state from completed prerequisite steps -- USE THESE "
            "EXACT VALUES, do not re-derive or invent your own numbers for "
            "anything already computed here:\n" + "\n".join(lines) + "\n"
        )

    def _spawn_child_task(self, description: str, role: str, parent_id: Optional[str],
                           dependencies: Optional[list] = None, task_id: Optional[str] = None):
        if task_id is None:
            task_id = self._generate_task_id()
        full_requirements = self.spec.get("requirement", []) if self.spec else []

        if role == "decomposer":
            requirements = full_requirements
        else:
            requirements = self._filter_requirements_for_task(description, full_requirements)

        child_node = TaskNode(
            task_id=task_id,
            description=description,
            dependencies=dependencies or [],
            required_role=role,
            requirements=requirements,
        )
        # N2b: embed this task's own description once, at spawn time, so
        # judge.decide's tier-2 check has a target that actually matches
        # what this task's agent was asked to do -- not the colony's overall
        # goal, which is what every child was being scored against before.
        if self.embed_model is not None:
            try:
                child_node.description_embedding = self.embed_model.encode(
                    description, convert_to_numpy=True
                )
            except Exception as e:
                print(f"Warning: failed to embed task description for '{task_id}': {e}")
        self.task_graph.add_task(child_node)

        goal_text = (self.spec.get("goal") or self.spec.get("raw_text")) if self.spec else None
        if goal_text:
            goal_text = _dedupe_repeated_sentences(goal_text, max_chars=300)
        goal_line = (
            f"[Project goal -- for background only, NOT your task's scope: "
            f"your own task and constraints above are the ONLY things you "
            f"need to satisfy] {goal_text}\n"
        ) if goal_text else ""
        dep_context = self._build_dependency_context(dependencies)
        combined_context = (goal_line + dep_context) or None

        self.spawn_agent(role=role, task_id=task_id, parent_id=parent_id, ghost_context=combined_context)

    def _drain_pending_overflow(self, parent_id: Optional[str]):
        """
        Spawns exactly one queued overflow subtask (if any) for this
        decomposer -- called whenever one of its children genuinely
        completes and frees a fan-out slot. See pending_overflow's
        docstring in __init__ for why this replaces the old
        bundle-into-one-oversized-child approach.

        Draining exactly one per freed slot maintains the same "at most
        `cap` concurrently in-flight children" invariant the batch was
        capped to in the first place, without needing to separately track
        how many of this parent's children are currently live.
        """
        if not parent_id:
            return
        queue = self.pending_overflow.get(parent_id)
        if not queue:
            return
        next_spawn_kwargs = queue.pop(0)
        if not queue:
            del self.pending_overflow[parent_id]
        print(
            f"  [handle_spawn] fan-out slot freed under decomposer "
            f"{parent_id} -- draining next queued subtask "
            f"({len(queue)} still waiting)."
        )
        self._spawn_child_task(**next_spawn_kwargs)

    def handle_spawn(self, event: Event):
        """Triggered when an existing agent requests sub-agents (children)."""
        payload = event.payload
        parent_id = payload.get("parent_id")

        subtasks = payload.get("subtasks")

        if subtasks:
            parent_node = self.colony.get_agent(parent_id) if parent_id else None
            cap = None
            if parent_node is not None and parent_node.role == "decomposer":
                cap = (
                    self.MAX_SUBTASKS_ROOT if parent_node.generation == 0
                    else self.MAX_SUBTASKS_NON_ROOT
                )

            prepared = []
            label_to_id = {}
            for i, sub in enumerate(subtasks):
                # FIX (#1, crash-the-whole-run bug): a malformed subtask can
                # arrive as a bare string (or any non-dict) instead of the
                # expected {"role":..., "task":...} object. Without this
                # guard, _normalize_subtask_keys's sub.items() call throws
                # and takes the entire tick (and therefore the whole colony
                # run) down with it.
                if not isinstance(sub, dict):
                    print(f"Warning: subtask entry at index {i} is not a dict ({sub!r}), skipping.")
                    continue
                sub = self._normalize_subtask_keys(sub)
                description = sub.get("task", sub.get("description", ""))
                if not description:
                    print("Warning: subtask entry missing 'task'/'description', skipping.")
                    continue
                sub["role"] = self._enforce_child_role(sub.get("role", "worker"), parent_id)
                task_id = self._generate_task_id()
                label = sub.get("label")
                label_to_id[str(i)] = task_id  # positional reference always works
                if label:
                    label_to_id[str(label)] = task_id
                prepared.append((sub, description, task_id))

            # Fan-out cap: resolved against the FULL original batch's labels
            # above (so an item past the cap can still be a valid dependency
            # target/source), but only the first `cap` are spawned now --
            # the rest are queued (see pending_overflow's docstring in
            # __init__) instead of force-merged into one oversized child.
            if cap is not None and len(prepared) > cap:
                print(
                    f"  [handle_spawn] batch of {len(prepared)} subtasks "
                    f"from a generation-{parent_node.generation} decomposer "
                    f"exceeds the fan-out cap ({cap}) -- spawning the first "
                    f"{cap} now and queueing the remaining "
                    f"{len(prepared) - cap} to spawn one at a time as this "
                    f"decomposer's other children complete, instead of "
                    f"bundling them into one child that's reliably too "
                    f"large for a single agent."
                )

            for i, (sub, description, task_id) in enumerate(prepared):
                if "dependencies" in sub:
                    deps_raw = sub.get("dependencies") or []
                elif i > 0:
                    print(
                        f"  [handle_spawn] subtask '{description[:60]}...' "
                        f"omitted the required 'dependencies' field -- "
                        f"falling back to sequential-default (depends on "
                        f"the previous subtask in this batch)."
                    )
                    deps_raw = [str(i - 1)]
                else:
                    deps_raw = []

                resolved_deps = []
                for dep in deps_raw:
                    dep_id = label_to_id.get(str(dep))
                    if dep_id is None:
                        # FIX: a dependency label can arrive slightly mangled
                        # relative to how it was declared (e.g.
                        # "__coating_thicknes__" vs "__coating_thickness__").
                        # strip_chars="" here deliberately -- unlike an
                        # action/role/tool name, underscores in a label are
                        # part of its identity, not LLM decoration, so both
                        # sides are normalized (whitespace + casefold only)
                        # and compared as-is rather than trimmed.
                        normalized_dep = normalize_identifier(
                            str(dep), list(label_to_id.keys()), cutoff=0.75, strip_chars=""
                        )
                        if normalized_dep:
                            dep_id = label_to_id.get(normalized_dep)
                    if dep_id:
                        resolved_deps.append(dep_id)
                    else:
                        print(
                            f"Warning: subtask '{description[:60]}...' lists "
                            f"dependency '{dep}' which doesn't match any "
                            f"label/index in this SPAWN batch -- dropped."
                        )

                spawn_kwargs = dict(
                    description=description,
                    role=sub.get("role", "worker"),
                    parent_id=parent_id,
                    dependencies=resolved_deps,
                    task_id=task_id,
                )
                if cap is not None and i >= cap:
                    # Queued rather than spawned now. NOTE (known edge case,
                    # matching TaskGraph.add_task's own "depends on unknown
                    # task_id" limitation): if this item depends on ANOTHER
                    # still-queued overflow item rather than an
                    # already-kept/spawned one, its in_degree will never
                    # resolve until that sibling is drained first -- fine
                    # for the common case this replaces (independent
                    # overflow items), not handled for a genuine chain of
                    # dependent overflow items.
                    self.pending_overflow.setdefault(parent_id, []).append(spawn_kwargs)
                else:
                    self._spawn_child_task(**spawn_kwargs)
            return

        # Single-subtask case.
        description = payload.get("task_id") or payload.get("description")
        role = payload.get("role", "worker")
        role = self._enforce_child_role(role, parent_id)

        if not description:
            print("Warning: Spawn request ignored due to missing task description in payload.")
            return

        self._spawn_child_task(
            description=description,
            role=role,
            parent_id=parent_id,
            dependencies=payload.get("dependencies", []),
        )

    def handle_parent_notification(self, event: Event):
        """
        Delivers a completed child's result back into the parent's own live
        reasoning context, and clears the parent's `awaiting` gate so
        orchestrator._run_live_agents() resumes ticking it.
        """
        payload = event.payload
        parent_id = payload.get("parent_id")
        child_id = payload.get("child_id")
        result = payload.get("result")

        live_parent = self.live_agents.get(parent_id)
        if live_parent is None:
            return

        # FIX (#4, free-energy bug): this injection happens outside any
        # _run_live_agents() tick, so the per-tick energy debit (which only
        # measures thought_process growth across a single run() call) never
        # sees it -- confirmed via repro that injecting 5,000 characters
        # cost 0 energy. Capped at the same 400-char length
        # _build_dependency_context uses (consistency with the rest of the
        # "shared state" mechanism) and now explicitly debited using the
        # same chars-per-100 proxy used everywhere else in the energy model.
        result_str = str(result).strip()
        if len(result_str) > 400:
            result_str = result_str[:400] + "..."
        injected_text = f"\n[CHILD RESULT - {child_id}]: {result_str}\n"
        live_parent.thought_process += injected_text

        injection_cost = max(1, len(injected_text) // 100)
        self.colony.debit_energy(parent_id, injection_cost, category="injection")

        live_parent.awaiting = None

    def handle_completion(self, event: Event):
        """PROMOTE: Routes completed results to parents, unblocks tasks, or triggers synthesizer."""
        payload = event.payload
        agent_id = event.from_agent
        task_id = payload.get("task_id")
        result = payload.get("result")

        if not task_id:
            print(f"Warning: completion_request from {agent_id} missing task_id, ignoring.")
            return

        # FIX (repetition-collapse bug): a REPORT payload can degenerate into
        # the same sentence/paragraph repeated 2+ times before trailing off
        # mid-sentence (confirmed via a real run -- greedy decoding locking
        # onto its own recent output). Previously only DIE text, ghost-context
        # goal lines, and EXECUTE critique text got this treatment; a
        # successful REPORT's result never did, so a degenerate-but-"passing"
        # result could get judged, stored, and promoted to a parent verbatim.
        # Deduped here, before the judge ever sees it, so the stored task
        # result and whatever gets handed to a parent via parent_notification
        # are both the cleaned version.
        if result:
            result = _dedupe_repeated_sentences(str(result), max_chars=2000)

        agent_node = self.colony.get_agent(agent_id)

        # FIX (root-acceptance bug): a generation-0 decomposer's contract is
        # to SPAWN, not to answer the goal itself. A REPORT on the root task
        # with no completed children means it skipped decomposition entirely
        # -- reject it structurally, before the judge (and its expensive
        # tier-3 critique) ever sees it, and force a respawn instead of
        # letting it fall through as a "result".
        if (task_id == self.root_task_id and agent_node is not None
                and agent_node.role == "decomposer" and agent_node.generation == 0):
            has_completed_child = False
            for child_id in agent_node.children:
                child_agent = self.colony.get_agent(child_id)
                if child_agent is None:
                    continue
                child_task = self.task_graph.tasks.get(child_agent.task_id)
                if child_task is not None and child_task.status == 2:
                    has_completed_child = True
                    break
            if not has_completed_child:
                print(f"REJECT (structural) on {agent_id}/{task_id}: root decomposer "
                      f"REPORTed with no completed children -- a generation-0 "
                      f"decomposer must SPAWN, not answer the goal directly.")
                agent_node.fail_reason = (
                    "Previous attempt was REJECTED: you REPORTed a final answer "
                    "directly instead of decomposing the task. As the ROOT "
                    "decomposer (generation 0) your job is to SPAWN subtasks, "
                    "not answer the question yourself."
                )
                self._kill_and_respawn(
                    agent_id, task_id, agent_node.role, agent_node.parent_id,
                    verdict={"verdict": "execute", "reason": agent_node.fail_reason},
                )
                return

        verdict = {"verdict": "promote", "reason": "no judge configured"}
        if self.judge is not None and agent_node is not None:
            output_embedding = None
            is_root = task_id == self.root_task_id

            # N2b: score against this task's OWN description, not the
            # colony's overall goal -- a subtask several decomposition
            # levels down was never going to read as semantically similar
            # to the whole project goal even when it's doing exactly the
            # right thing. The root task has no separate description to
            # target (its description already IS the goal), so it alone
            # keeps using colony.goal_embedding.
            if is_root:
                target_embedding = self.colony.goal_embedding
            else:
                completing_task_node = self.task_graph.tasks.get(task_id)
                target_embedding = (
                    completing_task_node.description_embedding
                    if completing_task_node is not None else None
                )

            word_count = len(str(result).split())
            skip_tier2 = (not is_root) and word_count <= self.SHORT_ANSWER_WORD_THRESHOLD
            print(f"  [judge-bypass-check] word_count={word_count} "
                  f"threshold={self.SHORT_ANSWER_WORD_THRESHOLD} "
                  f"will_skip_tier2={skip_tier2} "
                  f"result={str(result)!r}")
            if (self.embed_model is not None and target_embedding is not None
                    and not skip_tier2):
                try:
                    output_embedding = self.embed_model.encode(str(result), convert_to_numpy=True)
                except Exception as e:
                    print(f"Warning: failed to embed output for judging: {e}")
            verdict = self.judge.decide(
                agent_node,
                output=result,
                output_type="text",
                output_embedding=output_embedding,
                target_embedding=target_embedding,
                needs_deep_check=True,
            )

            if verdict.get("tier") == 3:
                critique_cost = max(1, len(str(verdict.get("reason", ""))) // 100)
                self.colony.debit_energy(agent_id, critique_cost, category="tier3_critique")
                print(f"  [judge tier-3 cost] agent={agent_id} deep_critique "
                      f"debited {critique_cost} energy (previously untracked).")

        if verdict["verdict"] == "warn":
            print(f"Judge WARN on {agent_id}/{task_id}: {verdict['reason']}")
            return

        if verdict["verdict"] == "execute":
            print(f"Judge EXECUTE on {agent_id}/{task_id}: {verdict['reason']}")

            if agent_node is not None:
                critique_text = str(verdict.get("reason", ""))[:400]
                critique_text = _dedupe_repeated_sentences(critique_text)
                agent_node.fail_reason = (
                    f"Previous attempt was REJECTED by review with this specific "
                    f"feedback: {critique_text}"
                )

            role = getattr(agent_node, "role", "worker") if agent_node else "worker"
            parent_id = getattr(agent_node, "parent_id", None) if agent_node else None
            self._kill_and_respawn(agent_id, task_id, role, parent_id, verdict=verdict)
            return

        print(
            f"SUBTASK COMPLETE: {agent_id}/{task_id} -- "
            f"\"{self.task_graph.tasks.get(task_id).description if self.task_graph.tasks.get(task_id) else '?'}\" "
            f"-- verdict={verdict['verdict']} ({verdict.get('reason', 'no judge configured')})"
        )
        self.task_graph.complete_task(task_id)
        self.colony.store_result(task_id, result)

        task_node = self.task_graph.tasks.get(task_id)
        if task_node:
            task_node.result = result

        # Root completion check
        if task_id == self.root_task_id:
            print("Root task completed! Triggering Synthesizer...")
            if self.synthesizer is not None:
                goal_text = self.spec.get("goal", "") if self.spec else ""
                try:
                    final_answer = self.synthesizer.run(
                        self.colony, self.task_graph, goal_text, root_task_id=self.root_task_id
                    )
                except Exception:
                    print(f"Warning: Synthesizer failed on root completion -- "
                          f"falling back to the raw agent result instead of crashing.\n"
                          f"{traceback.format_exc()}")
                    final_answer = result
            else:
                final_answer = result
            self.colony.results["final_spec"] = final_answer
            self.colony.update_status(agent_id, "completed")
            return

        # FIX (smaller #1): the old fallback
        # `... if agent_node else payload.get("parent_id")` could never fire
        # -- the next line already required `agent_node` truthy regardless.
        # Simplified to be honest about that instead of carrying dead code.
        # If you actually want the payload fallback to work, loosen the
        # `if agent_node and parent_id:` gate below to `if parent_id:`.
        parent_id = getattr(agent_node, 'parent_id', None) if agent_node else None

        # This child just genuinely completed -- a real fan-out slot under
        # its decomposer parent opened up (unlike _kill_and_respawn, which
        # reuses the same task_id/slot). Drain one queued overflow subtask
        # into it, if that parent has any waiting.
        self._drain_pending_overflow(parent_id)

        if agent_node and parent_id:
            self.messenger.push_event(
                "parent_notification",
                "orchestrator",
                {"parent_id": parent_id, "child_id": agent_id, "result": result}
            )

        self.colony.update_status(agent_id, "completed")

        if self.memory_store is not None and task_node is not None:
            try:
                self.memory_store.write(
                    "success", task_node.description, {"result": result, "task_id": task_id}
                )
            except Exception:
                print(f"Warning: failed writing success cache for {task_id}:\n"
                      f"{traceback.format_exc()}")

    def handle_failure(self, event: Event):
        """EXECUTE: Harvest context, kill agent, and respawn a smarter version."""
        payload = event.payload
        agent_id = event.from_agent
        task_id = payload.get("task_id")
        role = payload.get("role", "worker")
        parent_id = payload.get("parent_id")
        verdict = payload.get("judge_verdict")

        die_text = payload.get("result", "") or ""
        if verdict is None and role == "executor" and die_text.startswith("TASK TOO LARGE:"):
            print(f"Agent {agent_id} reported its task as too large -- "
                  f"respawning as a decomposer for the same task instead of an executor.")
            role = "decomposer"

        self._kill_and_respawn(agent_id, task_id, role, parent_id, verdict=verdict)

    def _kill_and_respawn(self, agent_id: str, task_id: Optional[str], role: str,
                           parent_id: Optional[str], verdict: Optional[Dict[str, Any]] = None):
        """
        Shared kill/ghost/respawn path used by both a self-reported DIE
        (handle_failure) and a judge-triggered EXECUTE (handle_completion).
        """
        live_agent = self.live_agents.pop(agent_id, None)

        if self.memory_store is not None:
            try:
                ghost_source = live_agent if live_agent is not None else self.colony.get_agent(agent_id)
                if ghost_source is not None:
                    record = ghost_extractor.extract(ghost_source, verdict)
                    self.memory_store.write("ghost", record.get("task", task_id or ""), record)
            except Exception:
                print(f"Warning: failed writing ghost record for {agent_id}:\n"
                      f"{traceback.format_exc()}")

        if live_agent is not None:
            live_agent.KV_Cache = None
            live_agent.last_hidden_state = None

        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        ghost_context = self.colony.extract_agent_ghost(agent_id)
        self.colony.unregister_agent(agent_id)

        if not task_id:
            print(f"Agent {agent_id} failed with no task_id -- cannot respawn.")
            return

        task_node = self.task_graph.tasks.get(task_id)
        if self.memory_store is not None and task_node is not None:
            cached = self.memory_store.get_success_cache(task_node.description)
            if cached is not None:
                print(f"Success cache hit for task {task_id} -- skipping respawn.")
                self.task_graph.complete_task(task_id)
                cached_result = cached.get("result")
                self.colony.store_result(task_id, cached_result)
                task_node.result = cached_result

                # FIX (#3, silent-stall bug): this branch previously marked
                # the task complete and stored the result but never pushed
                # parent_notification, unlike handle_completion's normal
                # success path. If this child was the only (or last) thing
                # a parent was `awaiting`, the parent never resumed -- and
                # since awaiting agents are exempt from the deadlock
                # watchdog, nothing else caught it either. Mirrors the
                # normal-path notification exactly.
                if parent_id:
                    self.messenger.push_event(
                        "parent_notification",
                        "orchestrator",
                        {"parent_id": parent_id, "child_id": agent_id, "result": cached_result}
                    )
                return

        print(f"Agent {agent_id} failed. Respawning {role} with ghost context.")
        self.respawn_counts[task_id] = self.respawn_counts.get(task_id, 0) + 1
        self.spawn_agent(role=role, task_id=task_id, parent_id=parent_id, ghost_context=ghost_context)

    @staticmethod
    def _summarize_tool_result(result) -> str:
        """
        The single string an agent actually gets to read about its tool call.

        The previous `result.get("data") or result.get("message") or str(result)`
        had two holes that both ended with the agent unable to act:
          - a successful run whose script printed nothing has data == "",
            which is falsy, so the agent was handed the raw dict repr;
          - an error carried its "reason" ("script_crash", "timeout",
            "output_overflow") only inside that repr, so a condensed
            TracebackSummarizer line arrived with no marker saying it WAS a
            failure -- easy for the model to read as ordinary output.
        Status is now stated explicitly and the reason is kept alongside the
        message.
        """
        if not isinstance(result, dict):
            return str(result)

        status = result.get("status")
        if status == "success":
            data = result.get("data")
            if data is None:
                data = result.get("message")
            if data is None or str(data).strip() == "":
                return ("SUCCESS: the tool ran without error but produced no "
                        "output. If you expected output, your code did not "
                        "print anything -- add a print() of the value you "
                        "need.")
            return f"SUCCESS: {data}"

        reason = result.get("reason")
        message = result.get("message") or result.get("data") or str(result)
        reason_str = f" ({reason})" if reason else ""
        return f"ERROR{reason_str}: {message}"

    def handle_tool_request(self, event: Event):
        """Handles external tool execution requests from agents."""
        payload = event.payload
        agent_id = event.from_agent
        tool_name = payload.get("tool_name")
        args = payload.get("args", {}) or {}
        domain = self.spec.get("domain", "General Discourse") if self.spec else "General Discourse"

        print(f"Executing tool '{tool_name}' for agent {agent_id}...")

        result = ToolRegistry.execute(tool_name, args, domain=domain, agent_id=agent_id)

        output_type = "text"
        if tool_name == "run_code":
            output_type = "code_result"
        elif tool_name == "verify_math":
            output_type = "math_result"

        fast_result = {"pass": True, "error": None}
        if self.judge is not None:
            fast_result = self.judge.fast_check(result, output_type)

        summary = self._summarize_tool_result(result)

        # Whether the call succeeded is decided by the tool's own status
        # first, and only then narrowed by the judge. Reading it off
        # fast_result alone had two failure modes that both told the agent an
        # error was a success -- clearing its fail_reason and (since the
        # consecutive-failure circuit breaker counts these) letting a call
        # that fails every time run to the 15-attempt ceiling:
        #   - with no judge configured, fast_result was hardcoded to pass;
        #   - for any tool other than run_code/verify_math, output_type is
        #     "text", and fast_check's text branch only asks "is this
        #     non-empty?" -- an error dict is non-empty, so a failed
        #     write_file/safe_read_file/query_dataframe passed.
        tool_ok = not (isinstance(result, dict) and result.get("status") == "error")
        success = tool_ok and bool(fast_result["pass"])

        live_agent = self.live_agents.get(agent_id)
        if live_agent is not None:
            live_agent.receive_tool_result(tool_name, summary, success)

        self.messenger.push_event(
            "tool_response",
            "orchestrator",
            {"agent_id": agent_id, "result": result, "tool_name": tool_name}
        )

    def merge_agents(self):
        """MERGE: Consolidation mechanism to survive low-energy bottlenecks."""
        print("System strain detected! Consolidating idle / overlapping agents...")
        culled_ids = self.colony.consolidate_idle_agents()

        for agent_id in culled_ids:
            live_agent = self.live_agents.pop(agent_id, None)
            if live_agent is not None:
                live_agent.KV_Cache = None
                live_agent.last_hidden_state = None

        print(f"Consolidated {len(culled_ids)} idle agents and reclaimed energy.")

    def handle_deadlock(self):
        """
        Runs systemic watchdog checks to detect and resolve system bottlenecks or silent agents.
        """
        ready_tasks = self.task_graph.get_ready_tasks()
        root_task = self.task_graph.tasks.get(self.root_task_id)
        root_incomplete = root_task is not None and root_task.status != 2

        if self.messenger.peek() == 0 and len(ready_tasks) == 0 and root_incomplete:
            now = time.time()
            running_tasks = [t for t in self.task_graph.tasks.values() if t.status == 1]
            
            active_agents = []
            stuck_agents = []

            tickable_agent_count = sum(
                1 for t in running_tasks
                if t.agent_id and self.colony.get_agent(t.agent_id)
                and self.colony.get_agent(t.agent_id).awaiting is None
            )

            for task in running_tasks:
                if task.agent_id:
                    agent = self.colony.get_agent(task.agent_id)
                    if agent:
                        if agent.awaiting is not None:
                            continue

                        effective_timeout = self.timeout_threshold * max(1, tickable_agent_count)
                        silence_duration = now - agent.last_active
                        if silence_duration > effective_timeout:
                            stuck_agents.append((agent, task))
                        else:
                            active_agents.append(agent)

            if not stuck_agents and len(active_agents) > 0:
                return

            print("System deadlock confirmed! Investigating state nodes...")

            if stuck_agents:
                for agent, task in stuck_agents:
                    print(f"Deadlock Watchdog: Agent {agent.agent_id} on Task {task.task_id} "
                          f"is non-responsive for {now - agent.last_active:.2f}s. Forcing respawn.")
                    # FIX (#2, wrong-agent-killed bug): this previously
                    # pushed with from_agent="orchestrator" and put the real
                    # target in payload["agent_id"] -- but handle_failure
                    # reads its target from event.from_agent, not from the
                    # payload. That meant the watchdog was always calling
                    # _kill_and_respawn("orchestrator", ...): a no-op pop/
                    # unregister (nothing registered under that id), so the
                    # actually-stuck agent was never removed from
                    # live_agents/the colony, and a duplicate agent got
                    # spawned onto the same task alongside it. Setting
                    # from_agent to the real agent id matches the
                    # self-reported-DIE convention handle_failure already
                    # expects.
                    self.messenger.push_event(
                        "failure_request",
                        agent.agent_id,
                        {
                            "task_id": task.task_id,
                            "role": agent.role,
                            "parent_id": agent.parent_id
                        }
                    )
            else:
                pending_unassigned_tasks = [
                    t for t in self.task_graph.tasks.values() 
                    if t.status == 0 and t.agent_id is None
                ]
                if pending_unassigned_tasks:
                    print(f"Deadlock Watchdog: Recovered {len(pending_unassigned_tasks)} unassigned pending tasks. Spawning workers...")
                    for task in pending_unassigned_tasks:
                        self.spawn_agent(role=task.required_role, task_id=task.task_id)
                else:
                    print("Deadlock Watchdog: System is stable. No actionable stalls detected.")

    def tick(self) -> bool:
        """Processes a single heartbeat of the orchestrator loop."""
        rem_energy = self._check_energy()
        status = self._get_energy_status(rem_energy)
        self.peak_live_agents = max(self.peak_live_agents, len(self.live_agents))

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                print(f"[VRAM:tick] allocated={allocated:.1f}MB reserved={reserved:.1f}MB "
                      f"live_agents={len(self.live_agents)}")
        except Exception:
            pass
        
        if status == "death":
            print(f"Critical failure. Colony energy depleted ({rem_energy} remaining).")
            return False
        elif status == "stressed":
            self.merge_agents()

        self.handle_deadlock()

        events = self.messenger.drain()
        self._route_events(events)
        self._process_unblocked_tasks()

        self._run_live_agents()
        
        root_task = self.task_graph.tasks.get(self.root_task_id)
        if root_task and root_task.status == 2:
            return False
    
        return True

    def _print_energy_report(self):
        """Prints the energy ledger, credits, reconciliation, and respawn counts.

        A1 deliverable: answers "where did the budget go, and which tasks
        cycled?" at the end of every run. Read-only -- it must never be able
        to change the outcome of a run, hence the broad guard at the bottom.
        """
        try:
            ledger = dict(getattr(self.colony, "energy_ledger", {}) or {})
            credits = dict(getattr(self.colony, "energy_credits", {}) or {})
            start = getattr(self.colony, "starting_budget", None)
            remaining = getattr(self.colony, "budget_remaining", 0)

            total_debits = sum(ledger.values())
            total_credits = sum(credits.values())

            print("\n" + "=" * 62)
            print("ENERGY LEDGER (where the budget went)")
            print("=" * 62)

            if not ledger:
                print("  (no energy was debited this run)")
            else:
                print(f"  {'category':<20}{'total':>10}{'% of start':>14}")
                for category, total in sorted(ledger.items(), key=lambda kv: -kv[1]):
                    pct = f"{(100.0 * total / start):.1f}%" if start else "n/a"
                    print(f"  {category:<20}{total:>10}{pct:>14}")
                print(f"  {'-' * 44}")
                print(f"  {'TOTAL DEBITED':<20}{total_debits:>10}")

            print("\n  CREDITS (refunds)")
            if not credits:
                print("    (none)")
            else:
                for category, total in sorted(credits.items(), key=lambda kv: -kv[1]):
                    print(f"    {category:<18}{total:>10}")
                print(f"    {'TOTAL CREDITED':<18}{total_credits:>10}")

            # These three must agree. If they don't, there is an energy path
            # writing budget_remaining directly instead of going through
            # debit_energy/credit_energy -- find it before tuning anything.
            expected = (start - total_debits + total_credits) if start is not None else None
            print("\n  RECONCILIATION")
            print(f"    starting budget            : {start}")
            print(f"    start - debits + credits   : {expected}")
            print(f"    colony.budget_remaining    : {remaining}")
            if expected is None:
                print("    [WARN] no starting_budget recorded -- cannot reconcile.")
            elif expected != remaining:
                print(f"    [MISMATCH] delta = {remaining - expected} -- there is an "
                      f"energy path bypassing debit_energy/credit_energy.")
            else:
                print("    [OK] ledger reconciles with the remaining budget.")

            print("\n  RESPAWNS (top 10 by count)")
            respawns = getattr(self, "respawn_counts", {}) or {}
            if not respawns:
                print("    (no respawns this run)")
            else:
                for task_id, count in sorted(respawns.items(), key=lambda kv: -kv[1])[:10]:
                    task = self.task_graph.tasks.get(task_id)
                    status = task.status if task is not None else "?"
                    description = (task.description or "") if task is not None else "<unknown task>"
                    if len(description) > 60:
                        description = description[:57] + "..."
                    print(f"    {count:>3}x  status={status}  {task_id}  {description}")

            live_count = getattr(self, "_live_agents_at_terminate", len(self.live_agents))
            print(f"\n  live_agents at terminate : {live_count}")
            print(f"  peak live_agents        : {getattr(self, 'peak_live_agents', 'n/a')}")
            print("=" * 62 + "\n")
        except Exception:
            print(f"Warning: failed printing energy report:\n{traceback.format_exc()}")

    def terminate(self) -> Any:
        """
        1. Prints system resolution (Success or Energy Death).
        2. Captures system execution trace using a snapshot.
        3. Clears execution loop flags.
        4. Isolates and returns the highest quality result spec.
        """
        self.running = False
        self.run_trace = self.colony.get_snapshot()

        # Captured before the teardown below empties the dict -- the energy
        # report is printed after it and would otherwise always read 0.
        self._live_agents_at_terminate = len(self.live_agents)

        for live_agent in self.live_agents.values():
            live_agent.KV_Cache = None
            live_agent.last_hidden_state = None
        self.live_agents.clear()

        if self.memory_store is not None:
            try:
                self.memory_store.save_ghosts()
            except Exception as e:
                print(f"Warning: failed to save ghost index on terminate: {e}")

        root_task = self.task_graph.tasks.get(self.root_task_id)
        is_successful = root_task is not None and root_task.status == 2

        if is_successful:
            print("System Status: TERMINATED [SUCCESS]")
            print("The root task successfully completed and synthesized.")
        else:
            rem_energy = self._check_energy()
            print("System Status: TERMINATED [ENERGY DEATH]")
            print(f"The colony depleted its energy allocation. Remaining budget: {rem_energy}")

        # Printed on BOTH exit paths on purpose: a successful run's ledger is
        # the baseline the failing runs get compared against.
        self._print_energy_report()

        best_result = self.colony.results.get("final_spec")

        if not best_result and not is_successful and self.synthesizer is not None:
            try:
                partial_results = self.synthesizer.collect_results(
                    self.colony, self.task_graph, root_task_id=self.root_task_id
                )
            except Exception:
                print(f"Warning: failed collecting partial results for synthesis:\n"
                      f"{traceback.format_exc()}")
                partial_results = []

            if partial_results:
                goal_text = self.spec.get("goal", "") if self.spec else ""
                try:
                    partial_answer = self.synthesizer.format_output(partial_results, goal_text)
                    best_result = (
                        f"[PARTIAL RESULT -- the colony ran out of its energy budget "
                        f"before finishing every subtask. {len(partial_results)} "
                        f"subtask(s) completed and are synthesized below; anything "
                        f"not mentioned was not reached.]\n\n{partial_answer}"
                    )
                except Exception as e:
                    print(f"Warning: failed synthesizing partial result: {e}")

        if not best_result and root_task:
            best_result = getattr(root_task, 'result', None)

        if not best_result:
            if not is_successful:
                rem_energy = self._check_energy()
                best_result = (
                    f"The colony was unable to produce any completed subtask "
                    f"before exhausting its energy budget (remaining: "
                    f"{rem_energy}). No partial result is available to "
                    f"synthesize. Consider increasing the energy budget or "
                    f"breaking the problem into a more explicit, narrower "
                    f"prompt."
                )
            else:
                best_result = self.colony.results

        return best_result

    def run(self, problem_spec: str) -> Any:
        """
        1. Bootstraps the root node space with initial specifications.
        2. Loops ticking cycles until a stop signal or death triggers.
        3. Terminates the cycle and retrieves results.
        """
        self.running = True
        self.initialize_colony(problem_spec)

        while self.running:
            tick_success = self.tick()
            if not tick_success:
                break

        return self.terminate()