'''
general tree with hashing ids
'''

from dataclasses import dataclass, field 
import time

from text_utils import dedupe_and_cap as _dedupe_and_cap  # PATCH 55

@dataclass
class AgentNode:
    agent_id: str
    role: str
    status: str
    parent_id: str
    task: str
    children: list = field(default_factory=list)
    energy_spent: int = 0
    ghost_context: dict = None
    last_active: float = field(default_factory=time.time)
    fail_reason: str = None
    think_cycle: int = 0
    warning_count: int = 0
    task_id: str = ""
    requirements: list = field(default_factory=list)
    awaiting: str | None = None
    tool_call_count: int = 0
    generation: int = 0
    crash_count: int = 0
    has_spawned: bool = False
    # PATCH 8. The project goal text, threaded from TaskNode.goal_referent at
    # spawn, and ONLY when the requirements filter left this task with none.
    # Rendered in the prompt as background, never as a constraint -- an agent
    # with an empty constraints list previously had no project-level referent
    # anywhere in its context, which is the state every drifted run-4 subtask
    # in the ISBN subtree was run in.
    goal_referent: str | None = None
    # PATCH 29. The user's request verbatim (spec["raw_text"]), on EVERY
    # agent, executor and decomposer alike. The phaser's goal and
    # constraints paraphrase it, and a paraphrase can lose or bend a figure
    # ("12 plots" -> "12 gardens", 9,000 -> 9,067); the request cannot.
    request_text: str | None = None


class ColonyState:
    def __init__(self, initial_budget: int, goal_embedding):
        self.agents = {}
        self.budget_remaining = initial_budget
        self.goal_embedding = goal_embedding
        self.results = {}

        # Energy accounting (A1). Kept here rather than on Orchestrator
        # because debit_energy already lives here and this is the object
        # that owns budget_remaining -- the Orchestrator is reconstructed
        # per run, the ledger should describe the same object as the budget.
        #   energy_ledger:  category -> total debited
        #   energy_credits: category -> total refunded
        # starting_budget is the figure the ledger reconciles against;
        # Orchestrator.initialize_colony() overwrites budget_remaining from
        # the phaser spec and must re-stamp this alongside it.
        self.starting_budget = initial_budget
        self.energy_ledger = {}
        self.energy_credits = {}

        # task_id -> total debited to agents while they worked that task.
        # Keyed by task, not agent, so it keeps accumulating across every
        # agent a task burns through: a respawn starts a fresh Agent (and a
        # fresh AgentNode.energy_spent) but not a fresh entry here. Read by
        # Orchestrator's per-task energy ceiling.
        self.task_energy_spent = {}

        # Tier-3 verdict tally (Step 1). Deliberately NOT folded into
        # energy_ledger: that dict is summed against starting_budget in
        # _print_energy_report's reconciliation, so a row holding a COUNT
        # rather than an energy amount would break the check. Kept as its
        # own counter and printed as its own section.
        #   verdict_counts: "tier3_accept" / "tier3_reject" -> int
        self.verdict_counts = {}

    def register_agent(self, agent: AgentNode):
        self.agents[agent.agent_id] = agent
        parent = agent.parent_id
        if parent:
            if parent in self.agents:
                self.agents[parent].children.append(agent.agent_id)
            else:
                print(f"Warning: Parent agent {parent} not registered yet.")

    def update_status(self, agent_id: str, new_status: str, new_task: str = None):
        agent = self.agents.get(agent_id)
        if agent:
            agent.status = new_status
            if new_task:
                agent.task = new_task 
        else:
            print(f"Warning: {agent_id} invalid for status update.")

    def debit_energy(self, agent_id: str, amount: int, category: str = "uncategorized"):
        """Spends `amount` from the colony budget and records it under `category`.

        The budget is decremented WHETHER OR NOT the agent is still
        registered. Previously the else-branch printed a warning and paid
        nothing, which made a slice of the respawn cost invisible --
        _kill_and_respawn() calls unregister_agent() before some debits
        land, so those were free. A cost you cannot attribute to an agent
        is still a cost you paid; it is booked under the "orphaned" ledger
        key instead, and only the per-agent energy_spent attribution is
        skipped.

        `category` defaults so that a missed call site does not break --
        it shows up as an "uncategorized" row in terminate()'s ledger
        table instead.
        """
        agent = self.agents.get(agent_id)
        if agent:
            agent.energy_spent += amount
            self.budget_remaining -= amount
            self.energy_ledger[category] = self.energy_ledger.get(category, 0) + amount
            if agent.task_id:
                self.task_energy_spent[agent.task_id] = (
                    self.task_energy_spent.get(agent.task_id, 0) + amount
                )
        else:
            print(f"Warning: {agent_id} invalid for energy debit "
                  f"(category={category}) -- charging budget as 'orphaned'.")
            self.budget_remaining -= amount
            self.energy_ledger["orphaned"] = self.energy_ledger.get("orphaned", 0) + amount

    def record_verdict(self, name: str):
        """Increments a named verdict counter (e.g. "tier3_accept").

        Pure bookkeeping -- affects no budget and no agent state. Exists so
        the tier-3 accept/reject split is a measured number in the final
        report rather than something inferred from log lines.
        """
        self.verdict_counts[name] = self.verdict_counts.get(name, 0) + 1

    def credit_energy(self, amount: int, category: str = "uncategorized"):
        """Refunds `amount` to the colony budget and records it under `category`."""
        self.budget_remaining += amount
        self.energy_credits[category] = self.energy_credits.get(category, 0) + amount

    def can_spawn(self, cost: int) -> bool:
        return self.budget_remaining >= cost

    def get_snapshot(self):
        return {
            "budget": self.budget_remaining,
            "goal": self.goal_embedding,
            "goal_shape": self.goal_embedding.shape if self.goal_embedding is not None else None,
            "agents": self.agents,
        }
    
    def get_agent(self, agent_id: str) -> AgentNode:
        return self.agents.get(agent_id)

    def store_result(self, task_id: str, result: any):
        self.results[task_id] = result

    def extract_agent_ghost(self, agent_id: str) -> dict:
        """Extracts context, memory, and failure reason before agent death.

        PATCH 55 (1/2): a respawn now carries the background it needs.
          "base_context"     -- the dying agent's string ghost (project goal
                                line + prerequisite results / shared state),
                                or the base_context its own ghost dict carried;
                                None if there was none.
          "earlier_failures" -- reasons earlier agents on this task died,
                                oldest first, each capped to 300 chars via
                                text_utils.dedupe_and_cap, at most 3 kept,
                                each flattened to one line. An attempt that
                                ended with no fail_reason is recorded as
                                "previous attempt ended without a stated
                                reason".
                                Excludes this agent's own fail_reason, which
                                stays in "fail_reason".
        last_task / energy_spent / role are kept for compatibility only.
        The prompt renders this dict via Agent._ghost_context_str in
        agent_node.py (PATCH 55 part 2).
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return {}
        # PATCH 55: carry the base ghost context and failure history forward.
        prev = agent.ghost_context
        if isinstance(prev, str) and prev.strip():
            base_context = prev
        elif isinstance(prev, dict):
            base_context = prev.get("base_context")
        else:
            base_context = None
        if isinstance(prev, dict):
            items = list(prev.get("earlier_failures") or [])
            # PATCH 55 (review): prev being a dict means the dying agent was
            # itself a respawn, so a previous attempt did end -- record it
            # even when it died without a stated reason (e.g. watchdog kill).
            prev_reason = prev.get("fail_reason")
            prev_reason = " ".join(str(prev_reason).split()) if prev_reason else ""
            items.append(prev_reason or "previous attempt ended without a stated reason")
        else:
            items = []
        capped = []
        for x in items:
            # PATCH 55 (review): flatten to one line so a reason containing
            # newlines cannot break the rendered bullet list.
            c = _dedupe_and_cap(" ".join(str(x).split()), max_chars=300)
            # PATCH 55: dedupe_and_cap keeps a single over-long sentence
            # whole (soft cap); hard-cap that case so the ~300 limit holds.
            if c and len(c) > 300:
                c = c[:297].rstrip() + "..."
            if c and c.strip():
                capped.append(c)
        earlier_failures = capped[-3:]
        return {
            "last_task": agent.task,
            "energy_spent": agent.energy_spent,
            "role": agent.role,
            "fail_reason": agent.fail_reason,
            "base_context": base_context,
            "earlier_failures": earlier_failures,
        }

    def unregister_agent(self, agent_id: str):
        """Deletes an agent permanently and cleans up parent/children registries.

        FIX: previously any live children of the unregistered agent were left
        pointing at a parent_id key that no longer exists in self.agents --
        orphaned from tree traversal. They are now reparented one level up.
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            parent_id = agent.parent_id

            for child_id in list(agent.children):
                child = self.agents.get(child_id)
                if child:
                    child.parent_id = parent_id
                    if parent_id and parent_id in self.agents:
                        self.agents[parent_id].children.append(child_id)

            if parent_id and parent_id in self.agents:
                if agent_id in self.agents[parent_id].children:
                    self.agents[parent_id].children.remove(agent_id)
            del self.agents[agent_id]

    def consolidate_idle_agents(self) -> list:
        """Finds idle agents, unregisters them, and returns some energy to the budget.

        FIX NEEDED (flagged, NOT patched -- needs a design decision before
        touching): confirmed this is now a permanent no-op. Nothing in the
        system ever sets agent.status = "idle" anymore -- orchestrator.
        spawn_agent() sets "running" unconditionally at creation (the
        deliberate fix for the earlier "everything looks idle" bug), and
        update_status() is only ever called with "completed". So
        idle_agents below is always [], and merge_agents() -- the whole
        MERGE mechanism for surviving energy stress -- silently does
        nothing except print "Consolidated 0 idle agents" every time the
        colony hits the stressed threshold. That's a second energy-budget
        relief mechanism (alongside the fan-out cap and per-role costs)
        that is quietly disabled.

        The real invariant this should be checking is closer to "genuinely
        not doing anything" -- e.g. agent.awaiting is not None for longer
        than some threshold, or a task sitting at status==1 whose
        last_active is stale beyond a threshold that ISN'T already covered
        by handle_deadlock's own scaled-timeout logic. Deliberately left
        unpatched here rather than guessed at, since a wrong guess risks
        this mechanism fighting handle_deadlock's watchdog over the same
        agent (one flagging "stuck, respawn" while this flags "idle,
        cull") -- needs to be designed together with that logic, not
        patched in isolation.
        """
        idle_agents = [aid for aid, a in self.agents.items() if a.status == "idle"]
        for aid in idle_agents:
            # Routed through credit_energy so a stress-triggered merge shows
            # up as a line item instead of silently topping up the budget.
            self.credit_energy(2, category="consolidation")
            self.unregister_agent(aid)
        return idle_agents