'''
general tree with hashing ids
'''

from dataclasses import dataclass, field 
import time

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

        NOTE (flagged, not patched): this captures last_task, energy_spent,
        role, and fail_reason -- but not the agent's own ghost_context. If
        an agent that already had a ghost context (e.g. a second-generation
        respawn) dies again, that earlier context is dropped rather than
        carried forward or merged with the new fail_reason -- each death
        only ever hands the next respawn ITS OWN immediate failure reason,
        never the accumulated history across multiple respawns of the same
        task. Left as-is for now (matches the "Phase 1 honesty" scoping
        used elsewhere), but worth revisiting if repeated respawns on a
        stubborn task look like each one has amnesia about the ones before
        it. A fix would look like:
            "ghost_context": agent.ghost_context,
        added to the returned dict, with the caller (orchestrator._kill_and_respawn)
        deciding how to merge it with the new fail_reason rather than overwrite.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return {}
        return {
            "last_task": agent.task,
            "energy_spent": agent.energy_spent,
            "role": agent.role,
            "fail_reason": agent.fail_reason,
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