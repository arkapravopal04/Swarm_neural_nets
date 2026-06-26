'''
general tree with hashing ids
'''

import torch 
import torch.nn as nn
import numpy as np
# torch and numpy needed when goal_embedding is a real tensor
from dataclasses import dataclass, field 
import time  # Changed from 'from time import time' to avoid AttributeError in default_factory

'''
agent - node data class, used to store and generate the things from like an agent which ever is called
this should store
    {agent id
    role
    status
    parent_id: points up
    children: list, points down, grows dynamically
    task: what is this agent working on
    energy_spent:  how much has this agent consumed}

ColonyState : registry, energy budget, and goal embedding
register_agent()
update_status()
debit_energy()
get_snapshot()

'''

@dataclass
class AgentNode:
    agent_id: str
    role: str
    status: str
    parent_id: str
    task: str
    children: list = field(default_factory=list)
    energy_spent: int = 0
    ghost_context: dict = None # Added for smarter respawns
    last_active: float = field(default_factory=time.time)


class ColonyState:
    def __init__(self, initial_budget: int, goal_embedding):
        self.agents = {} #empty dict for all the agents that need to be here
        self.budget_remaining = initial_budget
        self.goal_embedding = goal_embedding
        self.results = {} # Added separated result storage

# needs to add in a new agent, and to append itself in the children of the parent...we have to make sure that the parent is registered first
    def register_agent(self, agent: AgentNode):
        self.agents[agent.agent_id] = agent
        parent = agent.parent_id
        if parent:
            if parent in self.agents:
                self.agents[parent].children.append(agent.agent_id)
            else:
                print(f"Warning: Parent agent {parent} not registered yet.")
        
# just updates the status and task if assigned
    def update_status(self, agent_id: str, new_status: str, new_task: str = None):
        agent = self.agents.get(agent_id)
        if agent:
            agent.status = new_status
            if new_task:
                agent.task = new_task 
        else:
            print(f"Warning: {agent_id} invalid for status update.")

# tracks the energy of the whole system
    def debit_energy(self, agent_id: str, amount: int):
        agent = self.agents.get(agent_id)
        if agent:
            agent.energy_spent += amount
            self.budget_remaining -= amount
        else:
            print(f"Warning: {agent_id} invalid for energy debit.")

# the orchestrator calls this one
    def can_spawn(self, cost: int) -> bool:
        return self.budget_remaining >= cost

    def get_snapshot(self):
        system_status = {
            "budget": self.budget_remaining,
            "goal": self.goal_embedding,
            "goal_shape": self.goal_embedding.shape if self.goal_embedding is not None else None,
            "agents": self.agents,
        }
        return system_status
    
    def get_agent(self, agent_id: str) -> AgentNode:
        return self.agents.get(agent_id)

    def store_result(self, task_id: str, result: any):
        """Stores a task result decoupled from the graph."""
        self.results[task_id] = result

    def extract_agent_ghost(self, agent_id: str) -> dict:
        """Extracts context, memory, and failure reason before agent death."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {}
        # Returns relevant state info needed to inform the respawn
        return {"last_task": agent.task, "energy_spent": agent.energy_spent, "role": agent.role}

    def unregister_agent(self, agent_id: str):
        """Deletes an agent permanently and cleans up parent children registries."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            parent_id = agent.parent_id
            if parent_id and parent_id in self.agents:
                if agent_id in self.agents[parent_id].children:
                    self.agents[parent_id].children.remove(agent_id)
            del self.agents[agent_id]

    def consolidate_idle_agents(self) -> int:
        """Finds idle agents, unregisters them, and returns some energy to the budget."""
        idle_agents = [aid for aid, a in self.agents.items() if a.status == "idle"]
        for aid in idle_agents:
            self.budget_remaining += 2  # Reclaim some energy
            self.unregister_agent(aid)
        return len(idle_agents)