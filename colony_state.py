'''
general tree with hashing ids
'''

import torch 
import torch.nn as nn
import numpy as np
# torch and numpy needed when goal_embedding is a real tensor
from dataclasses import dataclass , field , asdict

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
    children: list = field(default_factory = list)
    energy_spent: int = 0



class ColonyState:
    def __init__(self , initial_budget: int , goal_embedding):
        self.agents = {} #empty dict for all the agents that need to be here
        self.budget_remaining = initial_budget
        self.goal_embedding = goal_embedding

# needs to add in a new agent, and to append itself in the children of the parent...we have to make sure that the parent is registered first
    def register_agent(self , agent: AgentNode):
        self.agents[agent.agent_id] = agent
        parent = agent.parent_id
        if parent:
            self.agents[parent].children.append(agent.agent_id)
        
# just updates the status and task if assigned
    def update_status(self, agent_id : str , new_status : str , new_task : str = None):
        agent = self.agents.get(agent_id)
        if agent:
            agent.status = new_status
            if new_task:
                agent.task = new_task 
        else:
            print(f"{agent_id} invalid")

# tracks the energy of the while system
    def debit_energy(self, agent_id : str , amount : int):
        agent = self.agents.get(agent_id)
        if agent:
            agent.energy_spent += amount
            self.budget_remaining -= amount
        else:
            print(f"{agent_id} invalid")

# the orchestratir calls this one
    def can_spawn(self, cost: int) -> bool:
        return self.budget_remaining >= cost

    def get_snapshot(self):
        system_status = {
            "budget" : self.budget_remaining,
            "goal" : self.goal_embedding,
            "goal_shape": self.goal_embedding.shape if self.goal_embedding is not None else None,
            "agents" : self.agents,
        }
        return system_status
    

