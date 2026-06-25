#Swarm


##colony_state.py
###AgentNode (Dataclass)
The blueprint for an individual agent's state. It tracks localized data and tree pointers:
agent_id: Unique identifier for the agent.
role: The agent's specialized function (e.g., commander, miner, scout).
status: Current operational state (e.g., active, idle, repairing).
parent_id: ID of the supervisor node (points up the tree).
children: Dynamically updated list of subordinate agent IDs (points down the tree).
task: The current specific objective the agent is executing.
energy_spent: Accumulated energy consumed by this specific node.

###ColonyState (Coordinator)
The global state manager that handles the registry database and global metrics:
Tracks total remaining initial_budget.
Holds the goal_embedding vector for agent alignment.
Maintains the central agents registry mapping IDs to AgentNode objects.


>thanks for reading, will try to be more consistent with updates