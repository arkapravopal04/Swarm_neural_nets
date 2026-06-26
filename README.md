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



##task_graph.py

###TaskNode (The Dataclass): 

Acts as a data blueprint representing an individual task, tracking its metadata (ID, description, assigned agent, execution status, and result) alongside its structural relationships (lists of parent dependencies, children dependents, and a counter for how many incoming blocks remain).

###TaskGraph (The Class): 

Acts as the central manager that coordinates the overall workflow; it stores all tasks in a dictionary and handles adding new tasks, updating execution statuses, assigning agents, fetching ready-to-run tasks, and recursively cascading failures down through dependent tasks.


##event_queue.py

###Event (The Dataclass): 

Acts as a structured data container representing a single communication packet, capturing what happened (type), who sent it (from_agent), when it happened (timestamp), and any relevant data (payload).

###Messenger (The Class): 

Acts as a centralized mailbox or message broker that stores these communication packets privately, allowing agents to easily create and send messages, while letting an orchestrator peek at the message count or completely flush and retrieve the queue.
>thanks for reading, will try to be more consistent with updates