'''
main brain of the project,

functions this should perform:
SPAWN: agent requests children, orchestrator creates them
PROMOTE: agent finished, orchestrator routes result to parent
EXECUTE: agent failed, orchestrator kills it, extracts ghost, respawns smarter
MERGE: energy low or deadlock, orchestrator consolidates agents

this also needs to know the dependices of the tasks
if task a can be run in parallel while task c needs task b it knows this and does exactly that


the file should:
'''



class Orchestrator:
    def __init__(self, colony_state, task_graph):
        # holds reference to colony state
        # holds the task graph with dependencies
        # holds a queue of incoming events from agents
        pass

    def handle_spawn_request(self, parent_id, subtask_spec):
        pass

    def handle_completion(self, agent_id, result):
        pass

    def handle_death(self, agent_id, ghost):
        pass

    def check_energy(self):
        pass

    def tick(self):
        # one cycle of the orchestrator loop
        # checks energy
        # processes pending events
        # checks task graph for newly unblocked tasks
        pass