'''
this is kind of the mail box between agents, for now its in strings 
but later on we shift to latent spaces

Event data class
type: what happened (spawn_request / completion / death / tool_request)
from_agent: who sent it
payload: the relevant data
timestamp: when it was sent
'''


from dataclasses import dataclass , field
import time

@dataclass
class Event:
    type: str
    from_agent: str
    timestamp: float = field(default_factory=time.time)
    payload: dict = field(default_factory=lambda: {
        "agent_id": None,
        "parent_id": None,
        "task_id": None,
        "role": None,
        "result": None,
        "tool_name": None,
        "args": None,
        "subtasks": None,
        "requirements": None,  # constraint strings threaded down from the spec
        "judge_verdict": None,  # judge.decide() result dict, when a failure_request
                                 # originates from a judge "execute" verdict rather
                                 # than a self-reported DIE -- lets ghost_extractor
                                 # distinguish TIER_1_CRASH/SEMANTIC_DRIFT from
                                 # SELF_REPORTED correctly.
    })

# this is going to be like the messenger between agents, agents can drop in thier messag in mailbox and out it goes ig
class Messenger:
    def __init__(self):
        self._events = [] #"_" convention so this is always private
    
# add event, easy
    def push(self, event : Event):
        self._events.append(event)

# empties the current mailbox and return a copy to orchestrator
    def drain(self):
        temp = self._events
        self._events = []
        return temp

# orchestrator wants to peep   
    def peek(self):
        return len(self._events)
    
# massive help to agents
    def push_event(self, event_type: str, from_agent: str, payload: dict = None):
        event = Event(
            type=event_type,
            from_agent=from_agent,
            payload=payload or {}
        )
        self.push(event)