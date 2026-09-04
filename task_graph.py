'''
important script as this is the actual task space of the model
the models get to split the task up and so on here


TaskNode:
task_id
description
status (pending / running / complete / failed)
agent_id (who is working on it, None if unassigned)
dependencies (list of task_ids that must complete first)
dependents (list of task_ids waiting on this one)
in_degree (count of incomplete dependencies)
result (populated when complete)


graph_class
add_task(task): add a node, calculate its in_degree
complete_task(task_id): mark done, decrement dependents in_degree return list of newly unblocked task_ids
fail_task(task_id): mark failed, handle dependents
get_ready_tasks(): return all tasks with in_degree == 0 and status == pending
assign_agent(task_id, agent_id): link a task to the agent working it
get_snapshot(): full graph state for the orchestrator
'''

from dataclasses import dataclass , field
from typing import Any


@dataclass
class TaskNode:
    task_id: str
    description : str
    agent_id : str | None = None #only when the task has one incoming or outgoing or both...is an agent assigned
    # if the task needs to be split up, we dont assign an agent to it, also agent is assigned only after sort
    status : int = 0 # 0 for pending , 1 for running , 2 for completed , 3 for failed
    dependencies : list = field(default_factory = list)
    dependents : list = field(default_factory=list)
    in_degree : int = 0
    result : Any = None  # NOTE: was annotated `bool` but actually holds arbitrary
                          # str/dict results (synthesizer.py reads it as text) --
                          # `Any` reflects real usage. Must keep a type annotation or
                          # this silently stops being a dataclass field at all.
    required_role : str = "worker"  # orchestrator.py constructs TaskNode(required_role=...)
                                     # in several places; this field was previously missing
                                     # entirely, which would raise a TypeError at runtime.
    requirements : list = field(default_factory=list)  # constraint strings threaded down
                                                         # from Problem_Phaser's spec so a
                                                         # subtask can carry its relevant
                                                         # slice of the parent's constraints
    description_embedding : Any = None  # this task's own description, embedded once at
                                         # spawn time (orchestrator._spawn_child_task) --
                                         # judge.decide's tier-2 target for this task's
                                         # agent. Left None on the root task, which has no
                                         # separate description to compare against (its
                                         # description IS the colony goal already covered
                                         # by colony.goal_embedding).


class TaskGraph:
    def __init__(self):
        self.tasks = {}
    
# just adds a new task, update dependencies of dependents...by just appending this there
    def add_task(self, task: TaskNode):
        self.tasks[task.task_id] = task
        task.in_degree = len(task.dependencies)
        
        for parent_id in task.dependencies:
            parent_task = self.tasks.get(parent_id)
            if parent_task:
                parent_task.dependents.append(task.task_id)
            else:
                print(
                    f"WARNING: Task '{task.task_id}' depends on unknown "
                    f"task_id '{parent_id}' -- this dependency will never "
                    f"resolve and '{task.task_id}' may never become ready."
                )

        # Lightweight cycle check: walk dependents from this task and see if we
        # ever loop back to task.task_id. A cyclic graph means in_degree can
        # never reach 0 for something in the cycle, which would hang the colony
        # forever with no diagnostic -- surface it immediately instead.
        if self._creates_cycle(task.task_id):
            print(
                f"WARNING: Task '{task.task_id}' introduces a dependency cycle. "
                f"This task graph may never fully resolve."
            )

    def _creates_cycle(self, start_id: str) -> bool:
        """BFS over dependents from start_id; True if we loop back to start_id."""
        visited = set()
        queue = list(self.tasks.get(start_id).dependents) if start_id in self.tasks else []
        while queue:
            current = queue.pop(0)
            if current == start_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            current_task = self.tasks.get(current)
            if current_task:
                queue.extend(current_task.dependents)
        return False

# should return the list of tasks that we can begin now , should also be able to change the task status to 1 of the task we just completed
# when we loop thru the unblocked we reduce their in dregree as one of them is done
    def complete_task(self, task_id : str):
        task = self.tasks.get(task_id)
        if not task:
            print(f"{task_id} invalid")
            return []

        task.status = 2
        newly_unblocked = []
        for dependent_id in task.dependents:
            dep_task = self.tasks.get(dependent_id)
            if dep_task:
                dep_task.in_degree -= 1  
                if dep_task.in_degree == 0:
                    newly_unblocked.append(dependent_id)
                    # FIX (#5, dependency-gating bug): a dependent task that
                    # already has an agent_id (assigned eagerly at spawn time
                    # -- see assign_agent's fix below) needs its status
                    # actually flipped to "running" the moment it becomes
                    # unblocked, or its already-created live Agent will sit
                    # forever un-ticked (orchestrator._run_live_agents only
                    # ticks status==1 tasks). Previously assign_agent set
                    # status=1 unconditionally at spawn time regardless of
                    # in_degree, so this was never needed; now that
                    # assign_agent withholds status=1 until in_degree==0,
                    # this is the other half of that fix -- the transition
                    # has to happen somewhere once the last dependency
                    # actually completes, and this is the natural place
                    # since it's already iterating dependents whose
                    # in_degree just changed.
                    if dep_task.agent_id is not None:
                        dep_task.status = 1
                    
        return newly_unblocked
    

# if a task was worked on and failed it needs to be broken down into more atomic tasks, will be done by some other function...this 
# one needs to somehow handle the consequences of faliure 
    def fail_task(self , task_id : str):
        task = self.tasks.get(task_id)
        if not task:
            print(f"{task_id} invalid")
            return []
        
        task.status = 3
        doomed_tasks = []
        queue = list(task.dependents) #apparently this copies and doesnt pop the whole thing

        while queue:
            current_id = queue.pop(0)
            current_task = self.tasks.get(current_id)
            if current_task and current_task.status != 3:
                current_task.status = 3
                doomed_tasks.append(current_id)
                queue.extend(current_task.dependents)
        return doomed_tasks
    

# return all tasks with in_degree 0 and status 0
    def get_ready_tasks(self):
        ready_tasks = []
        for task_id in self.tasks:
            task = self.tasks[task_id]
            if task.in_degree == 0 and task.status == 0:
                ready_tasks.append(task)
        return ready_tasks
    
# needs to assign agent to the task , add change the status and link the agent to the task
    def assign_agent(self, task_id : str , agent_id : str = None):
        """
        FIX (#5, dependency-gating bug -- confirmed): this previously set
        status=1 ("running") unconditionally the instant an agent was
        assigned, with zero check on in_degree. Since orchestrator.
        _spawn_child_task calls spawn_agent() -> assign_agent() immediately
        after creating every child (dependencies or not), a child declared
        to depend on a sibling from the same SPAWN batch started reasoning
        the same tick it was created, regardless of whether that sibling
        had finished -- meaning _build_dependency_context's "shared state"
        block (built specifically for the TBC-Thickness -> Cooling-Layout
        case) was normally still empty exactly when it was supposed to
        matter, and _process_unblocked_tasks (whose whole job is dispatching
        tasks once their dependencies clear) never got a chance to act on
        any dependency-gated child at all.

        Now: only flip status to "running" here if the task's dependencies
        are already satisfied (in_degree == 0). A dependency-gated task
        keeps its agent_id (so orchestrator._process_unblocked_tasks won't
        try to double-spawn it) but stays at status=0 (pending) until
        complete_task() flips it to running once its last dependency
        finishes -- see the matching fix there.

        NOTE: orchestrator.spawn_agent() still eagerly creates a live Agent
        object and debits its spawn energy cost regardless of this gate --
        that live Agent simply sits un-ticked in orchestrator.live_agents
        (guarded by _run_live_agents' existing `task_node.status != 1: continue`
        check) until this task's status actually becomes 1. That's a
        deliberate smaller-footprint choice rather than deferring agent
        creation itself -- see orchestrator.py's spawn_agent docstring.
        """
        task = self.tasks.get(task_id)
        if task:
            task.agent_id = agent_id
            if task.in_degree == 0:
                task.status = 1
        else:
            print(f"Warning: {task_id} not found for assignment.")

    def get_snapshot(self):
        task_status = {
            "tasks" : self.tasks
# any other shape requirements
        }
        return task_status