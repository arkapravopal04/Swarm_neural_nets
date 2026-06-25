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
    result : bool = False


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

        pass

# should return the list of tasks that we can begin now , should also be able to change the task status to 1 of the task we just completed
# when we loop thru the unblocked we reduce their in dregree as one of them is done
    def complete_task(self, task_id : str):
        task = self.tasks[task_id]
        if not task:
            print(f"{task_id} invalid")

        task.status = 2
        newly_unblocked = []
        for dependent_id in task.dependents:
            dep_task = self.tasks.get(dependent_id)
            if dep_task:
                dep_task.in_degree -= 1  
                if dep_task.in_degree == 0:
                    newly_unblocked.append(dependent_id)
                    
        return newly_unblocked
    

# if a task was worked on and failed it needs to be broken down into more atomic tasks, will be done by some other function...this 
# one needs to somehow handle the consequences of faliure 
    def fail_task(self , task_id : str):
        task = self.tasks[task_id]
        if not task:
            print(f"{task_id} invalid")
        
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
        task = self.tasks[task_id]
        if task:
            task.status = 1
            task.agent_id = agent_id
        else:
            print(f"Warning: {task_id} not found for assignment.")

    def get_snapshot(self):
        task_status = {
            "tasks" : self.tasks
# any other shape requirements
        }
        return task_status
    
