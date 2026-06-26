"""
main brain of the project,

functions this should perform:
SPAWN: agent requests children, orchestrator creates them
PROMOTE: agent finished, orchestrator routes result to parent
EXECUTE: agent failed, orchestrator kills it, extracts ghost, respawns smarter
MERGE: energy low or deadlock, orchestrator consolidates agents
WATCHDOG: monitors agent response latency and resolves systemic lockups
RUN & TERMINATE: manages system lifespan, captures run traces, and retrieves execution specs
"""

import uuid
import time
from typing import Optional, Dict, Any, List


# Assuming these are your custom modules
from colony_state import ColonyState, AgentNode
from task_graph import TaskGraph, TaskNode
from event_queue import Messenger, Event


class Orchestrator:
    def __init__(self, colony_state: ColonyState, task_graph: TaskGraph, messenger: Messenger):
        self.colony = colony_state
        self.task_graph = task_graph
        self.messenger = messenger
        self.agent_counter = 0 
        self.root_task_id: Optional[str] = None
        self.running = True
        self.run_trace: Optional[Dict[str, Any]] = None
        
        # Hyperparameters
        self.energy_when_new = 10
        self.energy_threshold_stress = 10
        self.energy_threshold_death = 5
        self.timeout_threshold = 30.0  # Seconds an agent can remain silent before being killed/probed
        
        self.energy_map = {
            0: "fine",
            1: "stressed",
            2: "death"
        }

    def _generate_id(self) -> str:
        """Generates a unique 8-character ID for new agents."""
        return f"agent_{uuid.uuid4().hex[:8]}"
    
    def _check_energy(self) -> int:
        """Returns the current energy budget of the colony."""
        return getattr(self.colony, 'budget_remaining', 0)
    
    def _get_energy_status(self, rem_energy: int) -> str:
        """Evaluates the energy levels safely against the thresholds."""
        if rem_energy <= self.energy_threshold_death:
            return "death"
        elif rem_energy <= self.energy_threshold_stress:
            return "stressed"
        return "fine"
    
    def _route_events(self, events: List[Event]):
        """Routes incoming events with their payloads and updates agent activity trackers."""
        now = time.time()
        for e in events:
            # Maintain active pulse tracking on every agent dispatching events
            if e.from_agent and e.from_agent != "orchestrator":
                agent = self.colony.get_agent(e.from_agent)
                if agent:
                    agent.last_active = now

            if e.type == "spawn_request":
                self.handle_spawn(e)
            elif e.type == "completion_request":
                self.handle_completion(e)
            elif e.type == "failure_request":
                self.handle_failure(e)
            elif e.type == "tool_request":     
                self.handle_tool_request(e)

    def _process_unblocked_tasks(self):
        """Finds tasks whose dependencies are met and spawns agents for them."""
        ready_tasks = self.task_graph.get_ready_tasks()
        for task in ready_tasks:
            # Spawn a new worker if the ready task doesn't have an agent assigned yet
            if getattr(task, 'agent_id', None) is None:
                required_role = getattr(task, 'required_role', 'worker')
                self.spawn_agent(role=required_role, task_id=task.task_id)

    def initialize_colony(self, problem_spec: str):
        """System entry point. Bootstraps the first task and the root agent."""
        self.root_task_id = "root_task_0"
        
        # Instantiate the root task node with the initial problem description
        root_node = TaskNode(task_id=self.root_task_id, description=problem_spec, required_role="planner")
        
        # Inject the root task into the task graph
        self.task_graph.add_task(root_node)
            
        print(f"Colony initialized. Bootstrapping root task: {self.root_task_id}")
        self.spawn_agent(role="planner", task_id=self.root_task_id)

    def spawn_agent(self, role: str, task_id: str, parent_id: Optional[str] = None, ghost_context: Optional[Dict[str, Any]] = None) -> str:
        """SPAWN: Creates an agent, assigns it to a task, and deducts energy."""
        agent_id = self._generate_id()
        
        # Create the agent with initial states and any inherited ghost context (failure memory)
        new_agent = AgentNode(
            agent_id=agent_id,
            role=role,
            status="idle",
            parent_id=parent_id,
            task=task_id,
            ghost_context=ghost_context 
        )
        
        self.colony.register_agent(new_agent)
        self.task_graph.assign_agent(task_id, agent_id)
        
        # Debit the initialization energy cost from the colony budget
        self.colony.debit_energy(agent_id, self.energy_when_new)
        self.agent_counter += 1
        
        return agent_id

    def handle_spawn(self, event: Event):
        """Triggered when an existing agent requests sub-agents (children)."""
        payload = event.payload
        task_id = payload.get("task_id", payload.get("task"))
        
        if not task_id:
            print("Warning: Spawn request ignored due to missing task ID in payload.")
            return

        description = payload.get("description", f"Subtask derived from task {task_id}")
        dependencies = payload.get("dependencies", [])
        role = payload.get("role", "worker")
        
        # Construct the child task node and append it to the graph before spawning its agent
        child_node = TaskNode(
            task_id=task_id,
            description=description,
            dependencies=dependencies,
            required_role=role
        )
        self.task_graph.add_task(child_node)

        self.spawn_agent(
            role=role,
            task_id=task_id,
            parent_id=payload.get("parent_id")
        )

    def handle_completion(self, event: Event):
        """PROMOTE: Routes completed results to parents, unblocks tasks, or triggers synthesizer."""
        payload = event.payload
        # agent_id = payload.get("agent_id")
        agent_id = event.from_agent
        task_id = payload.get("task_id")
        result = payload.get("result")
        
        # Complete task in graph and store results in the colony state memory
        self.task_graph.complete_task(task_id)
        self.colony.store_result(task_id, result) 
        
        # Directly store result on the task node to guarantee reliable recovery during termination
        task_node = self.task_graph.tasks.get(task_id)
        if task_node:
            task_node.result = result

        # Root completion check
        if task_id == self.root_task_id:
            print("Root task completed! Triggering Synthesizer...")
            # Set colony goal completion state safely
            self.colony.results["final_spec"] = result
            return
        
        # Route the notification up the hierarchy to the parent agent
        agent = self.colony.get_agent(agent_id)
        parent_id = getattr(agent, 'parent_id', None)
        
        if agent and parent_id:
            self.messenger.push_event(
                "parent_notification",
                "orchestrator",
                {"parent_id": parent_id, "child_id": agent_id, "result": result}
            )
            
        self.colony.update_status(agent_id, "completed")

    def handle_failure(self, event: Event):
        """EXECUTE: Harvest context, kill agent, and respawn a smarter version."""
        payload = event.payload
        # agent_id = payload.get("agent_id")
        agent_id = event.from_agent
        task_id = payload.get("task_id")
        role = payload.get("role", "worker")
        parent_id = payload.get("parent_id")
        
        # Extract the state logic and errors leading to death before unregistering
        ghost_context = self.colony.extract_agent_ghost(agent_id)
        
        # Unregister the failed node to free immediate memory
        self.colony.unregister_agent(agent_id)
        
        # Respawn an agent for the same task, injecting the failure memory
        print(f"Agent {agent_id} failed. Respawning {role} with ghost context.")
        self.spawn_agent(role=role, task_id=task_id, parent_id=parent_id, ghost_context=ghost_context)

    def handle_tool_request(self, event: Event):
        """Handles external tool execution requests from agents."""
        payload = event.payload
        # agent_id = payload.get("agent_id")
        agent_id = event.from_agent
        tool_name = payload.get("tool_name")
        
        print(f"Executing tool '{tool_name}' for agent {agent_id}...")
        
        # Provide a mock tool execution response. (To be replaced with actual tool registry execution logic)
        mock_result = f"Tool {tool_name} executed successfully"
        
        self.messenger.push_event(
            "tool_response", 
            "orchestrator",
            {"agent_id": agent_id, "result": mock_result}
        )

    def merge_agents(self):
        """MERGE: Consolidation mechanism to survive low-energy bottlenecks."""
        print("System strain detected! Consolidating idle / overlapping agents...")
        culled = self.colony.consolidate_idle_agents()
        print(f"Consolidated {culled} idle agents and reclaimed energy.")

    def handle_deadlock(self):
        """
        Runs systemic watchdog checks to detect and resolve system bottlenecks or silent agents.
        
        Check 1: Evaluates global lockup (empty mailbox, zero tasks ready for scheduling, root incomplete).
        Check 2: Recovers silent active agents executing tasks that have breached activity thresholds.
        Check 3: Reschedules unassigned task nodes that fell through during previous cycles.
        """
        ready_tasks = self.task_graph.get_ready_tasks()
        root_task = self.task_graph.tasks.get(self.root_task_id)
        root_incomplete = root_task is not None and root_task.status != 2

        # Check 1: Potential Deadlock Detection (no events, no ready tasks, root still incomplete)
        if self.messenger.peek() == 0 and len(ready_tasks) == 0 and root_incomplete:
            now = time.time()
            running_tasks = [t for t in self.task_graph.tasks.values() if t.status == 1]
            
            # Count how many currently running agents are actually "active" (not timed out)
            active_agents = []
            stuck_agents = []
            
            for task in running_tasks:
                if task.agent_id:
                    agent = self.colony.get_agent(task.agent_id)
                    if agent:
                        silence_duration = now - agent.last_active
                        if silence_duration > self.timeout_threshold:
                            stuck_agents.append((agent, task))
                        else:
                            active_agents.append(agent)

            # If active agents are still running and responsive, let them work (false alarm)
            if not stuck_agents and len(active_agents) > 0:
                return

            print("System deadlock confirmed! Investigating state nodes...")

            # Check 2: Force respawn of non-responsive (stuck) active agents
            if stuck_agents:
                for agent, task in stuck_agents:
                    print(f"Deadlock Watchdog: Agent {agent.agent_id} on Task {task.task_id} "
                          f"is non-responsive for {now - agent.last_active:.2f}s. Forcing respawn.")
                    self.messenger.push_event(
                        "failure_request",
                        "orchestrator",
                        {
                            "agent_id": agent.agent_id,
                            "task_id": task.task_id,
                            "role": agent.role,
                            "parent_id": agent.parent_id
                        }
                    )
            # Check 3: Rescue Orphaned / Unassigned Pending Tasks
            else:
                pending_unassigned_tasks = [
                    t for t in self.task_graph.tasks.values() 
                    if t.status == 0 and t.agent_id is None
                ]
                if pending_unassigned_tasks:
                    print(f"Deadlock Watchdog: Recovered {len(pending_unassigned_tasks)} unassigned pending tasks. Spawning workers...")
                    for task in pending_unassigned_tasks:
                        required_role = getattr(task, 'required_role', 'worker')
                        self.spawn_agent(role=required_role, task_id=task.task_id)
                else:
                    print("Deadlock Watchdog: System is stable. No actionable stalls detected.")

    def tick(self) -> bool:
        """Processes a single heartbeat of the orchestrator loop."""
        rem_energy = self._check_energy()
        status = self._get_energy_status(rem_energy)
        
        if status == "death":
            print(f"Critical failure. Colony energy depleted ({rem_energy} remaining).")
            return False
        elif status == "stressed":
            self.merge_agents()

        # Handle systemic deadlock detection before proceeding with queue processing
        self.handle_deadlock()

        # Drain the current event queue, route them, and unblock sequential tasks
        events = self.messenger.drain()
        self._route_events(events)
        self._process_unblocked_tasks()
        
        # Halt execution loop if the root task is complete
        root_task = self.task_graph.tasks.get(self.root_task_id)
        if root_task and root_task.status == 2:
            return False
    
        return True

    def terminate(self) -> Any:
        """
        1. Prints system resolution (Success or Energy Death).
        2. Captures system execution trace using a snapshot.
        3. Clears execution loop flags.
        4. Isolates and returns the highest quality result spec.
        """
        self.running = False
        self.run_trace = self.colony.get_snapshot()

        root_task = self.task_graph.tasks.get(self.root_task_id)
        is_successful = root_task is not None and root_task.status == 2

        if is_successful:
            print("System Status: TERMINATED [SUCCESS]")
            print("The root task successfully completed and synthesized.")
        else:
            rem_energy = self._check_energy()
            print("System Status: TERMINATED [ENERGY DEATH]")
            print(f"The colony depleted its energy allocation. Remaining budget: {rem_energy}")

        # Resolve the highest quality result available
        best_result = self.colony.results.get("final_spec")
        if not best_result and root_task:
            best_result = getattr(root_task, 'result', None)
        if not best_result:
            # Fallback output to full captured registry
            best_result = self.colony.results

        return best_result

    def run(self, problem_spec: str) -> Any:
        """
        1. Bootstraps the root node space with initial specifications.
        2. Loops ticking cycles until a stop signal or death triggers.
        3. Terminates the cycle and retrieves results.
        """
        self.running = True
        self.initialize_colony(problem_spec)

        while self.running:
            tick_success = self.tick()
            if not tick_success:
                break

        return self.terminate()