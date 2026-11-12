"""
fine_tune_v2/render.py -- training prompts rendered by the runtime itself
(TRAINING_SPEC.md §5).

No prompt here is written by hand. A State describes what an agent would know
at one decide() call; render() builds a real Agent on a stub model, sets that
state through the same attributes the orchestrator sets, and returns
Agent._build_prompt(...) -- the exact user turn decide() would send. Ghost
context and the Shared-state dependency block come from a real Orchestrator
(_goal_line, _child_ghost_context), tool results go through the real
Orchestrator._summarize_tool_result and Agent.receive_tool_result, and decision
records through text_utils.record_decision_text.
"""
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent_node import Agent  # noqa: E402
from colony_state import AgentNode, ColonyState  # noqa: E402
from event_queue import Messenger  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from task_graph import TaskGraph, TaskNode  # noqa: E402
from text_utils import record_decision_text  # noqa: E402

PRODUCTION_ROLES = ["decomposer", "executor", "verifier"]
PRODUCTION_TOOLS = ["run_code", "verify_math"]


@dataclass
class Dependency:
    """A completed prerequisite step whose result a child receives."""
    description: str
    result: str


@dataclass
class ToolCall:
    """One past tool call: name, the registry's result dict, and whether the
    agent's code ran (the circuit breaker reads it)."""
    name: str
    result: dict


@dataclass
class State:
    role: str                       # "decomposer" | "executor"
    generation: int
    request: str                    # THE REQUEST, verbatim
    task: str                       # Your Task
    goal: Optional[str] = None      # phaser goal; None -> the request (what probe 4 saw)
    requirements: list = field(default_factory=list)
    goal_referent: Optional[str] = None
    dependencies: list = field(default_factory=list)      # [Dependency]
    respawn_failures: list = field(default_factory=list)  # earlier failure reasons -> dict ghost
    fail_reason: Optional[str] = None
    thought: str = ""               # raw think() text (sanitized by the runtime path)
    earlier_choices: list = field(default_factory=list)   # raw decide() outputs to record
    tool_calls: list = field(default_factory=list)        # [ToolCall], in order
    child_results: list = field(default_factory=list)     # [(child_id, text)] injected
    children: Optional[list] = None  # child_status_fn rows, or None (no colony view)
    has_spawned: bool = False
    spawn_closed: bool = False
    non_terminal_cycles: int = 0
    disabled_tool_attempts: int = 0
    tool_call_count: Optional[int] = None
    available_tools: list = field(default_factory=lambda: list(PRODUCTION_TOOLS))
    agent_id: Optional[str] = None
    task_id: Optional[str] = None


def _orchestrator(state: State) -> Orchestrator:
    orch = Orchestrator(ColonyState(initial_budget=1000, goal_embedding=None),
                        TaskGraph(), Messenger())
    orch.spec = {"goal": state.goal or state.request, "raw_text": state.request,
                 "requirement": list(state.requirements)}
    return orch


def ghost_for(state: State):
    """The ghost context the orchestrator gives this agent at spawn."""
    orch = _orchestrator(state)
    if state.generation == 0:
        base = f"[Project goal] {orch._goal_line(orch.spec['goal'])}"
    else:
        dep_ids = []
        for i, dep in enumerate(state.dependencies):
            tid = f"t-dep-{i}"
            orch.task_graph.add_task(TaskNode(task_id=tid, description=dep.description,
                                              status=2, result=dep.result))
            dep_ids.append(tid)
        base = orch._child_ghost_context(dep_ids)
    if not state.respawn_failures:
        return base
    return {"base_context": base,
            "earlier_failures": list(state.respawn_failures[:-1]),
            "fail_reason": state.respawn_failures[-1]}


def summarize_tool_result(result: dict) -> str:
    """The runtime's own tool-result text. Line endings are normalised to the
    Linux form the Kaggle runtime produces (a Windows run returns \\r\\n)."""
    text = Orchestrator._summarize_tool_result(result)
    return text.replace("\r\n", "\n")


def build_agent(state: State) -> Agent:
    # Ids are fixed on the state the first time, so every rebuild of the same
    # state (render, then validation) produces the identical prompt.
    if not state.agent_id:
        state.agent_id = f"agent_{uuid.uuid4().hex[:8]}"
    if not state.task_id:
        state.task_id = f"task_{uuid.uuid4().hex[:6]}"
    agent_id = state.agent_id
    node = AgentNode(
        agent_id=agent_id, role=state.role, status="running",
        parent_id=None if state.generation == 0 else "agent_parent",
        task=state.task, task_id=state.task_id,
        generation=state.generation, request_text=state.request,
        requirements=list(state.requirements), goal_referent=state.goal_referent,
    )
    node.ghost_context = ghost_for(state)
    node.has_spawned = state.has_spawned
    agent = Agent(tokeniser=None, model=None, message=Messenger(), node=node)
    # thought_process in the order the runtime writes it: think() text, then
    # decision records, tool results and injected child results.
    if state.thought:
        agent.thought_process += state.thought
    for raw in state.earlier_choices:
        agent.thought_process += "\n" + record_decision_text(raw) + "\n"
    for call in state.tool_calls:
        ok = call.result.get("status") == "success"
        agent.receive_tool_result(call.name, summarize_tool_result(call.result), ok)
    for child_id, text in state.child_results:
        agent.thought_process += f"\n[CHILD RESULT - {child_id}]: {text[:400]}" + (
            "..." if len(text) > 400 else "") + "\n"
    if state.tool_call_count is not None:
        node.tool_call_count = state.tool_call_count
    elif state.tool_calls:
        node.tool_call_count = len(state.tool_calls)
    agent.disabled_tool_attempts = state.disabled_tool_attempts
    agent.spawn_closed = state.spawn_closed
    agent.non_terminal_cycles = state.non_terminal_cycles
    if state.children is not None:
        rows = [dict(r) for r in state.children]
        agent.child_status_fn = lambda rows=rows: rows
    # fail_reason last: receive_tool_result may set it on a failed call, and an
    # explicit verdict replaces that, as a later ruling does at runtime.
    if state.fail_reason is not None:
        agent.fail_reason = state.fail_reason
    return agent


def render(state: State) -> str:
    """The user turn decide() would send for this state."""
    agent = build_agent(state)
    return agent._build_prompt(PRODUCTION_ROLES, list(state.available_tools),
                               list(state.requirements))


def child_row(task, status, label=None, result=None, reused=False):
    """A child_status_fn row as Orchestrator._direct_child_statuses builds it
    (excerpt = the first CHILD_EXCERPT_CHARS of a completed child's result)."""
    excerpt = None
    if status == "completed" and result:
        excerpt = " ".join(str(result).split())
        n = Orchestrator.CHILD_EXCERPT_CHARS
        if len(excerpt) > n:
            excerpt = excerpt[:n].rstrip() + "..."
    return {"task": task, "status": status, "label": label,
            "excerpt": excerpt, "reused": reused}
