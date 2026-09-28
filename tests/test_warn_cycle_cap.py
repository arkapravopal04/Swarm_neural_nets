"""
A judge WARN sends an agent back to revise. It was the one decision point
that counted nothing -- a task has respawn_counts, an agent has
non_terminal_cycles, a WARN had neither -- so REPORT -> WARN -> REPORT ran
for as long as the wording kept changing, at full think() cost, and never
reached _kill_and_respawn where MAX_TASK_ATTEMPTS and the per-task energy
ceiling live. A WARN now costs a cycle, and a WARN at the cap is routed the
same way an EXECUTE is.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode, ColonyState
from event_queue import Event, Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


class _WarnJudge:
    """Always WARNs, which is the loop this module is about."""

    def __init__(self, verdict="warn", reason="the answer restates the task"):
        self.verdict = verdict
        self.reason = reason
        self.calls = 0

    def decide(self, agent_node, **kwargs):
        self.calls += 1
        return {"verdict": self.verdict, "reason": self.reason}


def _orchestrator(starting_budget=2500, judge=None):
    colony = ColonyState(initial_budget=starting_budget, goal_embedding=None)
    orch = Orchestrator(colony, TaskGraph(), Messenger(),
                        judge=judge if judge is not None else _WarnJudge(),
                        memory_store=_NullMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan",
                                      agent_id="p", status=1))
    orch.colony.register_agent(AgentNode(agent_id="p", role="decomposer", status="running",
                                         parent_id=None, task="plan", task_id="t-parent"))
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="name the tradeoff",
                                      agent_id="a1", status=1))
    node = AgentNode(agent_id="a1", role="executor", status="running",
                     parent_id="p", task="name the tradeoff", task_id="t-1")
    orch.colony.register_agent(node)
    # A live Agent is what carries non_terminal_cycles/cycles_capped. The real
    # class, not a stub: cycles_capped is a property over MAX_NON_TERMINAL_CYCLES
    # and the point is that the WARN path now reads the same one run() does.
    orch.live_agents["a1"] = Agent(None, None, orch.messenger, node)
    return orch


def _report(orch, text, agent_id="a1", task_id="t-1"):
    event = Event(type="completion_request", from_agent=agent_id)
    event.payload.update({"agent_id": agent_id, "parent_id": "p",
                          "task_id": task_id, "result": text})
    orch.handle_completion(event)


def _agents_on(orch, task_id):
    return [a for a in orch.colony.agents.values() if a.task_id == task_id]


def test_warn_costs_a_cycle():
    orch = _orchestrator()
    _report(orch, "A tradeoff exists between cost and weight.")

    assert orch.live_agents["a1"].non_terminal_cycles == 1
    assert "a1" in orch.live_agents          # still working, not killed
    assert orch.task_graph.tasks["t-1"].status == 1


def test_warn_loop_is_bounded_by_the_cycle_cap():
    """Every REPORT differs, so the byte-identical short-circuit never fires.
    Before the fix this ran forever; the cap is the only thing that stops it."""
    orch = _orchestrator()
    cap = Agent.MAX_NON_TERMINAL_CYCLES

    for i in range(cap):
        assert "a1" in orch.live_agents, f"killed early at warn {i}"
        _report(orch, f"Cost against weight, phrasing number {i}.")

    # The cap-th WARN is the one that respawns instead of warning again.
    assert "a1" not in orch.live_agents
    assert orch.respawn_counts["t-1"] == 1
    assert len(_agents_on(orch, "t-1")) == 1          # a fresh agent has the task
    assert orch.task_graph.tasks["t-1"].status == 1   # not abandoned, just retried


def test_warn_at_the_cap_reaches_the_energy_ceiling():
    """The point of routing a capped WARN through _kill_and_respawn: that is
    the only place the per-task ceiling is consulted, so a task that warn-looped
    past it can now actually be abandoned."""
    orch = _orchestrator()
    orch.colony.debit_energy("a1", orch.task_energy_ceiling(), category="think_tick")
    orch.live_agents["a1"].non_terminal_cycles = Agent.MAX_NON_TERMINAL_CYCLES - 1

    _report(orch, "Some answer that review will not accept.")

    task = orch.task_graph.tasks["t-1"]
    assert task.status == 3
    assert orch.abandon_reasons["t-1"] == "energy ceiling"
    assert _agents_on(orch, "t-1") == []
    notes = [e for e in orch.messenger.drain() if e.type == "parent_notification"]
    assert [e.payload["parent_id"] for e in notes] == ["p"]


def test_byte_identical_short_circuit_still_fires_first():
    orch = _orchestrator()
    _report(orch, "identical text")
    assert "a1" in orch.live_agents
    assert orch.live_agents["a1"].non_terminal_cycles == 1

    _report(orch, "identical text")

    # Respawned on the repeat, well before the cap, with the reason that
    # names repetition rather than the cap.
    assert "a1" not in orch.live_agents
    assert orch.respawn_counts["t-1"] == 1
    assert "a1" not in orch.last_warned_report
    fresh = _agents_on(orch, "t-1")[0]
    assert "same REPORT twice" in fresh.ghost_context["fail_reason"]


def test_warn_tells_the_agent_it_was_sent_back():
    """A counted cycle has to be a usable one: the agent used to be re-ticked
    with no indication a WARN had happened."""
    orch = _orchestrator(judge=_WarnJudge(reason="restates the task instead of answering it"))
    _report(orch, "The task is to name the tradeoff.")

    reason = orch.colony.get_agent("a1").fail_reason
    assert reason.startswith("SENT BACK BY REVIEW -- ")
    assert "restates the task" in reason


def test_warn_with_no_live_agent_still_behaves():
    orch = _orchestrator()
    orch.live_agents.pop("a1")

    _report(orch, "an answer")

    assert orch.last_warned_report["a1"] == "an answer"
    assert orch.task_graph.tasks["t-1"].status == 1


def test_promote_is_unaffected():
    orch = _orchestrator(judge=_WarnJudge(verdict="promote", reason="fine"))
    _report(orch, "Weight is traded against cost.")

    assert orch.task_graph.tasks["t-1"].status == 2
    assert orch.live_agents["a1"].non_terminal_cycles == 0


def test_ledger_reports_warns_and_cap_respawns(capsys):
    orch = _orchestrator()
    for i in range(Agent.MAX_NON_TERMINAL_CYCLES):
        _report(orch, f"phrasing number {i}.")
    capsys.readouterr()

    orch._print_energy_report()
    out = capsys.readouterr().out

    assert f"judge WARNs issued      : {Agent.MAX_NON_TERMINAL_CYCLES}" in out
    assert "warn loops cut at the cap: 1" in out


def test_cap_reached_is_tallied_on_the_warn_path():
    """_run_live_agents tallies the cap by watching cycles_capped flip across a
    run() call. A WARN flips it outside run() and respawns in the same call, so
    without an explicit tally the ledger reported zero agents reaching a cap
    that had just cut a warn loop."""
    orch = _orchestrator()
    for i in range(Agent.MAX_NON_TERMINAL_CYCLES):
        _report(orch, f"phrasing {i}.")

    assert orch.colony.verdict_counts["cycle_cap_reached"] == 1
    assert orch.colony.verdict_counts["warn_cycle_cap_respawn"] == 1


def test_cap_reached_is_not_double_counted_after_a_think_cap():
    """An agent that already capped on THINK was tallied by _run_live_agents;
    the WARN that follows must not tally it again."""
    orch = _orchestrator()
    agent = orch.live_agents["a1"]
    agent.non_terminal_cycles = Agent.MAX_NON_TERMINAL_CYCLES   # capped by THINK
    orch.colony.record_verdict("cycle_cap_reached")             # as _run_live_agents would

    _report(orch, "a capped agent's forced REPORT")

    assert orch.colony.verdict_counts["cycle_cap_reached"] == 1
    assert orch.colony.verdict_counts["warn_cycle_cap_respawn"] == 1


def test_warn_loop_terminates_through_the_real_tick_loop():
    """The end-to-end shape: a judge that always WARNs and an agent that always
    REPORTs used to spin until MAX_TICKS. It now walks the cycle cap, then the
    attempt cap, and closes the task."""
    original_run = Agent.run
    counter = [0]

    def always_reports(self, available_roles=None, available_tools=None, requirements=None):
        counter[0] += 1
        self.thought_process += f"thinking {counter[0]}. "
        self.report(f"answer variant {counter[0]}")
        return "REPORT"

    Agent.run = always_reports
    try:
        # model/tokeniser must be non-None for spawn_agent to build live Agents
        # for the respawns, or the chain stops for the wrong reason.
        orch = _orchestrator()
        orch.model, orch.tokeniser = object(), object()
        orch.root_task_id = "t-parent"

        for _ in range(200):
            if orch.task_graph.tasks["t-1"].status in (2, 3):
                break
            orch.tick()
        else:
            raise AssertionError("warn loop did not terminate in 200 ticks")
    finally:
        Agent.run = original_run

    task = orch.task_graph.tasks["t-1"]
    assert task.status == 3
    assert orch.respawn_counts["t-1"] == orch.MAX_TASK_ATTEMPTS
    # 1 + MAX_TASK_ATTEMPTS agents, each cut off at MAX_NON_TERMINAL_CYCLES warns
    expected = (1 + orch.MAX_TASK_ATTEMPTS) * Agent.MAX_NON_TERMINAL_CYCLES
    assert orch.colony.verdict_counts["judge_warn"] == expected
    assert orch.colony.verdict_counts["warn_cycle_cap_respawn"] == 1 + orch.MAX_TASK_ATTEMPTS
    assert orch.live_agents == {}
