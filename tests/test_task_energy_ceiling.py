"""
Per-task cumulative-energy ceiling: a second, independent check next to
MAX_TASK_ATTEMPTS on the respawn decision. The per-agent cycle cap bounds one
agent, but each respawn starts a fresh agent with a fresh cycle budget; this
bounds what all of a task's agents spend together.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colony_state import AgentNode, ColonyState
from event_queue import Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


def _orchestrator(starting_budget=1000):
    colony = ColonyState(initial_budget=starting_budget, goal_embedding=None)
    orch = Orchestrator(colony, TaskGraph(), Messenger(), memory_store=_NullMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan", agent_id="p", status=1))
    orch.colony.register_agent(AgentNode(agent_id="p", role="decomposer", status="running",
                                         parent_id=None, task="plan", task_id="t-parent"))
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="compute x", agent_id="a1", status=1))
    orch.colony.register_agent(AgentNode(agent_id="a1", role="executor", status="running",
                                         parent_id="p", task="compute x", task_id="t-1"))
    return orch


# ~4 chars per token: the proxy the energy model bills think output in, used
# here only to size a fake full-cap cycle.
_CHARS_PER_TOKEN = 4


def _agents_on(orch, task_id):
    return [a for a in orch.colony.agents.values() if a.task_id == task_id]


def test_task_energy_accumulates_across_respawned_agents():
    orch = _orchestrator()
    orch.colony.debit_energy("a1", 30, category="think_tick")
    orch.colony.unregister_agent("a1")
    orch.colony.register_agent(AgentNode(agent_id="a2", role="executor", status="running",
                                         parent_id="p", task="compute x", task_id="t-1"))
    orch.colony.debit_energy("a2", 12, category="think_tick")

    assert orch.colony.task_energy_spent["t-1"] == 42
    assert orch.colony.agents["a2"].energy_spent == 12  # per-agent figure still resets


def test_ceiling_is_priced_in_attempts_not_in_budget():
    """It used to be 10% of the starting budget, which at the configured
    budget_override=2500 computed to 250 -- above the 214 that the worst case
    the caps can structurally produce costs, so it could never fire. Now it is
    TASK_CEILING_ATTEMPTS attempts of what the task itself costs."""
    orch = _orchestrator(starting_budget=2500)
    ceiling = orch.task_energy_ceiling("t-1")

    assert ceiling == int(orch.TASK_CEILING_ATTEMPTS * orch.RESERVE_EXECUTOR)
    for budget in (100, 800, 3000):
        orch.colony.starting_budget = budget
        assert orch.task_energy_ceiling("t-1") == ceiling, "budget must not be an input"


def test_executor_ceiling_prices_every_attempt_the_cap_grants():
    """One p75 executor attempt (27) for each of the 4 agents MAX_TASK_ATTEMPTS
    allows: 108. The old 81 priced three."""
    assert _orchestrator().task_energy_ceiling("t-1") == 108
    assert _orchestrator().task_energy_ceiling() == 108   # the reference figure


def test_the_ceiling_is_tied_to_the_attempt_cap():
    orch = _orchestrator()
    assert orch.TASK_CEILING_ATTEMPTS == orch.MAX_TASK_ATTEMPTS + 1


def test_a_task_that_needs_its_fourth_attempt_gets_it():
    """task_4ae60b94 / task_4c62ea1f: four attempts of ~21 (three tier-3
    rejects at 4-5 energy each) ran to 82-85. With the ceiling at three
    attempts (81) the fourth attempt was cut off by the ceiling although no
    attempt ran over p75; the attempt cap, not the ceiling, has to decide."""
    orch = _orchestrator()
    agent = "a1"
    for attempt in range(2, 5):
        orch.colony.debit_energy(agent, 21, category="think_tick")
        orch._kill_and_respawn(agent, "t-1", "executor", "p")
        respawned = _agents_on(orch, "t-1")
        assert "t-1" not in orch.abandoned_tasks and respawned, f"attempt {attempt} refused"
        agent = respawned[0].agent_id

    orch.colony.debit_energy(agent, 21, category="think_tick")   # the fourth attempt runs
    spent = orch.colony.task_energy_spent["t-1"]                  # 4 x 21 + 3 respawn spawns
    assert 81 < spent < orch.task_energy_ceiling("t-1")
    assert not orch._enforce_task_energy_ceiling(agent, "t-1")
    assert "t-1" not in orch.abandoned_tasks


def test_ceiling_is_priced_at_the_tasks_own_role_and_batch():
    """A decomposer's attempt grows with its batch size k (SPAWN payload + one
    injection per child) and the root's carries sigma, so one flat figure
    cannot fit both. Priced per task instead."""
    orch = _orchestrator()
    res = orch._reservation("t-parent")
    assert res.kind == "decomposer"
    by_k = []
    for k in (1, 2, 3, 4):
        res.k, res.spawned = k, True
        by_k.append(orch.task_energy_ceiling("t-parent"))
    # Flat up to the fan-out cap (2), then one beta per extra child.
    assert by_k[0] == by_k[1] < by_k[2] < by_k[3]
    assert by_k[3] - by_k[2] == int(orch.TASK_CEILING_ATTEMPTS * orch.RESERVE_PER_CHILD)

    orch.root_task_id = "t-root"
    orch.task_graph.add_task(TaskNode(task_id="t-root", description="goal", status=1))
    root = orch._reservation("t-root")
    root.k, root.spawned = 3, True
    assert root.kind == "root"
    assert orch.task_energy_ceiling("t-root") > by_k[2]


def test_a_conversion_carries_its_executor_spend_into_the_ceiling():
    """TASK TOO LARGE: the executor's spend so far is added on top of the new
    decomposer ceiling instead of eating it."""
    orch = _orchestrator()
    orch.colony.debit_energy("a1", 40, category="think_tick")
    assert orch._admit_conversion("t-1")
    res = orch._reservation("t-1")
    assert res.kind == "decomposer" and res.conv_sunk == 40
    assert orch.task_energy_ceiling("t-1") == int(
        40 + orch.TASK_CEILING_ATTEMPTS
        * (orch.RESERVE_DECOMPOSER_BASE
           + orch.MAX_SUBTASKS_NON_ROOT * orch.RESERVE_PER_CHILD))


def test_a_decomposer_ceiling_is_above_an_executor_ceiling_before_it_spawns():
    """task_4c62ea1f: four decomposer attempts, none with a valid SPAWN batch,
    each paying for THINK ticks and a rejected payload. Priced at k=1 its
    ceiling was below an executor's. Priced at the fan-out cap it is 130
    (executor 108), and the root's 195."""
    orch = _orchestrator()
    orch.task_graph.add_task(TaskNode(task_id="t-dec", description="plan part",
                                      agent_id="d", status=1))
    orch.colony.register_agent(AgentNode(agent_id="d", role="decomposer", status="running",
                                         parent_id="p", task="plan part", task_id="t-dec"))
    res = orch._reservation("t-dec")
    assert res.kind == "decomposer" and not res.spawned
    assert orch.task_energy_ceiling("t-dec") == 130
    assert orch.task_energy_ceiling("t-dec") > orch.task_energy_ceiling("t-1")

    orch.root_task_id = "t-root"
    orch.task_graph.add_task(TaskNode(task_id="t-root", description="goal", status=1))
    assert orch._reservation("t-root").kind == "root"
    assert orch.task_energy_ceiling("t-root") == 195
    # Admission still prices an unspawned decomposer at its cheapest plan.
    assert orch._own_cost(res) == orch.RESERVE_DECOMPOSER_BASE + orch.RESERVE_PER_CHILD


def test_a_decomposer_ceiling_does_not_drop_when_a_respawn_replans():
    """A respawned decomposer plans again from k=0, which admission needs. The
    ceiling kept following k, so a task that had admitted 4 children dropped
    from 178 to 130 on its respawn."""
    orch = _orchestrator()
    res = orch._reservation("t-parent")
    res.k, res.spawned, res.attempts = 4, True, 1
    before = orch.task_energy_ceiling("t-parent")
    orch.colony.unregister_agent("p")
    assert orch.spawn_agent(role="decomposer", task_id="t-parent") is not None
    assert (res.k, res.spawned) == (0, False)
    assert orch.task_energy_ceiling("t-parent") == before == 178


def test_ceiling_clears_a_healthy_task_and_catches_a_bad_one():
    """The calibration property, and the one that should fail loudly if either
    cap is retuned without re-fitting the RESERVE_* figures. Figures are
    measured against the real tick loop: a healthy subtask costs 13, a healthy
    root decomposer 22 including child injections, and a task that warn-loops
    through every attempt costs 214 unbounded."""
    orch = _orchestrator(starting_budget=2500)
    ceiling = orch.task_energy_ceiling("t-1")

    assert ceiling > 3 * 13, "must not threaten a healthy subtask's retries"
    assert ceiling < 214, "must fire before a task exhausts every attempt"


def test_task_at_ceiling_is_abandoned_instead_of_respawned():
    orch = _orchestrator()
    ceiling = orch.task_energy_ceiling()
    orch.colony.debit_energy("a1", ceiling, category="think_tick")

    orch._kill_and_respawn("a1", "t-1", "executor", "p")

    task = orch.task_graph.tasks["t-1"]
    assert task.status == 3
    assert "per-task ceiling" in task.result
    assert orch.abandon_reasons["t-1"] == "energy ceiling"
    assert _agents_on(orch, "t-1") == []
    assert orch.respawn_counts.get("t-1", 0) == 0  # attempts were not the reason
    notes = [e for e in orch.messenger.drain() if e.type == "parent_notification"]
    assert [e.payload["parent_id"] for e in notes] == ["p"]


def test_task_under_ceiling_still_respawns():
    orch = _orchestrator()
    orch.colony.debit_energy("a1", orch.task_energy_ceiling() - 1, category="think_tick")

    orch._kill_and_respawn("a1", "t-1", "executor", "p")

    assert orch.task_graph.tasks["t-1"].status == 1
    assert len(_agents_on(orch, "t-1")) == 1
    assert orch.respawn_counts["t-1"] == 1
    assert "t-1" not in orch.abandoned_tasks


def test_attempt_cap_still_fires_independently_of_energy():
    orch = _orchestrator()
    orch.respawn_counts["t-1"] = orch.MAX_TASK_ATTEMPTS

    orch._kill_and_respawn("a1", "t-1", "executor", "p")

    assert orch.abandon_reasons["t-1"] == "attempt cap"
    assert orch.colony.task_energy_spent.get("t-1", 0) < orch.task_energy_ceiling()


def test_energy_report_shows_task_energy_against_ceiling(capsys):
    orch = _orchestrator()
    orch.colony.debit_energy("a1", orch.task_energy_ceiling() + 5, category="think_tick")
    orch._kill_and_respawn("a1", "t-1", "executor", "p")
    capsys.readouterr()

    orch._print_energy_report()
    out = capsys.readouterr().out

    assert "per-task ceiling" in out and f"executor {orch.task_energy_ceiling()}" in out
    line = next(l for l in out.splitlines() if "t-1" in l and "agents=" in l)
    assert "OVER" in line and "[energy ceiling]" in line


class _AlwaysWarns:
    def decide(self, agent_node, **kwargs):
        return {"verdict": "warn", "reason": "restates the task"}


class _AlwaysPromotes:
    def decide(self, agent_node, **kwargs):
        return {"verdict": "promote", "reason": "ok"}


def _drive_colony(judge, ticks=300):
    """Run the real tick loop with an agent that always REPORTs after a
    full-cap think()+decide() cycle. Returns the settled orchestrator."""
    from agent_node import Agent, agent_cycle_tokens

    original_run = Agent.run
    chars = agent_cycle_tokens("executor") * 4        # _CHARS_PER_TOKEN
    seq = [0]

    def always_reports(self, available_roles=None, available_tools=None, requirements=None):
        seq[0] += 1
        self.thought_process += "x" * chars
        self.report(f"answer variant {seq[0]}")
        return "REPORT"

    Agent.run = always_reports
    try:
        orch = _orchestrator(starting_budget=2500)
        orch.judge = judge
        orch.model, orch.tokeniser = object(), object()
        orch.root_task_id = "t-parent"
        orch.live_agents["a1"] = Agent(object(), object(), orch.messenger,
                                       orch.colony.get_agent("a1"))
        for _ in range(ticks):
            if orch.task_graph.tasks["t-1"].status in (2, 3):
                break
            orch.tick()
        else:
            raise AssertionError("did not settle")
    finally:
        Agent.run = original_run

    return orch


def _drive(judge, ticks=300):
    """(status, reason, energy, agents) for the tests that only read those."""
    orch = _drive_colony(judge, ticks)
    return (orch.task_graph.tasks["t-1"].status,
            orch.abandon_reasons.get("t-1"),
            orch.colony.task_energy_spent.get("t-1", 0),
            1 + orch.respawn_counts.get("t-1", 0))


def test_healthy_task_is_never_touched_by_the_ceiling():
    status, reason, energy, agents = _drive(_AlwaysPromotes())

    assert status == 2 and reason is None
    assert energy < _orchestrator().task_energy_ceiling()


def test_runaway_task_is_stopped_by_the_ceiling_before_the_attempt_cap():
    """The behaviour the ceiling exists for and never had: under the old
    budget fraction this ran all 1+MAX_TASK_ATTEMPTS agents and was closed by
    the attempt cap at 214 energy."""
    status, reason, energy, agents = _drive(_AlwaysWarns())

    assert status == 3
    assert reason == "energy ceiling"
    orch = _orchestrator()
    assert agents < 1 + orch.MAX_TASK_ATTEMPTS, "must stop short of every attempt"
    assert energy < 214, "must cost less than letting the attempt cap close it"


def test_over_flag_separates_a_stopped_task_from_a_still_funded_one(capsys):
    """"No task row should say OVER" cannot hold: the ceiling is checked at
    respawn time and the live agent is never killed mid-cycle, so a task always
    crosses the line before it can be caught. OVER on a stopped task is the fix
    working; OVER on a task nothing stopped is the bug."""
    orch = _orchestrator()
    over = orch.task_energy_ceiling() + 5
    orch.colony.debit_energy("a1", over, category="think_tick")
    orch.task_graph.add_task(TaskNode(task_id="t-2", description="still going",
                                      agent_id="a2", status=1))
    orch.colony.register_agent(AgentNode(agent_id="a2", role="executor", status="running",
                                         parent_id="p", task="still going", task_id="t-2"))
    orch.colony.debit_energy("a2", over, category="think_tick")
    orch._kill_and_respawn("a1", "t-1", "executor", "p")   # stops t-1 by the ceiling
    capsys.readouterr()

    orch._print_energy_report()
    lines = capsys.readouterr().out.splitlines()

    stopped = next(l for l in lines if "t-1" in l and "agents=" in l)
    running = next(l for l in lines if "t-2" in l and "agents=" in l)
    assert "OVER" in stopped and "NOT STOPPED" not in stopped
    assert "[energy ceiling]" in stopped
    assert "OVER -- NOT STOPPED" in running


# ---------------------------------------------------------------------------
# The ceiling as a per-cycle check, not just a respawn-time one.


def _thinking_agent(orch, agent_id="a1"):
    """A live agent whose cycle costs a full agent-cycle and never reaches a
    terminal action -- the shape that produces no respawn decision at all, so
    nothing used to consult the ceiling on its behalf.
    """
    from agent_node import Agent, agent_cycle_tokens

    chars = agent_cycle_tokens("executor") * _CHARS_PER_TOKEN
    agent = Agent(object(), object(), orch.messenger,
                  orch.colony.get_agent(agent_id))

    def only_thinks(available_roles=None, available_tools=None, requirements=None):
        agent.thought_process += "x" * chars
        agent.non_terminal_cycles += 1
        return "THINK"

    agent.run = only_thinks
    orch.live_agents[agent_id] = agent
    return agent


def test_ceiling_fires_mid_agent_with_no_respawn_in_between():
    """The gap task_ac3e5d85 fell through: its earlier agents left the task one
    unit under the line, and the current agent crossed it on an ordinary THINK.
    No terminal action, so no respawn decision, so nothing checked.
    """
    orch = _orchestrator(starting_budget=2500)
    orch.colony.debit_energy("a1", orch.task_energy_ceiling() - 1,
                             category="think_tick")
    _thinking_agent(orch)

    orch._run_live_agents()

    assert orch.task_graph.tasks["t-1"].status == 3
    assert orch.abandon_reasons["t-1"] == "energy ceiling"
    assert orch.respawn_counts.get("t-1", 0) == 0, "no respawn boundary was involved"
    assert _agents_on(orch, "t-1") == []
    assert "a1" not in orch.live_agents


def test_mid_agent_overshoot_is_one_cycle_not_one_agent():
    """What the move buys. Under the respawn-only check the same agent ran its
    whole remaining cycle budget past the line before anything looked.
    """
    from agent_node import Agent, agent_cycle_tokens

    orch = _orchestrator(starting_budget=2500)
    ceiling = orch.task_energy_ceiling()
    orch.colony.debit_energy("a1", ceiling - 1, category="think_tick")
    _thinking_agent(orch)

    orch._run_live_agents()

    one_cycle = max(1, (agent_cycle_tokens("executor")
                        * _CHARS_PER_TOKEN) // 100)
    overshoot = orch.colony.task_energy_spent["t-1"] - ceiling
    assert 0 <= overshoot <= one_cycle
    assert overshoot < Agent.MAX_NON_TERMINAL_CYCLES * one_cycle


def test_mid_agent_abandon_releases_dependents_and_tells_the_parent():
    """Same disposition the attempt cap produces -- the task is closed, not left
    running with no agent, and the parent gets the partial instead of hanging.
    """
    orch = _orchestrator(starting_budget=2500)
    orch.task_graph.add_task(TaskNode(task_id="t-dep", description="needs t-1",
                                      status=0, dependencies=["t-1"]))
    orch.last_partial_result["t-1"] = "half an answer"
    orch.colony.debit_energy("a1", orch.task_energy_ceiling(), category="think_tick")
    _thinking_agent(orch)

    orch._run_live_agents()

    result = orch.task_graph.tasks["t-1"].result
    assert "ABANDONED" in result and "per-task ceiling" in result
    assert "half an answer" in result
    notes = [e for e in orch.messenger.drain() if e.type == "parent_notification"]
    assert [e.payload["parent_id"] for e in notes] == ["p"]
    # _release_dependents does complete_task's in_degree bookkeeping without
    # the status=2 that would be a lie here.
    assert orch.task_graph.tasks["t-dep"].in_degree == 0


def test_a_healthy_cycle_under_the_ceiling_is_left_alone():
    orch = _orchestrator(starting_budget=2500)
    _thinking_agent(orch)

    orch._run_live_agents()

    assert orch.task_graph.tasks["t-1"].status == 1
    assert "t-1" not in orch.abandoned_tasks
    assert "a1" in orch.live_agents


def test_a_reporting_cycle_is_judged_before_the_ceiling_can_stop_it():
    """A REPORT is queued on the cycle it is produced and judged on the next
    tick. Killing the agent the moment it crosses would spend the energy and
    then throw the result away -- the opposite of what the ceiling is for. It
    is stopped one cycle later instead, if it is still going.
    """
    from agent_node import Agent, agent_cycle_tokens

    orch = _orchestrator(starting_budget=2500)
    orch.colony.debit_energy("a1", orch.task_energy_ceiling() - 1,
                             category="think_tick")
    chars = agent_cycle_tokens("executor") * _CHARS_PER_TOKEN
    agent = Agent(object(), object(), orch.messenger, orch.colony.get_agent("a1"))

    def reports(available_roles=None, available_tools=None, requirements=None):
        agent.thought_process += "x" * chars
        agent.report("the answer")
        return "REPORT"

    agent.run = reports
    orch.live_agents["a1"] = agent

    orch._run_live_agents()

    assert orch.colony.task_energy_spent["t-1"] >= orch.task_energy_ceiling()
    assert "t-1" not in orch.abandoned_tasks, "the REPORT must reach the judge"
    assert "a1" in orch.live_agents
    queued = [e for e in orch.messenger.drain() if e.type == "completion_request"]
    assert len(queued) == 1


def test_stale_completion_for_an_abandoned_task_is_dropped():
    """The event queue is a tick behind the ceiling now. handle_completion finds
    no agent_node for a retired agent, so it skips the judge and would promote
    the result straight over the abandonment its parent was already handed.
    """
    orch = _orchestrator(starting_budget=2500)
    orch.colony.debit_energy("a1", orch.task_energy_ceiling(), category="think_tick")
    orch.messenger.push_event("completion_request", "a1",
                              {"task_id": "t-1", "result": "late answer"})
    _thinking_agent(orch)
    orch._run_live_agents()
    assert "t-1" in orch.abandoned_tasks

    orch._route_events(orch.messenger.drain())

    assert orch.task_graph.tasks["t-1"].status == 3
    assert "late answer" not in str(orch.task_graph.tasks["t-1"].result)
    assert "ABANDONED" in str(orch.colony.results["t-1"])


def test_an_awaiting_decomposer_is_not_stopped_while_its_children_run():
    """Its children keep billing injections to its task while it cannot act. It
    is checked when it is next actually runnable, not while it waits -- closing
    a parent mid-flight would orphan the work it is waiting on.
    """
    orch = _orchestrator(starting_budget=2500)
    orch.colony.debit_energy("p", orch.task_energy_ceiling("t-parent") + 10,
                             category="injection")
    parent = _thinking_agent(orch, agent_id="p")
    parent.awaiting = "children"

    orch._run_live_agents()

    assert orch.task_graph.tasks["t-parent"].status == 1
    assert "t-parent" not in orch.abandoned_tasks


def test_the_warn_loop_is_stopped_mid_agent_not_at_the_respawn_boundary():
    """The regression guard for how this was first got wrong.

    A REPORT the judge WARNs is the shape the runaway actually takes -- every
    pass through the loop is a REPORT cycle -- and the WARN branch is the one
    decision point that lets an agent continue without routing through
    _kill_and_respawn. A per-cycle check placed only in _run_live_agents is
    never reached on this path at all: measured on this exact scenario it took
    zero calls while the task still ran to 106 against a ceiling of 81, caught
    (as before the change) only by the respawn-time check.
    """
    from agent_node import agent_cycle_tokens

    orch = _drive_colony(_AlwaysWarns())
    ceiling = orch.task_energy_ceiling()
    spent = orch.colony.task_energy_spent["t-1"]

    # Stopped by the mid-agent check, not only by a respawn decision.
    assert orch.colony.verdict_counts.get("energy_ceiling_midagent") == 1
    assert orch.abandon_reasons["t-1"] == "energy ceiling"

    # And stopped within a cycle of crossing, rather than after the agent
    # spends out the rest of its cycle budget.
    one_cycle = max(1, (agent_cycle_tokens("executor")
                        * _CHARS_PER_TOKEN) // 100)
    assert 0 <= spent - ceiling <= one_cycle

    # The rejected REPORT is already in last_partial_result when the WARN
    # lands, so abandoning there salvages it instead of discarding it.
    assert "answer variant" in orch.task_graph.tasks["t-1"].result


def test_the_ledger_separates_a_mid_agent_stop_from_a_respawn_time_one(capsys):
    """Both stops label the task "energy ceiling" in the abandoned list,
    which is right for a reader of that list and useless for deciding
    whether the per-cycle check earns its place. The split is the only
    thing in a run that answers that -- and a guard nobody can see firing
    is how the cycle cap ended up suspected of not existing at all."""
    orch = _orchestrator()
    orch.colony.debit_energy("a1", orch.task_energy_ceiling() + 5, category="think_tick")
    orch._kill_and_respawn("a1", "t-1", "executor", "p")
    capsys.readouterr()

    orch._print_energy_report()
    out = capsys.readouterr().out
    assert "stopped mid-agent (per cycle)  : 0" in out
    assert "stopped at a respawn decision  : 1" in out


def test_a_mid_agent_stop_is_counted_as_one(capsys):
    orch = _orchestrator()
    orch.colony.debit_energy("a1", orch.task_energy_ceiling() + 5, category="think_tick")
    assert orch._enforce_task_energy_ceiling("a1", "t-1") is True
    capsys.readouterr()

    orch._print_energy_report()
    out = capsys.readouterr().out
    assert "stopped mid-agent (per cycle)  : 1" in out
    assert "stopped at a respawn decision  : 0" in out


# ---------------------------------------------------------------------------
# Bill-time check on a child's result, and the post-judgment check on promote.


def _awaiting_parent(orch):
    from agent_node import Agent

    parent = Agent(object(), object(), orch.messenger, orch.colony.get_agent("p"))
    parent.awaiting = "children"
    orch.live_agents["p"] = parent
    return parent


def _notify_parent(orch, child_id="a1", result="child answer"):
    from event_queue import Event

    event = Event(type="parent_notification", from_agent="orchestrator")
    event.payload.update({"parent_id": "p", "child_id": child_id, "result": result})
    orch.handle_parent_notification(event)


def test_injection_that_crosses_the_ceiling_abandons_the_parent_at_bill_time():
    """The parent is awaiting, so _run_live_agents never ticks it and never
    checks it -- the debit that crosses the line has to be the check."""
    orch = _orchestrator(starting_budget=2500)
    orch.colony.debit_energy("p", orch.task_energy_ceiling("t-parent") - 1, category="injection")
    _awaiting_parent(orch)

    _notify_parent(orch)

    assert orch.task_graph.tasks["t-parent"].status == 3
    assert orch.abandon_reasons["t-parent"] == "energy ceiling"
    assert "p" not in orch.live_agents


def test_injection_under_the_ceiling_leaves_the_parent_alone():
    orch = _orchestrator(starting_budget=2500)
    parent = _awaiting_parent(orch)

    _notify_parent(orch)

    assert orch.task_graph.tasks["t-parent"].status == 1
    assert "child answer" in parent.thought_process
    assert "p" in orch.live_agents


def _promote_report(orch, agent_id="a1", task_id="t-1", result="x is 42."):
    from event_queue import Event

    orch.judge = _AlwaysPromotes()
    event = Event(type="completion_request", from_agent=agent_id)
    event.payload.update({"task_id": task_id, "result": result})
    orch.handle_completion(event)


def test_report_promoted_over_the_ceiling_is_kept_and_flagged(capsys):
    orch = _orchestrator(starting_budget=2500)
    orch.colony.debit_energy("a1", orch.task_energy_ceiling() + 5, category="think_tick")

    _promote_report(orch)

    task = orch.task_graph.tasks["t-1"]
    assert task.status == 2, "finished work is kept, not discarded"
    assert task.result == "x is 42."
    assert "t-1" not in orch.abandoned_tasks
    assert "t-1" in orch.completed_over_ceiling
    capsys.readouterr()

    orch._print_energy_report()
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines() if "t-1" in l and "agents=" in l)
    assert "OVER -- COMPLETED OVER CEILING" in row
    assert "NOT STOPPED" not in row


def test_report_promoted_under_the_ceiling_is_not_flagged():
    orch = _orchestrator(starting_budget=2500)

    _promote_report(orch)

    assert orch.task_graph.tasks["t-1"].status == 2
    assert orch.completed_over_ceiling == {}
