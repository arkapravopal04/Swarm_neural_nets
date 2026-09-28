"""
Regression tests for the per-agent cycle cap: an agent whose decide() keeps
answering THINK used to loop forever (one executor ran seven-plus THINK
cycles with no REPORT or DIE). After MAX_NON_TERMINAL_CYCLES, run() skips
think(), decide() is offered final_actions only, and anything else the
model writes is converted rather than executed.

Also covers the decomposer paths the first version of the cap broke:
  * a decomposer is not resumed (and so cannot spend its cap and REPORT a
    partial roll-up) until every one of its children has finished,
  * a decomposer that has not spawned yet is capped to SPAWN/DIE, not
    REPORT/DIE, and its capped prompt does not contradict that menu,
  * a SPAWN rejected by the agent itself counts toward the cap.
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_node import Agent
from colony_state import AgentNode, ColonyState
from event_queue import Event, Messenger
from orchestrator import Orchestrator
from task_graph import TaskGraph, TaskNode


def _agent(role="executor", has_spawned=False, generation=1):
    node = AgentNode(agent_id="a-1", role=role, status="running", parent_id="root",
                     task="do the thing", task_id="t-1", generation=generation,
                     has_spawned=has_spawned)
    return Agent(tokeniser=None, model=None, message=Messenger(), node=node)


def _capped(agent):
    agent.non_terminal_cycles = agent.MAX_NON_TERMINAL_CYCLES
    return agent


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeTokeniser:
    eos_token_id = 0

    def __init__(self, reply):
        self.reply = reply

    def __call__(self, prompt, return_tensors=None):
        return _FakeInputs(input_ids=torch.zeros((1, 3), dtype=torch.long))

    def decode(self, tokens, skip_special_tokens=True):
        return self.reply


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids=None, **kwargs):
        return torch.zeros((1, 5), dtype=torch.long)


def _with_reply(agent, reply):
    agent.tokeniser = _FakeTokeniser(reply)
    agent.model = _FakeModel()
    return agent


# ------------------------------------------------------------------ prompt

def test_capped_report_menu_prompt_drops_think_spawn_tool():
    for role, has_spawned in (("executor", False), ("verifier", False), ("decomposer", True)):
        agent = _capped(_agent(role, has_spawned=has_spawned))
        prompt = agent._build_prompt(available_tools=["run_code"], requirements=[])
        for closed in ("- THINK", "If THINK", "- TOOL", "ACTION: TOOL"):
            assert closed not in prompt, (role, closed)
        assert "- REPORT" in prompt and "- DIE" in prompt, role
        assert "THINKING BUDGET EXHAUSTED" in prompt, role


def test_capped_decomposer_with_children_prompt_never_mentions_spawn():
    for generation in (0, 1):
        agent = _capped(_agent("decomposer", has_spawned=True, generation=generation))
        prompt = agent._build_prompt(available_tools=[], requirements=[])
        assert "SPAWN" not in prompt, generation
        assert "combine those results" in prompt, generation


def test_capped_decomposer_without_children_is_offered_spawn_not_report():
    agent = _capped(_agent("decomposer", has_spawned=False))
    assert agent.final_actions == ("SPAWN", "DIE")
    prompt = agent._build_prompt(available_tools=[], requirements=[])
    assert "- SPAWN" in prompt and "- DIE" in prompt
    assert "- REPORT" not in prompt and "If REPORT" not in prompt
    assert "- THINK" not in prompt
    assert "SPAWN your subtasks" in prompt


def test_prompt_unchanged_below_cap():
    agent = _agent()
    agent.non_terminal_cycles = agent.MAX_NON_TERMINAL_CYCLES - 1
    prompt = agent._build_prompt(available_tools=[], requirements=[])
    assert "- THINK" in prompt and "THINKING BUDGET EXHAUSTED" not in prompt


# ---------------------------------------------------------------- counting

def _scripted_run(agent, decisions):
    """Runs agent.run() once per decision with think()/decide() stubbed."""
    think_calls = []
    queue = list(decisions)

    def fake_think(*a, **kw):
        think_calls.append(1)
        return "FORCE_DECIDE"

    agent.think = fake_think
    agent.decide = lambda *a, **kw: queue.pop(0)
    for _ in range(len(decisions)):
        agent.run(available_tools=[])
    return think_calls


def test_run_loop_terminates_at_cap():
    agent = _agent()
    think_calls = []

    def fake_think(*a, **kw):
        think_calls.append(1)
        return "FORCE_DECIDE"

    agent.think = fake_think
    agent.decide = lambda *a, **kw: (
        ("REPORT", "answer") if agent.cycles_capped else ("THINK", "more reasoning")
    )
    executed = []
    agent.execute = lambda action, payload: executed.append(action)

    for _ in range(agent.MAX_NON_TERMINAL_CYCLES + 3):
        agent.run(available_tools=[])

    assert executed.count("THINK") == agent.MAX_NON_TERMINAL_CYCLES
    assert executed[agent.MAX_NON_TERMINAL_CYCLES] == "REPORT"
    # think() is not generated once the cap is hit.
    assert len(think_calls) == agent.MAX_NON_TERMINAL_CYCLES


def test_self_rejected_spawn_counts_toward_cap():
    agent = _agent("decomposer")
    _scripted_run(agent, [("SPAWN", '{"subtasks": [broken')] * agent.MAX_NON_TERMINAL_CYCLES)
    assert agent.awaiting is None
    assert agent.cycles_capped


def test_accepted_spawn_and_dispatched_tool_do_not_count():
    agent = _agent("decomposer")
    _scripted_run(agent, [("SPAWN", {"role": "executor", "task": "x"})])
    assert agent.awaiting == "children" and agent.non_terminal_cycles == 0

    agent = _agent("executor")
    agent.request_tool = lambda payload: setattr(agent, "fail_reason", None)
    _scripted_run(agent, [("TOOL", {"tool_name": "run_code", "args": {}})] * 6)
    assert agent.non_terminal_cycles == 0

    agent = _agent("executor")
    agent.request_tool = lambda payload: setattr(agent, "fail_reason", "refused")
    _scripted_run(agent, [("TOOL", {"tool_name": "run_code", "args": {}})])
    assert agent.non_terminal_cycles == 1


# ---------------------------------------------------------------- coercion

def test_capped_decide_converts_think_to_report():
    agent = _capped(_with_reply(_agent(), "ACTION: THINK\nPAYLOAD: The density is 7.8 g/cm3 for steel."))
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "REPORT"
    assert "7.8" in payload and "ACTION:" not in payload
    assert agent.cap_coerced_last_run


def test_capped_decide_converts_parse_failure_and_clears_fail_reason():
    agent = _capped(_with_reply(_agent(), "ACTION: TOOL\nI would run some code here."))
    action, _ = agent.decide(available_tools=[], requirements=[])
    assert action in ("REPORT", "DIE")
    assert agent.fail_reason is None

    agent = _capped(_with_reply(_agent(), "ACTION: SPAWNSUBTASKSXYZ\nPAYLOAD: something"))
    action, _ = agent.decide(available_tools=[], requirements=[])
    assert action in ("REPORT", "DIE")


def test_capped_decide_keeps_real_report_and_die():
    for reply, expected in (("ACTION: REPORT\nPAYLOAD: 42", "REPORT"),
                            ("ACTION: DIE\nPAYLOAD: impossible", "DIE")):
        agent = _capped(_with_reply(_agent(), reply))
        action, _ = agent.decide(available_tools=[], requirements=[])
        assert action == expected
        assert not agent.cap_coerced_last_run


def test_capped_unspawned_decomposer_keeps_valid_spawn_and_dies_otherwise():
    spawn = 'ACTION: SPAWN\nPAYLOAD: {"subtasks": [{"role": "executor", "task": "compute x", "dependencies": []}]}'
    agent = _capped(_with_reply(_agent("decomposer"), spawn))
    action, payload = agent.decide(available_tools=[], requirements=[])
    assert action == "SPAWN" and isinstance(payload, dict)

    for reply in ("ACTION: THINK\nPAYLOAD: I should plan the pieces first.",
                  "ACTION: REPORT\nPAYLOAD: The answer is 42.",
                  'ACTION: SPAWN\nPAYLOAD: {"subtasks": [ broken'):
        agent = _capped(_with_reply(_agent("decomposer"), reply))
        action, payload = agent.decide(available_tools=[], requirements=[])
        assert action == "DIE", reply
        assert "SPAWN" in payload


def test_coerce_dies_when_nothing_to_report():
    agent = _capped(_agent())
    action, payload = agent._coerce_final_action("THINK", {"tool_name": "run_code"})
    assert action == "DIE" and "Cycle cap" in payload


# ------------------------------------------- orchestrator: waiting on children

class _NullMemoryStore:
    def write(self, record_type, text, metadata):
        return 1

    def get_success_cache(self, description, task_id=None):
        return None


class _StubDecomposer:
    role = "decomposer"
    cap_coerced_last_run = False

    def __init__(self, agent_id, task_id):
        self.agent_id = agent_id
        self.task_id = task_id
        self.parent_id = None
        self.awaiting = "children"
        self.thought_process = ""
        self.cycles_capped = False
        self.runs = 0

    def run(self, *args, **kwargs):
        self.runs += 1
        self.thought_process += "x" * 200


def _decomposer_with_two_children():
    orch = Orchestrator(ColonyState(initial_budget=1000, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=_NullMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan", agent_id="p", status=1))
    orch.colony.register_agent(AgentNode(agent_id="p", role="decomposer", status="running",
                                         parent_id=None, task="plan", task_id="t-parent"))
    for child in ("c1", "c2"):
        orch.task_graph.add_task(TaskNode(task_id=f"t-{child}", description=child,
                                          agent_id=child, status=1))
        orch.colony.register_agent(AgentNode(agent_id=child, role="executor", status="running",
                                             parent_id="p", task=child, task_id=f"t-{child}"))
    parent = _StubDecomposer("p", "t-parent")
    orch.live_agents["p"] = parent
    return orch, parent


def _notify(orch, child):
    orch.task_graph.tasks[f"t-{child}"].status = 2
    event = Event(type="parent_notification", from_agent="orchestrator")
    event.payload.update({"parent_id": "p", "child_id": child, "result": f"{child} done"})
    orch.handle_parent_notification(event)


def test_decomposer_not_resumed_until_every_child_finishes():
    orch, parent = _decomposer_with_two_children()

    _notify(orch, "c1")
    assert parent.awaiting == "children"
    assert "c1 done" in parent.thought_process
    for _ in range(5):
        orch._run_live_agents()
    assert parent.runs == 0

    _notify(orch, "c2")
    assert parent.awaiting is None
    orch._run_live_agents()
    assert parent.runs == 1


def test_decomposer_with_open_child_is_not_ticked_even_if_gate_cleared():
    orch, parent = _decomposer_with_two_children()
    parent.awaiting = None
    orch._run_live_agents()
    assert parent.runs == 0 and parent.awaiting == "children"


def test_queued_overflow_counts_as_open_child():
    orch, parent = _decomposer_with_two_children()
    orch.pending_overflow["p"] = [{"task_id": "t-queued"}]
    _notify(orch, "c1")
    _notify(orch, "c2")
    assert parent.awaiting == "children"


def test_spawning_marks_parent_has_spawned():
    orch, _ = _decomposer_with_two_children()
    orch.task_graph.add_task(TaskNode(task_id="t-new", description="new"))
    assert not orch.colony.get_agent("p").has_spawned
    orch.spawn_agent(role="executor", task_id="t-new", parent_id="p")
    assert orch.colony.get_agent("p").has_spawned


def test_cap_events_are_tallied_for_the_report():
    orch, parent = _decomposer_with_two_children()
    parent.awaiting = None
    for child in ("c1", "c2"):
        orch.task_graph.tasks[f"t-{child}"].status = 2

    def run(*a, **kw):
        parent.cycles_capped = True
        parent.cap_coerced_last_run = True

    parent.run = run
    orch._run_live_agents()
    orch._run_live_agents()
    assert orch.colony.verdict_counts["cycle_cap_reached"] == 1
    assert orch.colony.verdict_counts["cycle_cap_coerced"] == 2


# ----------------------------------------- SPAWN that starts nothing must not hang

def _spawning_decomposer(budget=1000, admission=True):
    orch = Orchestrator(ColonyState(initial_budget=budget, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=_NullMemoryStore())
    orch.ADMISSION_CONTROL = admission
    orch.task_graph.add_task(TaskNode(task_id="t-parent", description="plan", agent_id="p", status=1))
    node = AgentNode(agent_id="p", role="decomposer", status="running", parent_id=None,
                     task="plan", task_id="t-parent", generation=1)
    orch.colony.register_agent(node)
    agent = Agent(tokeniser=None, model=None, message=orch.messenger, node=node)
    orch.live_agents["p"] = agent
    return orch, agent


def _spawn(orch, agent, payload):
    agent.request_spawn(payload)
    assert agent.awaiting == "children"
    orch._route_events(orch.messenger.drain())


def test_batch_with_every_subtask_malformed_releases_the_decomposer():
    orch, agent = _spawning_decomposer()
    _spawn(orch, agent, {"subtasks": ["just a string", {"role": "executor"}, {"task": ""}]})
    assert agent.awaiting is None
    assert "none of your requested subtasks were started" in agent.fail_reason
    assert agent.non_terminal_cycles == 1


def test_single_spawn_without_task_text_releases_the_decomposer():
    orch, agent = _spawning_decomposer()
    _spawn(orch, agent, {"role": "executor", "task": ""})
    assert agent.awaiting is None and agent.fail_reason


def _child_tasks(orch):
    return {tid: t for tid, t in orch.task_graph.tasks.items() if tid != "t-parent"}


def test_spawn_with_no_energy_closes_children_instead_of_orphaning_them():
    # The can_spawn fallback, reached when admission is off or its
    # reservations underestimate. With admission on, the batch is refused
    # before any child exists (next test).
    orch, agent = _spawning_decomposer(budget=0, admission=False)
    _spawn(orch, agent, {"subtasks": [{"role": "executor", "task": "compute x", "dependencies": []}]})

    (task_id, task), = _child_tasks(orch).items()
    assert task.status == 3 and "NOT STARTED" in task.result
    assert task_id in orch.unstarted_tasks

    # It never had a child and cannot start one, so it is not released to
    # SPAWN again: it goes down the DIE path, and stays gated until then.
    events = orch.messenger.drain()
    assert agent.awaiting == "children"
    assert [e.from_agent for e in events if e.type == "failure_request"] == ["p"]
    notifications = [e for e in events if e.type == "parent_notification"]
    assert [e.payload["parent_id"] for e in notifications] == ["p"]

    # The old failure: a pending, agentless task that the next tick started
    # with no parent.
    agents_before = set(orch.colony.agents)
    orch._process_unblocked_tasks()
    assert set(orch.colony.agents) == agents_before


def test_admission_refuses_a_batch_no_child_of_which_fits():
    """Admission on: nothing is created, and a decomposer that never had a
    child goes down the DIE path exactly as the can_spawn fallback sends it."""
    orch, agent = _spawning_decomposer(budget=0)
    _spawn(orch, agent, {"subtasks": [{"role": "executor", "task": "compute x", "dependencies": []}]})

    assert _child_tasks(orch) == {}
    assert orch.colony.verdict_counts["admission_spawn_refused"] == 1
    events = orch.messenger.drain()
    assert agent.awaiting == "children"
    assert [e.from_agent for e in events if e.type == "failure_request"] == ["p"]


def test_partial_energy_closes_only_the_unstarted_children():
    cost = Orchestrator(ColonyState(initial_budget=1, goal_embedding=None), TaskGraph(),
                        Messenger(), memory_store=_NullMemoryStore())
    cost = cost.energy_when_new_by_role.get("executor", cost.energy_when_new)
    orch, agent = _spawning_decomposer(budget=cost, admission=False)
    _spawn(orch, agent, {"subtasks": [
        {"label": "a", "role": "executor", "task": "compute x", "dependencies": []},
        {"label": "b", "role": "executor", "task": "compute y", "dependencies": []},
        {"role": "executor", "task": "combine y", "dependencies": ["b"]},
    ]})

    # Fan-out cap is 2 for a non-root decomposer: "a" started, "b" closed,
    # the third is queued.
    statuses = sorted(t.status for t in _child_tasks(orch).values())
    assert statuses == [1, 3]
    assert len(orch.unstarted_tasks) == 1
    # One child is really running, so the parent keeps waiting for it.
    assert agent.awaiting == "children" and agent.fail_reason is None
    assert len([e for e in orch.messenger.drain() if e.type == "parent_notification"]) == 1

    # "a" finishing drains the queued subtask. Its dependency "b" is already
    # closed, so it is not left blocked; with no energy it is closed too,
    # instead of sitting pending as an open child the parent waits on forever.
    a_id = next(tid for tid, t in _child_tasks(orch).items() if t.status == 1)
    orch.task_graph.complete_task(a_id)
    orch._drain_pending_overflow("p")
    assert "p" not in orch.pending_overflow
    assert sorted(t.status for t in _child_tasks(orch).values()) == [2, 3, 3]
    assert orch._open_child_task_ids("p") == []


def test_task_added_after_its_dependency_closed_is_not_blocked():
    graph = TaskGraph()
    graph.add_task(TaskNode(task_id="done", description="d", status=2))
    graph.add_task(TaskNode(task_id="closed", description="c", status=3))
    graph.add_task(TaskNode(task_id="open", description="o", status=1))
    graph.add_task(TaskNode(task_id="t", description="t", dependencies=["done", "closed", "open"]))
    assert graph.tasks["t"].in_degree == 1


def test_valid_batch_keeps_the_decomposer_waiting():
    orch, agent = _spawning_decomposer()
    _spawn(orch, agent, {"subtasks": ["junk", {"role": "executor", "task": "compute x", "dependencies": []}]})
    assert agent.awaiting == "children"
    assert agent.fail_reason is None
    assert agent.non_terminal_cycles == 0


def test_rejected_batch_keeps_its_rejection_reason():
    from agent_node import EXEMPLAR_SUBTASK_DESCRIPTIONS
    orch, agent = _spawning_decomposer()
    copied = next(iter(EXEMPLAR_SUBTASK_DESCRIPTIONS))
    _spawn(orch, agent, {"subtasks": [{"role": "executor", "task": copied, "dependencies": []}]})
    assert "REJECTED" in agent.fail_reason
    assert "none of your requested subtasks were started" not in agent.fail_reason


# ------------------------------------------------------------- final sweep

def test_spawn_payload_check_requires_a_startable_subtask():
    ok = Agent._is_spawn_payload
    assert ok({"subtasks": [{"role": "executor", "task": "compute x"}]})
    assert ok({"subtasks": ["junk", {"taask": "compute x"}]})  # orchestrator repairs the key
    assert ok({"role": "executor", "task": "compute x"})
    assert not ok({"subtasks": ["junk"]})
    assert not ok({"subtasks": [{"role": "executor"}, {"task": "  "}]})
    assert not ok({"role": "executor", "task": ""})
    assert not ok('{"subtasks": [')


def test_capped_decomposer_junk_batch_dies_at_decide():
    agent = _capped(_with_reply(_agent("decomposer"),
                                'ACTION: SPAWN\nPAYLOAD: {"subtasks": ["junk", "more junk"]}'))
    action, _ = agent.decide(available_tools=[], requirements=[])
    assert action == "DIE"


def test_capped_decomposer_whose_spawn_starts_nothing_is_routed_to_failure():
    orch, agent = _spawning_decomposer()
    _capped(agent)
    _spawn(orch, agent, {"subtasks": [{"role": "executor"}]})
    failures = [e for e in orch.messenger.drain() if e.type == "failure_request"]
    assert [e.from_agent for e in failures] == ["p"]
    assert agent.awaiting == "children"  # not released for another try


def _completed_subtask(orch, task_id="t-done"):
    """A direct subtask of t-parent that finished: what "has something to
    REPORT" means now, instead of has_spawned alone."""
    orch.task_graph.add_task(TaskNode(task_id=task_id, description="compute w", status=2,
                                      parent_task_id="t-parent"))


def test_out_of_energy_with_finished_children_is_released_to_report():
    orch, agent = _spawning_decomposer(budget=0)
    agent.node.has_spawned = True
    _completed_subtask(orch)
    _spawn(orch, agent, {"subtasks": [{"role": "executor", "task": "compute x", "dependencies": []}]})
    assert agent.awaiting is None
    assert "REPORT what your completed subtasks produced" in agent.fail_reason
    assert not [e for e in orch.messenger.drain() if e.type == "failure_request"]


def test_budget_refused_spawn_closes_spawn_instead_of_offering_it_again():
    # The loop this closes: SPAWN -> refused -> released -> full think() +
    # decide() with SPAWN still on the menu -> the same SPAWN, until the cap.
    orch, agent = _spawning_decomposer(budget=0)
    agent.node.has_spawned = True
    _completed_subtask(orch)
    _spawn(orch, agent, {"subtasks": [{"role": "executor", "task": "compute x", "dependencies": []}]})

    assert agent.spawn_closed and agent.final_only and not agent.cycles_capped
    assert agent.final_actions == ("REPORT", "DIE")
    assert orch.colony.verdict_counts["spawn_closed_no_energy"] == 1
    prompt = agent._build_prompt(available_tools=[], requirements=[])
    for closed in ("\n- SPAWN", "\n- THINK", "If SPAWN"):
        assert closed not in prompt, closed
    assert "NO ENERGY FOR NEW SUBTASKS" in prompt
    assert "THINKING BUDGET EXHAUSTED" not in prompt


def test_malformed_batch_release_does_not_close_spawn():
    orch, agent = _spawning_decomposer()
    _spawn(orch, agent, {"subtasks": [{"role": "executor"}]})
    assert agent.awaiting is None and not agent.spawn_closed


def test_spawn_closed_agent_skips_think_and_cannot_spawn_again():
    agent = _with_reply(_agent("decomposer", has_spawned=True),
                        'ACTION: SPAWN\nPAYLOAD: {"subtasks": [{"role": "executor", "task": "compute x"}]}')
    agent.spawn_closed = True

    def _no_think(*args, **kwargs):
        raise AssertionError("think() ran on a spawn_closed agent")

    agent.think = _no_think
    action = agent.run(available_tools=[])
    assert action == "DIE"
    assert agent.awaiting is None
    assert "No energy left for new subtasks" in agent.message.drain()[0].payload["result"]


def test_retiring_a_decomposer_drops_its_queued_overflow():
    orch, agent = _spawning_decomposer()
    orch.pending_overflow["p"] = [dict(description="queued", role="executor", parent_id="p",
                                       dependencies=[], task_id="t-queued")]
    before = orch.committed_energy()

    orch._retire_agent("p", "t-parent")

    assert "p" not in orch.pending_overflow
    assert orch.committed_energy() == before - orch._new_task_reservation("executor")
    assert orch.colony.verdict_counts["overflow_dropped_parent_retired"] == 1


class _CacheHitStore(_NullMemoryStore):
    def get_success_cache(self, description, task_id=None):
        return SimpleNamespace(hit={"result": "cached answer", "task_id": "t-donor",
                                    "outcome": "accepted"},
                               hit_score=0.9, negative_count=0, negative_outcomes={})


def test_success_cache_completion_drains_queued_overflow():
    orch, parent = _decomposer_with_two_children()
    orch.memory_store = _CacheHitStore()
    orch.task_graph.tasks["t-c2"].status = 2
    orch.pending_overflow["p"] = [dict(description="queued", role="executor", parent_id="p",
                                       dependencies=[], task_id="t-queued")]

    orch._kill_and_respawn("c1", "t-c1", "executor", "p")

    assert orch.task_graph.tasks["t-c1"].status == 2
    assert "p" not in orch.pending_overflow
    assert "t-queued" in orch.task_graph.tasks


def test_respawn_without_energy_abandons_task_and_tells_parent():
    orch, parent = _decomposer_with_two_children()
    orch.colony.budget_remaining = 0

    orch._kill_and_respawn("c1", "t-c1", "executor", "p")

    task = orch.task_graph.tasks["t-c1"]
    assert task.status == 3 and "no energy left to respawn" in task.result
    assert "t-c1" in orch.abandoned_tasks
    notes = [e for e in orch.messenger.drain() if e.type == "parent_notification"]
    assert [e.payload["parent_id"] for e in notes] == ["p"]


# ------------------------------------------- confirming the strip in a run
#
# "actions forced by the cap" counts only decisions where the model IGNORED
# the reduced menu and the backstop overrode it. It reads 0 when the strip
# works, so it cannot be what confirms enforcement. "decisions at the cap"
# can: every one is a real decide() whose prompt had THINK/SPAWN removed.

def _capped_executor_in_colony(reply):
    orch = Orchestrator(ColonyState(initial_budget=1000, goal_embedding=None),
                        TaskGraph(), Messenger(), memory_store=_NullMemoryStore())
    orch.task_graph.add_task(TaskNode(task_id="t-1", description="do the thing",
                                      agent_id="a-1", status=1))
    node = AgentNode(agent_id="a-1", role="executor", status="running", parent_id=None,
                     task="do the thing", task_id="t-1", generation=1)
    orch.colony.register_agent(node)
    agent = _capped(_with_reply(
        Agent(tokeniser=None, model=None, message=orch.messenger, node=node), reply))
    orch.live_agents["a-1"] = agent
    return orch, agent


def _cap_ledger(orch, capsys):
    capsys.readouterr()
    orch._print_energy_report()
    out = capsys.readouterr().out
    return {label.strip(): value.strip()
            for label, _, value in (l.partition(":") for l in out.splitlines())
            if label.strip() in ("agents that reached it", "decisions at the cap",
                                 "model kept to the menu", "actions forced by the cap")}


def test_a_compliant_capped_decision_is_counted_even_though_nothing_was_forced(capsys):
    orch, agent = _capped_executor_in_colony("ACTION: REPORT\nPAYLOAD: The answer is 42.")
    orch._run_live_agents()

    assert not agent.cap_coerced_last_run
    assert orch.colony.verdict_counts.get("cycle_cap_decisions") == 1
    assert "cycle_cap_coerced" not in orch.colony.verdict_counts
    ledger = _cap_ledger(orch, capsys)
    assert ledger["decisions at the cap"] == "1"
    assert ledger["model kept to the menu"] == "1"
    assert ledger["actions forced by the cap"] == "0"


def test_a_capped_decision_that_ignored_the_menu_is_counted_as_forced(capsys):
    orch, agent = _capped_executor_in_colony(
        "ACTION: THINK\nPAYLOAD: The density is 7.8 g/cm3 for steel.")
    orch._run_live_agents()

    assert agent.cap_coerced_last_run
    ledger = _cap_ledger(orch, capsys)
    assert ledger["decisions at the cap"] == "1"
    assert ledger["model kept to the menu"] == "0"
    assert ledger["actions forced by the cap"] == "1"


def test_an_uncapped_run_is_not_a_capped_decision():
    orch, agent = _capped_executor_in_colony("ACTION: REPORT\nPAYLOAD: 42.")
    agent.non_terminal_cycles = 0
    agent.think = lambda *a, **kw: "FORCE_DECIDE"
    orch._run_live_agents()
    assert "cycle_cap_decisions" not in orch.colony.verdict_counts


def test_a_spawn_closed_decision_is_counted_with_the_cap_decisions(capsys):
    # Its override lands in cycle_cap_coerced, so without counting the
    # decision the ledger showed more actions forced than decisions made.
    orch, agent = _capped_executor_in_colony(
        'ACTION: SPAWN\nPAYLOAD: {"role": "executor", "task": "compute x"}')
    agent.node.role = "decomposer"
    agent.node.has_spawned = True
    agent.non_terminal_cycles = 1
    agent.spawn_closed = True
    orch._run_live_agents()

    assert agent.cap_coerced_last_run
    ledger = _cap_ledger(orch, capsys)
    assert ledger["decisions at the cap"] == "1"
    assert ledger["actions forced by the cap"] == "1"
    assert ledger["model kept to the menu"] == "0"
