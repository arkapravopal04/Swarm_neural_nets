"""
Drives the REAL Orchestrator tick loop with scripted agents.

Only Agent.think and Agent.decide are replaced (class-level, restored after
every run). Agent.run, execute, request_spawn, report, die and the whole
Orchestrator -- spawn, injection billing, judge routing, WARN loop, cycle cap,
respawn, TASK TOO LARGE conversion, overflow queue, per-task ceiling -- run
unmodified, so the energy figures here are what the real code bills.

What a scripted cycle costs is modelled on the real generation lengths:
  think():  role cap (decomposer 64 / executor 128 tokens) x ~4 chars, times
            a fraction for stripping and early degeneracy stops;
  decide(): REPORT up to REPORT_MAX_NEW_TOKENS (200), SPAWN JSON up to the
            400-token decide() budget, THINK/DIE short.
Only len(thought_process) growth matters to the bill, so the text is filler.

Point HIVE_SRC at a directory to test a different copy of the code (e.g. a
pinned snapshot); defaults to the repository this file lives in.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HIVE_SRC = os.environ.get("HIVE_SRC", _REPO)
if HIVE_SRC not in sys.path:
    sys.path.insert(0, HIVE_SRC)

import agent_node  # noqa: E402
from agent_node import Agent, think_token_cap, _ActionPayloadStop  # noqa: E402
from colony_state import ColonyState  # noqa: E402
from event_queue import Messenger  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from task_graph import TaskGraph  # noqa: E402

CPT = 4  # chars per token -- the same proxy Orchestrator.ENERGY_CHARS_PER_TOKEN uses

ROOT_PROMPT = "Organise a community park cleanup day for forty volunteers"


# ---------------------------------------------------------------------------
# Plans: what each task's agents will do.

@dataclass
class Behave:
    """What ONE agent on a task does. A task's agents take Plan.attempts in
    order (the last entry repeats)."""
    action: str = "report"          # report | die | too_large | think_loop | spawn
    thinks: int = 0                 # THINK decisions before the terminal/spawn decision
    verdicts: tuple = ("promote",)  # judge verdicts on successive REPORTs (last repeats)
    report_thinks: int = 0          # decomposer: THINKs after children finish, before REPORT


@dataclass
class Plan:
    name: str
    kind: str = "exec"              # exec | dec  (the role the PARENT asks for is forced
                                    # by Orchestrator._enforce_child_role anyway)
    children: List["Plan"] = field(default_factory=list)
    attempts: List[Behave] = field(default_factory=lambda: [Behave()])
    # For a merged (bundled) child: probability its executor DIEs TASK TOO LARGE.
    bundle_too_large: float = 0.0

    def behave(self, i: int) -> Behave:
        return self.attempts[min(i, len(self.attempts) - 1)]

    def leaves(self) -> int:
        return 1 if not self.children else sum(c.leaves() for c in self.children)

    @staticmethod
    def attempt_succeeds(b: "Behave", cap: int = 4) -> bool:
        """Would this one agent reach a promote under the real cycle cap?
        Cycle 1 is think-only (non-terminal 1), each THINK decision is +1,
        each WARN is +1 and a WARN that reaches the cap respawns; EXECUTE and
        DIE respawn. SPAWN hands off and is not counted."""
        if b.action == "die":
            return False
        if b.action == "too_large":
            return True  # converts; the decomposer attempt is judged on its own
        # A capped agent still gets its (forced) REPORT judged once.
        nt = 1 + (b.thinks if b.action != "think_loop" else cap)
        if b.action == "spawn":
            nt = 1 + b.thinks + b.report_thinks
        for i in range(8):
            v = b.verdicts[min(i, len(b.verdicts) - 1)]
            if v == "promote":
                return True
            if v == "execute":
                return False
            nt += 1
            if nt >= cap:
                return False
        return False

    def eventually_succeeds(self, max_agents: int) -> bool:
        """True if some agent within max_agents would get a promote -- used to
        tell a false kill (ceiling stopped a task that would have finished)
        from a justified one."""
        return any(self.attempt_succeeds(self.behave(i)) for i in range(max_agents))


MERGE_JOIN = " together with "


# ---------------------------------------------------------------------------
# Cost model for scripted generations.

@dataclass
class CostModel:
    think_frac: tuple = (0.70, 1.00)     # share of the think() cap kept after stripping
    report_tokens: tuple = (140, 200)    # REPORT decide(); 200 = REPORT_MAX_NEW_TOKENS
    think_decide_tokens: tuple = (40, 90)
    die_tokens: tuple = (40, 80)
    spawn_tokens_base: int = 30
    spawn_tokens_per_child: int = 45
    spawn_tokens_cap: int = 400          # decide()'s max_new_tokens
    critique_chars: tuple = (150, 350)   # tier-3 reason length -> 1..3 energy
    report_words: int = 55               # survives trim_to_sentences(3, 60 words)


_WORDS = ("gloves bags rakes water shade signage parking tables first-aid kit "
          "volunteers leads schedule shifts sorting recycling compost landfill "
          "hauler permit route map briefing safety vests sunscreen snacks "
          "registration wristbands photos tally weigh debris hazards sharps "
          "containers pickup drop-off morning afternoon marshal coordinator").split()


class World:
    """Holds the plan and every per-agent script cursor. One per run."""

    def __init__(self, root: Plan, seed: int = 0, cost: Optional[CostModel] = None):
        self.root = root
        self.rng = random.Random(seed)
        self.cost = cost or CostModel()
        self.plans: Dict[str, Plan] = {}
        self._index(root)
        self.agent_state: Dict[str, dict] = {}
        self.agents_per_task: Dict[str, int] = {}
        self.report_seq = 0
        self.merges = 0

    def _index(self, plan: Plan):
        self.plans[plan.name] = plan
        for c in plan.children:
            self._index(c)

    # -- plan lookup ------------------------------------------------------
    def plan_for(self, description: str) -> Plan:
        plan = self.plans.get(description)
        if plan is not None:
            return plan
        if MERGE_JOIN in description:
            parts = [self.plan_for(p) for p in description.split(MERGE_JOIN)]
            if all(p.kind == "dec" for p in parts):
                merged = Plan(description, "dec",
                              children=[c for p in parts for c in p.children],
                              attempts=[parts[0].behave(0)])
            else:
                # An executor handed several items. Code comments record this
                # "reliably DIEd as TASK TOO LARGE"; bundle_too_large is the
                # probability, set per experiment.
                too_large = self.rng.random() < self.bundle_too_large
                merged = Plan(description, "exec", children=parts,
                              attempts=([Behave("too_large", thinks=1), Behave("spawn")]
                                        if too_large else [Behave("report", thinks=1)]))
            self.plans[description] = merged
            return merged
        raise KeyError(f"no plan for task description {description!r}")

    bundle_too_large = 0.7

    # -- per-agent state --------------------------------------------------
    def state(self, agent: Agent) -> dict:
        st = self.agent_state.get(agent.agent_id)
        if st is None:
            plan = self.plan_for(agent.task)
            idx = self.agents_per_task.get(agent.task_id, 0)
            self.agents_per_task[agent.task_id] = idx + 1
            st = dict(plan=plan, idx=idx, behave=plan.behave(idx), thinks=0,
                      spawned=False, report_thinks=0, reports=0, verdicts_used=0)
            self.agent_state[agent.agent_id] = st
        return st

    # -- text -------------------------------------------------------------
    def filler(self, n_chars: int) -> str:
        return "t" * max(0, n_chars)

    def report_text(self, name: str) -> str:
        self.report_seq += 1
        words = [self.rng.choice(_WORDS) for _ in range(self.cost.report_words)]
        third = len(words) // 3
        s = [" ".join(words[i * third:(i + 1) * third]) for i in range(3)]
        return (f"For {name[:40]}, answer {self.report_seq}: {s[0]}. "
                f"Then {s[1]}. Finally {s[2]}.")

    def uni(self, lo_hi) -> float:
        lo, hi = lo_hi
        return self.rng.uniform(lo, hi)


WORLD: Optional[World] = None


# ---------------------------------------------------------------------------
# Scripted think() / decide() -- everything around them is real.

def _fake_think(self: Agent, available_roles=None, available_tools=None, requirements=None):
    w = WORLD
    w.state(self)
    self.think_cycle += 1
    n = int(think_token_cap(self.role) * CPT * w.uni(w.cost.think_frac))
    self.thought_process += w.filler(n)
    self._total_generated += think_token_cap(self.role)
    return "THINK" if self.think_cycle <= self.THINK_CYCLES_BEFORE_DECIDE else "FORCE_DECIDE"


def _policy(self: Agent, st: dict):
    w = WORLD
    c = w.cost
    b: Behave = st["behave"]
    plan: Plan = st["plan"]
    think_text = f"Still working out {plan.name[:40]} step by step."

    if self.role == "decomposer":
        if not st["spawned"] and not getattr(self.node, "has_spawned", False):
            if st["thinks"] < b.thinks:
                st["thinks"] += 1
                return "THINK", think_text, int(w.uni(c.think_decide_tokens))
            if plan.children:
                st["spawned"] = True
                subtasks = [{"role": "executor" if ch.kind == "exec" else "decomposer",
                             "task": ch.name, "dependencies": []} for ch in plan.children]
                toks = min(c.spawn_tokens_cap,
                           c.spawn_tokens_base + c.spawn_tokens_per_child * len(subtasks))
                return "SPAWN", {"subtasks": subtasks}, toks
        if st["report_thinks"] < b.report_thinks:
            st["report_thinks"] += 1
            return "THINK", think_text, int(w.uni(c.think_decide_tokens))
        st["reports"] += 1
        return "REPORT", w.report_text(plan.name), int(w.uni(c.report_tokens))

    # executor / verifier
    if b.action in ("report", "spawn"):
        if st["thinks"] < b.thinks:
            st["thinks"] += 1
            return "THINK", think_text, int(w.uni(c.think_decide_tokens))
        st["reports"] += 1
        return "REPORT", w.report_text(plan.name), int(w.uni(c.report_tokens))
    if b.action == "think_loop":
        # never chooses a terminal action; the cycle cap coerces a REPORT
        return "THINK", w.report_text(plan.name), int(w.uni(c.think_decide_tokens))
    if st["thinks"] < b.thinks:
        st["thinks"] += 1
        return "THINK", think_text, int(w.uni(c.think_decide_tokens))
    if b.action == "too_large":
        return "DIE", "TASK TOO LARGE: this needs to be split into several parts.", \
            int(w.uni(c.die_tokens))
    return "DIE", "I could not complete this subtask.", int(w.uni(c.die_tokens))


def _fake_decide(self: Agent, available_roles=None, available_tools=None, requirements=None):
    w = WORLD
    st = w.state(self)
    action, payload, toks = _policy(self, st)
    self.thought_process += "\n" + w.filler(toks * CPT) + "\n"
    self.cap_coerced_last_run = False
    if self.cycles_capped:
        final = self.final_actions
        if action not in final or (action == "SPAWN" and not self._is_spawn_payload(payload)):
            action, payload = self._coerce_final_action(action, payload)
            self.cap_coerced_last_run = True
    return action, payload


class ScriptJudge:
    """Returns the verdicts the plan scripts. Tier 3 (billed) on promote and
    execute, as the real judge's deep_critique does on promotion attempts;
    tier 2 (free) on warn."""

    def decide(self, agent, output=None, output_type="text", output_embedding=None,
               target_embedding=None, needs_deep_check=False):
        w = WORLD
        st = w.agent_state.get(agent.agent_id)
        if st is None:
            v = "promote"
        else:
            vs = st["behave"].verdicts
            v = vs[min(st["verdicts_used"], len(vs) - 1)]
            st["verdicts_used"] += 1
        crit = "c" * int(w.uni(w.cost.critique_chars))
        if v == "warn":
            return {"verdict": "warn", "reason": "drifted from the subtask", "tier": 2}
        return {"verdict": v, "reason": crit, "tier": 3}


class _NullMemory:
    def write(self, *a, **k):
        return 1

    def get_success_cache(self, description):
        return None


class _Null:
    def write(self, *_):
        return 0

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Instrumented orchestrator: records, changes nothing.

class InstrumentedOrchestrator(Orchestrator):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.task_roles: Dict[str, List[str]] = {}
        self.batch_sizes: Dict[str, List[int]] = {}

    def spawn_agent(self, role, task_id, parent_id=None, ghost_context=None):
        aid = super().spawn_agent(role, task_id, parent_id, ghost_context)
        if aid is not None:
            self.task_roles.setdefault(task_id, []).append(role)
        return aid

    def _handle_spawn_request(self, event):
        spawner = self.colony.get_agent(event.from_agent)
        subs = event.payload.get("subtasks") or []
        if spawner is not None:
            self.batch_sizes.setdefault(spawner.task_id, []).append(len(subs))
        return super()._handle_spawn_request(event)

    def handle_failure(self, event):
        p = event.payload
        if (p.get("role") == "executor" and p.get("judge_verdict") is None
                and str(p.get("result", "")).startswith("TASK TOO LARGE:")):
            tid = p.get("task_id")
            self.conversion_sunk = getattr(self, "conversion_sunk", {})
            self.conversion_sunk[tid] = self.colony.task_energy_spent.get(tid, 0)
        return super().handle_failure(event)


# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    outcome: str                      # success | root_abandoned | energy_death | tick_ceiling
    budget: int
    spent: int
    task_spent: Dict[str, int]
    task_roles: Dict[str, List[str]]
    task_desc: Dict[str, str]
    task_status: Dict[str, int]
    batch_sizes: Dict[str, List[int]]
    abandon_reasons: Dict[str, str]
    unstarted: int
    verdicts: Dict[str, int]
    ticks: int
    leaves_planned: int
    leaves_done: int
    orch: object = None
    world: object = None
    extra: dict = field(default_factory=dict)


def run_colony(root: Plan, budget: int, seed: int = 0, orch_cls=InstrumentedOrchestrator,
               orch_kwargs: Optional[dict] = None, cost: Optional[CostModel] = None,
               bundle_too_large: float = 0.7, max_ticks: int = 800,
               quiet: bool = True, keep: bool = False) -> RunResult:
    global WORLD
    WORLD = World(root, seed=seed, cost=cost)
    WORLD.bundle_too_large = bundle_too_large

    colony = ColonyState(initial_budget=budget, goal_embedding=None)
    orch = orch_cls(colony, TaskGraph(), Messenger(), judge=ScriptJudge(),
                    memory_store=_NullMemory(), budget_override=budget,
                    **(orch_kwargs or {}))
    orch.model, orch.tokeniser = object(), object()

    orig_think, orig_decide = Agent.think, Agent.decide
    Agent.think, Agent.decide = _fake_think, _fake_decide
    # _retire_agent runs gc.collect() on every kill to free VRAM; with no
    # model loaded it only costs wall-clock (~0.1s per kill).
    import gc
    orig_collect = gc.collect
    gc.collect = lambda *a, **k: 0
    old_stdout = sys.stdout
    if quiet:
        sys.stdout = _Null()
    outcome = "tick_ceiling"
    try:
        orch.initialize_colony(root.name)
        while True:
            if not orch.task_graph.tasks[orch.root_task_id].agent_id:
                outcome = "root_refused"   # admission refused the bootstrap
                break
            if orch.tick_count >= max_ticks:
                outcome = "tick_ceiling"
                break
            alive = orch.tick()
            if not alive:
                rt = orch.task_graph.tasks.get(orch.root_task_id)
                if rt is not None and rt.status == 2:
                    outcome = "success"
                elif rt is not None and rt.status == 3:
                    outcome = "root_abandoned"
                else:
                    outcome = "energy_death"
                break
    finally:
        sys.stdout = old_stdout
        Agent.think, Agent.decide = orig_think, orig_decide
        gc.collect = orig_collect

    tasks = orch.task_graph.tasks
    # Leaf coverage: DISTINCT planned leaves (by description) that some task
    # completed. A merged bundle answered by one executor covers one of its
    # leaves (it is one agent's partial answer to several items).
    covered = set()
    for t in tasks.values():
        if t.status != 2:
            continue
        try:
            p = WORLD.plan_for(t.description)
        except KeyError:
            continue
        if not p.children and p.kind == "exec":
            covered.add(p.name)
        elif orch.task_roles.get(t.task_id, ["executor"])[-1] == "executor":
            first = p
            while first.children:
                first = first.children[0]
            covered.add(first.name)
    done = len(covered)
    res = RunResult(
        outcome=outcome, budget=budget,
        spent=colony.starting_budget - colony.budget_remaining,
        task_spent=dict(colony.task_energy_spent),
        task_roles=dict(orch.task_roles),
        task_desc={tid: t.description for tid, t in tasks.items()},
        task_status={tid: t.status for tid, t in tasks.items()},
        batch_sizes=dict(orch.batch_sizes),
        abandon_reasons=dict(orch.abandon_reasons),
        unstarted=len(orch.unstarted_tasks),
        verdicts=dict(colony.verdict_counts),
        ticks=orch.tick_count,
        leaves_planned=root.leaves(),
        leaves_done=done,
        extra=dict(conversion_sunk=dict(getattr(orch, "conversion_sunk", {})),
                   completed_over_ceiling=dict(orch.completed_over_ceiling)),
    )
    if keep:
        res.orch, res.world = orch, WORLD
    return res


# ---------------------------------------------------------------------------
# Plan builders

def exec_plan(name, *attempts, children=None):
    return Plan(name, "exec", children=list(children or []),
                attempts=list(attempts) or [Behave()])


def dec_plan(name, children, *attempts):
    return Plan(name, "dec", children=list(children), attempts=list(attempts) or [Behave("spawn")])


def healthy_tree(k_root=3, k_dec=2, prefix="Part"):
    decs = []
    for i in range(k_root):
        kids = [exec_plan(f"{prefix} {i + 1}.{j + 1} of the cleanup plan")
                for j in range(k_dec)]
        decs.append(dec_plan(f"{prefix} {i + 1} of the cleanup plan", kids))
    return dec_plan(ROOT_PROMPT, decs)
