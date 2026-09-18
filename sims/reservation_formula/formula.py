"""
The proposed reservation / admission / kill-ceiling formula, prototyped as an
Orchestrator subclass so it runs inside the real tick loop. Nothing in the
architecture is edited; every hook calls through to the real method.

    R(executor)       = e
    R(decomposer, k)  = alpha + beta*k + sum R(children)
    R(root, k)        = R(decomposer, k) + sigma
    admission:  spent + sum remaining(unfinished) (after the change) <= B
    kill:       C = m * own_cost      own_cost = e | alpha+beta*k | alpha+beta*k+sigma

Where the proposal leaves a choice open, it is a switch here, so each reading
can be tested rather than assumed:

  provisional   reservation of a decomposer that has not SPAWNed yet (k unknown):
                "lazy"  -> k = 0, nothing reserved for its future children
                "min"   -> k = 1: its cheapest decomposition (one child)
                "exp"   -> k = 2: a typical decomposition
                "cap"   -> k = fan-out cap (3 root / 2 other), worst case
  merge         how k is lowered when a batch is not admissible:
                "bundle"   -> the overflow is merged into one child (old path)
                "truncate" -> the overflow is dropped
  floor         amount held back from B (the colony dies at <= 5 remaining)
  scope         "task"  -> C is checked against the task's cumulative spend
                           (all its agents, like today's per-task ceiling)
                "agent" -> C is checked against the current agent's own spend
  sunk          False   -> a TASK TOO LARGE conversion keeps the executor's spend
                           on the node and prices the node as a fresh decomposer
                           (the letter of the proposal)
                True    -> spend at conversion is carried as `sunk`: added to the
                           node's reservation and to its ceiling
  respawn       "letter"  -> a respawn is admission-checked but adds no reservation
                "reserve" -> a respawn re-reserves the node's own cost (sunk := spent)
  task_cap      None    -> no bound on a task's total across attempts (beyond
                           MAX_TASK_ATTEMPTS) once `respawn="reserve"` resets sunk
                A       -> also stop when the task's cumulative spend reaches
                           conv_sunk + A * own_cost (two-level ceiling)
  dynamic       False   -> C = m * own_cost, independent of the rest of the colony
                True    -> a node may spend past its reservation only while the
                           colony has unreserved slack (keeps the admission
                           invariant true after admission, not only at it)
"""
from collections import Counter
from dataclasses import dataclass

from harness import InstrumentedOrchestrator, MERGE_JOIN


@dataclass
class NodeRes:
    kind: str            # exec | dec | root
    k: int = 0
    spawned: bool = False
    sunk: int = 0
    conv_sunk: int = 0   # spend carried in by a TASK TOO LARGE conversion only
    gen: int = 0


class ReservationOrchestrator(InstrumentedOrchestrator):
    def __init__(self, *a, e=18.0, alpha=19.0, beta=5.0, sigma=0.0, m=2.0,
                 floor=0, scope="task", provisional="lazy", sunk=False,
                 respawn="letter", dynamic=False, admission=True, ceiling=True,
                 merge="bundle", dynamic_root=False, task_cap=None, **k):
        super().__init__(*a, **k)
        self.p = dict(e=e, alpha=alpha, beta=beta, sigma=sigma, m=m)
        self.floor = floor
        self.scope = scope
        self.provisional = provisional
        self.sunk_mode = sunk
        self.respawn_mode = respawn
        self.dynamic = dynamic
        self.admission_on = admission
        self.ceiling_on = ceiling
        self.merge = merge
        self.dynamic_root = dynamic_root
        # Second ceiling level: the task's cumulative spend across ALL its
        # attempts may not pass conv_sunk + task_cap * own_cost.
        self.task_cap = task_cap
        self.nodes = {}
        self.counters = Counter()
        self._converting = None
        self.max_committed = 0

    # ------------------------------------------------------------------ R
    def _cap(self, n: NodeRes) -> int:
        return self.MAX_SUBTASKS_ROOT if n.kind == "root" else self.MAX_SUBTASKS_NON_ROOT

    def _child_kind(self, n: NodeRes) -> str:
        # Orchestrator._enforce_child_role: the root's children are
        # decomposers, everyone else's are executors.
        return "dec" if n.kind == "root" else "exec"

    def _kprov(self, n: NodeRes) -> int:
        return {"lazy": 0, "min": 1, "exp": 2}.get(self.provisional, self._cap(n))

    def own_cost(self, n: NodeRes) -> float:
        p = self.p
        if n.kind == "exec":
            return p["e"]
        k = n.k if n.spawned else self._kprov(n)
        c = p["alpha"] + p["beta"] * k
        if n.kind == "root":
            c += p["sigma"]
        return c

    def r_new(self, kind: str) -> float:
        """Reservation of a node about to be created (nothing spent yet)."""
        n = NodeRes(kind=kind)
        return self.own_cost(n) + self._phantom(n)

    def _phantom(self, n: NodeRes) -> float:
        if n.kind == "exec" or n.spawned:
            return 0.0
        return self._kprov(n) * self.r_new(self._child_kind(n))

    def _spent(self, tid) -> int:
        return self.colony.task_energy_spent.get(tid, 0)

    def remaining(self, tid) -> float:
        n = self.nodes[tid]
        return max(0.0, self.own_cost(n) + n.sunk - self._spent(tid)) + self._phantom(n)

    def committed(self) -> float:
        spent = self.colony.starting_budget - self.colony.budget_remaining
        total = spent
        for tid, t in self.task_graph.tasks.items():
            if t.status in (0, 1) and tid in self.nodes:
                total += self.remaining(tid)
        for queue in self.pending_overflow.values():
            for kw in queue:
                total += self.r_new("dec" if kw.get("role") == "decomposer" else "exec")
        return total

    def slack(self) -> float:
        return (self.colony.starting_budget - self.floor) - self.committed()

    def _admissible(self, extra=0.0) -> bool:
        c = self.committed() + extra
        self.max_committed = max(self.max_committed, c)
        return c <= self.colony.starting_budget - self.floor

    # ------------------------------------------------------------ spawns
    def spawn_agent(self, role, task_id, parent_id=None, ghost_context=None):
        kind = ("root" if task_id == self.root_task_id
                else "dec" if role == "decomposer" else "exec")
        n = self.nodes.get(task_id)
        if n is None:
            self.nodes[task_id] = NodeRes(kind=kind)
            if task_id == self.root_task_id and self.admission_on and not self._admissible():
                self.counters["root_refused"] += 1
                del self.nodes[task_id]
                return None
            # A new child was reserved when its parent's SPAWN was admitted.
        elif self._converting == task_id:
            self._converting = None
        else:
            saved = (n.kind, n.k, n.spawned, n.sunk)
            if kind != "exec":
                n.spawned, n.k = False, 0   # a respawned decomposer plans again
            if self.respawn_mode == "reserve":
                n.sunk = self._spent(task_id)
            if self.admission_on and not self._admissible():
                n.kind, n.k, n.spawned, n.sunk = saved
                self.counters["respawn_refused"] += 1
                return None
            self.counters["respawns"] += 1
        return super().spawn_agent(role, task_id, parent_id, ghost_context)

    def _handle_spawn_request(self, event):
        spawner = self.colony.get_agent(event.from_agent)
        subs = event.payload.get("subtasks")
        if (not self.admission_on or spawner is None or spawner.role != "decomposer"
                or spawner.task_id not in self.nodes):
            return super()._handle_spawn_request(event)
        if not isinstance(subs, list) or not subs:
            subs = [{"role": event.payload.get("role"), "task": event.payload.get("task_id")}]
        n = self.nodes[spawner.task_id]
        child = "dec" if n.kind == "root" else "exec"
        saved = (n.k, n.spawned)
        chosen = None
        for k2 in range(len(subs), 0, -1):
            n.k, n.spawned = k2, True
            if self._admissible(k2 * self.r_new(child)):
                chosen = k2
                break
        if chosen is None:
            n.k, n.spawned = saved
            self.counters["spawn_refused"] += 1
            self.messenger.push_event("failure_request", spawner.agent_id, {
                "task_id": spawner.task_id, "role": spawner.role,
                "parent_id": spawner.parent_id,
                "result": "No energy budget left to decompose this task.",
            })
            return False
        if chosen < len(subs) and self.merge == "truncate":
            self.counters["truncations"] += 1
            event.payload["subtasks"] = subs[:chosen]
        elif chosen < len(subs):
            self.counters["merges"] += 1
            keep = [s for s in subs[:chosen - 1]]
            rest = [s for s in subs[chosen - 1:] if isinstance(s, dict)]
            merged = dict(rest[0])
            merged["task"] = MERGE_JOIN.join(str(s.get("task", "")) for s in rest)
            merged["dependencies"] = []
            event.payload["subtasks"] = keep + [merged]
        ok = super()._handle_spawn_request(event)
        if not ok:
            n.k, n.spawned = saved
        return ok

    def handle_failure(self, event):
        p = event.payload
        tid = p.get("task_id")
        too_large = (p.get("role") == "executor" and p.get("judge_verdict") is None
                     and str(p.get("result", "")).startswith("TASK TOO LARGE:"))
        if too_large and tid in self.nodes and tid not in self.abandoned_tasks:
            n = self.nodes[tid]
            saved = (n.kind, n.k, n.spawned, n.sunk)
            n.kind, n.k, n.spawned = "dec", 0, False
            if self.sunk_mode:
                n.sunk = self._spent(tid)
                n.conv_sunk = n.sunk
            if self.admission_on and not self._admissible():
                n.kind, n.k, n.spawned, n.sunk = saved
                n.conv_sunk = 0
                self.counters["conversion_refused"] += 1
                agent_id = event.from_agent
                self.conversion_sunk = getattr(self, "conversion_sunk", {})
                self._retire_agent(agent_id, tid)
                self._abandon_task(tid, p.get("parent_id"), agent_id,
                                   self.respawn_counts.get(tid, 0),
                                   reason="no energy budget to decompose it further",
                                   reason_label="admission: conversion refused")
                return
            self.counters["conversions"] += 1
            self._converting = tid
        return super().handle_failure(event)

    # ----------------------------------------------------------- ceiling
    def _task_energy_overrun(self, task_id):
        if not self.ceiling_on:
            return super()._task_energy_overrun(task_id)
        if not task_id or task_id not in self.nodes:
            return None
        n = self.nodes[task_id]
        own = self.own_cost(n)
        if self.scope == "agent":
            t = self.task_graph.tasks.get(task_id)
            a = self.colony.get_agent(t.agent_id) if t is not None and t.agent_id else None
            spent = a.energy_spent if a is not None else 0
            cap = self.p["m"] * own
        else:
            spent = self._spent(task_id)
            cap = self.p["m"] * own + (n.sunk if self.sunk_mode or self.respawn_mode == "reserve" else 0)
        if spent >= cap:
            return (spent, int(cap))
        if self.task_cap is not None:
            total = self._spent(task_id)
            tcap = n.conv_sunk + self.task_cap * own
            if total >= tcap:
                self.counters["task_cap_stop"] += 1
                return (total, int(tcap))
        if (self.dynamic and spent > own + n.sunk and self.slack() < 0
                and (self.dynamic_root or task_id != self.root_task_id)):
            self.counters["dynamic_stop"] += 1
            self.dynamic_stopped = getattr(self, "dynamic_stopped", set())
            self.dynamic_stopped.add(task_id)
            return (spent, int(own + n.sunk))
        return None

    def ceiling_for(self, task_id) -> float:
        n = self.nodes[task_id]
        return self.p["m"] * self.own_cost(n) + (n.sunk if self.sunk_mode else 0)


# ---------------------------------------------------------------------------
# The corrected formula that survived the sweeps (see REPORT.md). Parameters
# are p75 of a successful attempt, fitted by fit_quantiles.py for the two
# cost models; "full" = generations near their caps, "lean" = the lengths the
# source comments record from real runs.
RECOMMENDED_PARAMS = {
    "full": dict(e=27.0, alpha=20.5, beta=6.0),
    "lean": dict(e=17.0, alpha=14.1, beta=5.1),
}


def recommended(model="full", **over):
    p = RECOMMENDED_PARAMS[model]
    kw = dict(p,
              sigma=round(p["alpha"] / 2, 2),  # root roll-up retry reserve, not a cost
              floor=6,                         # tick() declares death at <= 5
              provisional="min",               # unspawned decomposer: alpha + beta + e
              sunk=True,                       # conversion carries executor spend
              respawn="reserve",               # every attempt reserves its own cost
              merge="truncate",                # lower k by dropping, never bundling
              task_cap=3.0,                    # C = conv_sunk + 3 * own_cost, per task
              m=1e9,                           # per-attempt m*own: redundant, off
              dynamic=False)                   # slack stop: no measurable effect, off
    kw.update(over)
    return kw
