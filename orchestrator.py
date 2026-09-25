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

import ast
import difflib
import uuid
import time
import re
import statistics
import traceback

import numpy as np
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any, List


# # Assuming these are your custom modules
from colony_state import ColonyState, AgentNode
from task_graph import TaskGraph, TaskNode
from event_queue import Messenger, Event
from agent_node import (
    Agent,
    _dedupe_repeated_sentences,
    EXEMPLAR_SUBTASK_DESCRIPTIONS,
    EXEMPLAR_REPORT_TEXTS,
    EMPTY_DECISION_PLACEHOLDER,
    PROMPT_EXEMPLARS,
    role_may_use_tools,
)
from tools import ToolRegistry
from text_utils import (
    normalize_identifier,
    normalize_words,
    trim_to_sentences,
    first_clause,
    degeneracy_cut,
    drop_incomplete_tail,
    scrub_result,
    cut_adjacent_repeat,
    trim_degenerate_tails,
    trim_artifact_tail,
    is_exemplar_echo,
    cut_prompt_header_echo,
    asks_for_software,
    asks_for_artifact,
    FIGURE_DIVERGENCE_MIN,
    FIGURE_DIVERGENCE_MAX_OVERLAP,
    artifact_request_words,
    asserted_figures,
    asserted_identifiers,
    figure_divergence,
    figure_fidelity,
    strip_task_ids,
    supplied_data,
    plain_register,
    software_artifact_reason,
    software_request_words,
)
# For DEGENERATE_ANSWER_MESSAGE only -- the synthesizer instance is injected,
# not constructed here. Imported so a run that produced nothing usable says so
# in the same words whichever guard caught it.
from synthesizer import Synthesizer
import ghost_extractor

# Software-framing interventions, all on by default (see Orchestrator.__init__).
FRAMING_LEVERS = ("reword", "block")

# memory_state.OUTCOME_ACCEPTED -- repeated rather than imported because
# memory_state pulls in faiss/sentence_transformers at import time.
CACHE_OUTCOME_ACCEPTED = "accepted"

# PATCH 33: memory_state.SUCCESS_CACHE_THRESHOLD, repeated for the same
# reason. Ghost read-back keeps a hit at/above it (tuned for MiniLM).
GHOST_LESSON_THRESHOLD = 0.45
GHOST_LESSON_TOP_K = 3


# --- PATCH 12: the goal-drift gate ------------------------------------------
# PATCH 7 measured; this acts, at the extreme only.
#
# RELATIVE, not absolute. Run 5's whole distribution sat under 0.702 -- that
# was the MAXIMUM, on "Run the voting method over the available books", which
# is almost a restatement of the goal. A fixed cutoff written against that
# corpus is a cutoff against MiniLM's ceiling for this one phrasing, and the
# next problem's ceiling will sit somewhere else entirely. So the threshold is
# a fraction of the highest goal-cosine the run has actually seen.
#
# WHERE THE FRACTION COMES FROM. Run 5's 18 goal-cosines, sorted:
#   0.341 0.365 0.369 | 0.439 0.448 0.452 0.503 0.523 0.628 0.639 0.653
#   0.666 0.666 0.666 0.667 0.702 0.702 0.702
# The largest gap in the low tail is 0.369 -> 0.439 (0.070); every other
# neighbouring gap below 0.5 is under 0.015. That gap is the cluster boundary:
# below it are the three deliverable-manufacturing subtasks (slide deck,
# stakeholder letter, formatted summary) that nobody asked for; above it is
# task_91dcc48e at 0.448, which completed and contributed. Threshold placed in
# the middle of the gap, 0.404, which is 0.575 of 0.702. 0.58 rounds it with
# the margin still on both sides: 0.58 * 0.702 = 0.407, which is 0.038 clear
# of the cluster's top and 0.032 clear of the nearest keeper.
GOAL_DRIFT_GATE_FRACTION = 0.58

# No gate until the run has this many measured goal-cosines, the current
# batch's own included. The threshold is a fraction of the observed MAX, and
# a max taken from two samples is not an observation -- one low-scoring first
# batch would set a ceiling low enough to gate nothing for the rest of the
# run, or a first batch of genuinely peripheral tasks would set one high
# enough to gate its own siblings. Run 5 reached 6 samples inside the first
# three spawns and the low cluster did not appear until the twelfth, so this
# costs nothing there.
GOAL_DRIFT_GATE_MIN_SAMPLES = 6

# MEASURE-ONLY AS OF RUN 7. Scoring, the ledger and the [GATED] marking all
# stay on; the DROP does not happen.
#
# Run 6 is the reason. Its goal_vector was built from a 246-character string
# whose last 116 characters were "---THESE THREE QUESTIONS NEED TO BE
# ANSWERED IN ORDER--END OF OUTPUT-- -- -- --" (fixed upstream now, in
# _clean_goal_text). Five subtasks were dropped, four of them conflict
# resolution -- which is the goal's OWN third clause -- while task_d4aace3c,
# "Design conflict-resolution protocols for unresolved preferences", scored
# 0.242, LOWER than three of the five, escaped the gate only because it
# spawned before the run reached GOAL_DRIFT_GATE_MIN_SAMPLES, and then
# COMPLETED and was promoted. That is a direct counterexample: the gate's
# own ordering put a task that produced accepted work below tasks it refused
# to start.
#
# Replaying the 19 scores against the CLEANED goal does not rescue them --
# it makes the picture worse (4 of the 5 stay under threshold, and 3 more
# join them, task_d4aace3c included). The clause-level numbers say why: the
# goal is a three-clause conjunction, its embedding sits in the average of
# the three, and a subtask that nails exactly ONE clause is far from that
# average by construction. task_24766b16 scores 0.202 against the whole goal
# and 0.259 against the clause it actually serves; task_3d7a8019 scores
# 0.296 and 0.466. The run's max, 0.658, belongs to a subtask that straddles
# two clauses. So the threshold is a multi-clause ceiling applied to
# single-clause children, and goal sanitation alone does not fix it.
#
# Re-arm (with a per-clause reference, most likely) only after a run shows a
# clean goal string AND a distribution where the gate's ordering and the
# run's own outcomes agree.
GOAL_DRIFT_GATE_ACTS = False


def _cosine(a, b) -> Optional[float]:
    """
    PATCH 7. Cosine between two embeddings, or None when the answer would be
    meaningless.

    None -- not 0.0 -- on a missing or zero-norm vector. Problem_Phaser falls
    back to np.zeros(embed_dim) for goal_vector on a degenerate parse
    (problem_phaser.py:601) and on an exception (:636), and a zero vector has
    no direction: scoring against it would report every subtask as maximally
    drifted and bury the real distribution in fabricated zeros. The caller
    counts these separately instead.

    Mirrors judge.semantic_check's arithmetic deliberately, so a drift number
    in the spawn log and a similarity number in a tier-2 verdict are the same
    measurement and can be compared directly.
    """
    if a is None or b is None:
        return None
    try:
        x = np.asarray(a, dtype="float32").ravel()
        y = np.asarray(b, dtype="float32").ravel()
        if x.size == 0 or y.size == 0 or x.shape != y.shape:
            return None
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx == 0.0 or ny == 0.0:
            return None
        return float(np.dot(x, y) / (nx * ny))
    except Exception:
        return None


def _without_quoted_example(reason: str) -> str:
    """software_artifact_reason minus its quoted evidence: "names a
    source-code file ('scorer.py')" -> "names a source-code file". Anything
    written to an agent's fail_reason becomes its respawn's ghost context,
    and quoting the file or function name back would hand the replacement
    the exact framing the rejection exists to stop."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", reason)


@dataclass
class TaskReservation:
    """What the admission rule and the per-task ceiling know about one task.
    See the RESERVE_* constants in Orchestrator.__init__."""
    kind: str                # "executor" | "decomposer" | "root"
    k: int = 0               # children in its admitted SPAWN batch (queued overflow included)
    spawned: bool = False    # False until a SPAWN batch is admitted; k is provisional until then
    sunk: int = 0            # task spend before its current attempt began
    conv_sunk: int = 0       # spend carried in by a TASK TOO LARGE conversion
    attempts: int = 0        # agents spawned for this task so far
    k_peak: int = 0          # largest k any earlier attempt had admitted (ceiling only)


class Orchestrator:
    def __init__(self, colony_state: ColonyState, task_graph: TaskGraph, messenger: Messenger,
                 phaser=None, judge=None, memory_store=None, synthesizer=None,
                 model=None, tokeniser=None, embed_model=None,
                 budget_override: Optional[int] = None,
                 framing_levers=FRAMING_LEVERS,
                 energy_trace_path: Optional[str] = None):
        self.colony = colony_state
        # Which software-framing interventions act this run: "reword"
        # (plain wording for code-coded subtask vocabulary) and/or "block"
        # (reject code-shaped SPAWN/REPORT, scrub code-shaped DIE text).
        # Detection and its counters run regardless, so a run with
        # framing_levers=() is the measured baseline the others compare to.
        unknown = set(framing_levers) - set(FRAMING_LEVERS)
        if unknown:
            raise ValueError(f"unknown framing_levers {sorted(unknown)}; "
                             f"valid: {sorted(FRAMING_LEVERS)}")
        self.framing_levers = frozenset(framing_levers)
        self.task_graph = task_graph
        self.messenger = messenger
        self.agent_counter = 0 
        self.root_task_id: Optional[str] = None
        self.running = True
        self.run_trace: Optional[Dict[str, Any]] = None
        # PATCH 7. One record per spawned subtask:
        # {task_id, description, goal_drift, parent_drift}. Measure-only this
        # round -- nothing reads it except the GOAL DRIFT AT SPAWN ledger
        # section, which exists to produce the distribution a threshold can
        # later be picked from. Appended even when a drift is None, so the
        # "not measurable" cases are visible rather than silently absent.
        self.goal_drift_samples: List[Dict[str, Any]] = []

        # PATCH 12. Description embeddings computed by the goal-drift gate at
        # batch-screen time, keyed by the task_id the batch already allocated,
        # and consumed (popped) by _spawn_child_task so a survivor is embedded
        # once per run rather than twice. A dropped subtask's entry is popped
        # by the gate itself -- nothing else ever looks it up.
        self._gate_description_embeddings: Dict[str, Any] = {}

        # PATCH 12, measure-only mode. task_ids the gate scored below
        # threshold while GOAL_DRIFT_GATE_ACTS is False. Read once, by
        # _measure_spawn_drift, to mark the ledger entry the spawn itself
        # files -- see the note in _screen_goal_drift_batch.
        self._gate_would_drop: set = set()

        # PATCH 22 (measure-only). Spawned subtasks whose description states
        # a figure the user's request does not, as [(task_id, [figures],
        # description)]. Nothing reads it but the ledger.
        self.unsupplied_figure_subtasks: List[Any] = []
        self.spawned_subtask_count = 0
        # PATCH 29. Agents spawned, and how many carried the request in
        # their prompt.
        self.request_prompt_counts = {"agents": 0, "with_request": 0}

        # PATCH 17 (measure-only). One vector per goal clause, from the
        # phaser's spec. Scored alongside goal_drift, never instead of it:
        # nothing reads clause_drift except the ledger, so the two
        # distributions can be compared on the same run.
        self.goal_clause_texts: List[str] = []
        self.goal_clause_embeddings: List[Any] = []

        # PATCH 16. The last REPORT each task produced, as
        # {task_id: (agent_id, untrimmed result)}. One entry per task,
        # overwritten at every REPORT, so the comparison is always against
        # the immediately preceding attempt at the SAME task -- a warn
        # retry by the same agent as much as a respawn by a new one.
        self._task_last_report: Dict[str, Any] = {}

        # Injected shared components -- NONE of these were wired in previously.
        # phaser: Problem_Phaser instance (turns raw text into a spec + budget)
        # judge: Judge instance (tiered verification before promotion)
        # memory_store: MemoryStore instance (ghost + success indices)
        # synthesizer: Synthesizer instance (final English decode)
        # model/tokeniser: shared LLM, needed to actually construct live Agent
        #   objects -- previously the orchestrator only ever touched AgentNode
        #   (data), never the Agent (reasoning) class, so nothing ever ticked.
        # embed_model: shared SentenceTransformer, used to embed agent outputs
        #   before handing them to judge.decide() for semantic_check.
        self.phaser = phaser
        self.judge = judge
        self.memory_store = memory_store
        self.synthesizer = synthesizer
        self.model = model
        self.tokeniser = tokeniser
        self.embed_model = embed_model

        # Populated by initialize_colony() once Problem_Phaser has run.
        self.spec: Optional[Dict[str, Any]] = None

        # task_id -> number of times _kill_and_respawn has respawned an
        # agent for that task. A2: this is now enforcement, not just
        # observation -- see MAX_TASK_ATTEMPTS below.
        self.respawn_counts: Dict[str, int] = {}

        # task_id -> the last REPORT text seen for that task, whatever the
        # judge made of it. The salvage value when a task is abandoned:
        # eleven agents burned on one task_id still produced *something*,
        # and handing the parent that plus an explicit abandonment marker
        # beats handing it nothing.
        self.last_partial_result: Dict[str, str] = {}

        # Tasks abandoned by the attempt cap, so terminate() can say so.
        self.abandoned_tasks: set = set()
        # task_id -> why it was abandoned, for terminate() and the partial
        # result's cause line: the attempt cap, the energy ceiling, or no
        # energy to respawn.
        self.abandon_reasons: Dict[str, str] = {}
        # task_id -> (spent, ceiling) for a task whose REPORT was promoted
        # after its agents had already crossed the energy ceiling. The
        # result is kept -- it was produced and judged good -- but the
        # overrun is recorded so the ledger stays honest about it.
        self.completed_over_ceiling: Dict[str, tuple] = {}

        # Child tasks that never got an agent because spawn_agent had no
        # energy for them -- see _close_unstartable_child.
        self.unstarted_tasks: set = set()

        # agent_id -> the exact REPORT text that last earned it a judge WARN.
        # Used to short-circuit an agent that answers a WARN by resubmitting
        # the identical string (see handle_completion's warn branch).
        self.last_warned_report: Dict[str, str] = {}

        # High-water mark of len(self.live_agents), sampled once per tick.
        self.peak_live_agents = 0

        # Ticks elapsed, and how often the energy ledger is printed mid-run.
        # A run interrupted before terminate() used to leave nothing but
        # thousands of heartbeat lines to read; one ledger every
        # LEDGER_EVERY_TICKS puts the whole budget picture on a single screen
        # while the run is still going.
        self.tick_count = 0
        self.LEDGER_EVERY_TICKS = 20

        # Hard ceiling on tick count. With a large budget_override a colony
        # that never converges can run until the session dies, which reads
        # identically to "still working" in the log. Stopping at a fixed tick
        # count and saying so distinguishes "ran out of time" from "ran out
        # of energy" at a glance.
        self.MAX_TICKS = 800
        self.hit_tick_ceiling = False

        # Diagnostic lever: when set, replaces the budget the Problem_Phaser
        # derives for the run (see initialize_colony). Non-standard by
        # design -- it exists to answer "is energy the binding constraint?"
        # without the phaser's estimate in the way.
        self.budget_override = budget_override

        # Where terminate() appends this run's per-task energy records (one
        # JSON line per run), the data RESERVE_* is re-fitted from. None = off.
        self.energy_trace_path = energy_trace_path

        # Live Agent objects (the think/decide/execute reasoning wrapper),
        # keyed by agent_id. Distinct from ColonyState.agents, which only
        # holds the AgentNode data records. Previously nothing populated
        # this at all -- the colony could spawn task nodes and agent
        # records forever with no reasoning loop ever actually running.
        self.live_agents: Dict[str, Agent] = {}

        # Hyperparameters
        self.energy_when_new_by_role = {
            "executor": 2,
            "verifier": 2,
            "decomposer": 4,
        }
        self.energy_when_new = 4  # fallback default for an unrecognized role
        self.energy_threshold_stress = 10
        self.energy_threshold_death = 5
        self.timeout_threshold = 30.0  # Seconds an agent can remain silent before being killed/probed

        # A crashing agent.run() is debited the same flat cost as spawning a
        # fresh agent of its role (energy_when_new_by_role) -- crashing isn't
        # free just because no tokens were produced. After this many
        # *consecutive* crashes (reset on any successful tick), the agent is
        # routed through the normal kill/respawn path (failure_request)
        # instead of being retried forever on the same broken state.
        self.MAX_CONSECUTIVE_CRASHES = 3

        # A2: how many times one task_id may be respawned before it is
        # abandoned outright. Without this, a task that no agent can satisfy
        # recycles until the colony hits energy death with nothing completed
        # (observed: one task_id consuming eleven agents in a single run).
        # Abandoning it converts that into a bounded failure that still
        # returns a partial answer -- the task is marked failed, its best
        # partial result is pushed to the parent with an explicit marker,
        # and its dependents are released so the rest of the graph can
        # finish.
        self.MAX_TASK_ATTEMPTS = 3

        # Whole-batch rejections by the derived-subtask guard (a copied
        # exemplar, or software framing on a non-software project) respawn
        # the decomposer without any agent having executed anything, so they
        # do not spend MAX_TASK_ATTEMPTS: task_97d0d827 was abandoned on
        # guard trips alone. They get their own, separate cap instead; past
        # it a rejection counts as an ordinary attempt again, and the
        # per-task energy ceiling bounds them throughout.
        self.MAX_DERIVED_REJECTIONS = 3
        self.derived_rejection_counts: Dict[str, int] = {}

        # Energy reservation, admission and the per-task ceiling. Evaluated
        # and calibrated against the real tick loop in
        # sims/reservation_formula (REPORT.md there has the numbers).
        #
        # What one ATTEMPT of a task costs its own task (spawn + think ticks
        # + child-result injections + tier-3 critiques; never its children's
        # spend), at p75 of a successful attempt:
        #   executor        e             = RESERVE_EXECUTOR
        #   decomposer(k)   alpha + beta*k   (linear in its batch size k:
        #                   each child adds its SPAWN-payload tokens and one
        #                   result injection)
        #   root(k)         decomposer(k) + sigma. The root is billed exactly
        #                   like any decomposer; sigma is a reserve for its
        #                   roll-up being sent back by review, whose failure
        #                   ends the run.
        # These are measured quantities -- re-fit them from a real run's
        # task_energy_spent ledger (sims/reservation_formula/fit_quantiles.py)
        # when the token caps or the cycle cap change.
        self.RESERVE_EXECUTOR = 27.0
        self.RESERVE_DECOMPOSER_BASE = 20.5
        self.RESERVE_PER_CHILD = 6.0
        self.RESERVE_ROOT_EXTRA = self.RESERVE_DECOMPOSER_BASE / 2

        # Admission: new work (bootstrap, a SPAWN batch, a TASK TOO LARGE
        # conversion, a respawn) is allowed only while
        #     spent + sum of every unfinished task's remaining reservation
        #           + the new work's reservation  <=  budget - ADMISSION_FLOOR
        # A decomposer that has not SPAWNed yet reserves its cheapest
        # decomposition (k=1, one child) -- reserving nothing for its children
        # let the top of the tree admit work the bottom could never afford,
        # and reserving its full fan-out made the root unstartable below ~230.
        # A batch that does not fit is cut to the largest k that does; the
        # rest is DROPPED and the decomposer told so. Bundling the overflow
        # into one child does not save energy: that child DIEs TASK TOO LARGE
        # and needs the budget admission just said was not there.
        # The floor keeps the colony out of energy death (tick() stops at
        # <= energy_threshold_death with work in flight).
        self.ADMISSION_CONTROL = True
        self.ADMISSION_FLOOR = self.energy_threshold_death + 1
        self.task_reservations: Dict[str, TaskReservation] = {}
        # Tasks whose next spawn is a TASK TOO LARGE conversion already
        # admitted in handle_failure, so spawn_agent must not re-check it.
        self._admitted_conversions: set = set()

        # Per-task energy ceiling: the runaway guard across ALL of a task's
        # agents. MAX_TASK_ATTEMPTS counts agents; this counts what they cost.
        #     ceiling = conv_sunk + TASK_CEILING_ATTEMPTS * own_cost
        # i.e. one p75 attempt of what THIS task should cost for every agent
        # the attempt cap allows it. A decomposer's attempt grows with k, the
        # root's carries sigma, and a converted task carries the executor
        # spend it arrived with (conv_sunk) instead of having it eat its
        # decomposer share. A per-ATTEMPT ceiling was tested and dropped: the
        # cycle cap (Agent.MAX_NON_TERMINAL_CYCLES) already stops one agent
        # first.
        #
        # TASK_CEILING_ATTEMPTS is MAX_TASK_ATTEMPTS + 1 -- the agents the
        # attempt cap grants -- not a separate 3. At 3 the ceiling priced
        # three attempts while the cap allowed four, so any task that used its
        # fourth ran into the ceiling DURING it, even with every attempt under
        # p75: task_4ae60b94 (executor, three tier-3 rejects at 4-5 energy
        # each, 82 against 81) and task_4c62ea1f (decomposer, 85) were both
        # four attempts of ~21. The ceiling was silently lowering the attempt
        # cap. Tied, the cap decides how many tries a task gets and the
        # ceiling only fires when those tries average over p75: executor 108.
        #
        # A decomposer's ceiling is priced at its fan-out cap (or its largest
        # admitted batch, if bigger), not at the k=1 admission uses. At k=1 a
        # decomposer stuck re-planning -- every attempt paying for THINK ticks
        # and a SPAWN payload that fails validation -- was priced BELOW an
        # executor. Priced at the cap it is 130 (root 195), and it does not
        # drop back when a respawn re-plans from k=0. This is the role
        # multiplier, at whichever role the task holds now: a TASK TOO LARGE
        # conversion re-prices it as a decomposer from that point.
        #
        # No depth multiplier: each task is billed only its own spend, so
        # depth adds nothing to a task's cost. Measured over every scenario in
        # sims/reservation_formula (real tier-3 cost), the highest spend/
        # ceiling is 0.37 at the root and 0.20 three levels down -- headroom
        # grows with depth, the opposite of what a depth multiplier corrects.
        #
        # Checked on every cycle (_run_live_agents) as well as on the respawn
        # decision, because a task can cross the line mid-agent with no
        # respawn boundary in between. Overshoot is therefore bounded by one
        # cycle, except on the cycle that produced a REPORT/DIE or handed off
        # a SPAWN/TOOL: that result is adjudicated first rather than
        # discarded, so the agent gets one more cycle before it can be stopped.
        self.TASK_CEILING_ATTEMPTS = self.MAX_TASK_ATTEMPTS + 1

        self.SHORT_ANSWER_WORD_THRESHOLD = 12

        self.MAX_SUBTASKS_ROOT = 3
        self.MAX_SUBTASKS_NON_ROOT = 2

        # A decomposer's SPAWN batch over the fan-out cap used to get its
        # overflow crammed into one bundled child -- which, being several
        # unrelated items forced into one agent, reliably DIEd as "TASK TOO
        # LARGE" and got re-split anyway (a wasted respawn cycle, twice per
        # run at cap=2/3). Now the overflow beyond `cap` is parked here
        # (keyed by the decomposer's agent_id, already fully resolved --
        # see handle_spawn) and drained one item at a time in
        # _drain_pending_overflow, called whenever one of that decomposer's
        # children genuinely completes and frees a slot. Keeps the same
        # "at most `cap` concurrently in-flight children" invariant the cap
        # exists for, without discarding or force-merging anything.
        self.pending_overflow: Dict[str, list] = {}

        self.energy_map = {
            0: "fine",
            1: "stressed",
            2: "death"
        }

    def _generate_id(self) -> str:
        """Generates a unique 8-character ID for new agents."""
        return f"agent_{uuid.uuid4().hex[:8]}"

    def _generate_task_id(self) -> str:
        """Generates a unique ID for new task graph nodes."""
        return f"task_{uuid.uuid4().hex[:8]}"

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

            # An agent's REPORT/DIE is queued on the tick it is produced and
            # adjudicated on the next one, so its task can be abandoned in
            # between -- the per-cycle energy ceiling closes tasks without
            # waiting for an event to route. Letting the stale event through
            # would re-open a settled task: handle_completion would find no
            # agent_node, skip the judge, and promote the result over the
            # abandonment marker its parent was already handed.
            if e.type in ("completion_request", "failure_request"):
                stale_task = (e.payload or {}).get("task_id")
                if stale_task and stale_task in self.abandoned_tasks:
                    print(f"  [stale-event] dropping {e.type} from {e.from_agent} "
                          f"for abandoned task {stale_task}.")
                    continue

            if e.type == "spawn_request":
                self.handle_spawn(e)
            elif e.type == "completion_request":
                self.handle_completion(e)
            elif e.type == "failure_request":
                self.handle_failure(e)
            elif e.type == "tool_request":     
                self.handle_tool_request(e)
            elif e.type == "parent_notification":
                self.handle_parent_notification(e)
            # "tool_response" is an orchestrator -> nowhere-in-particular
            # broadcast event; nothing needs to route it further since
            # handle_tool_request already delivers the result straight to the
            # live Agent via receive_tool_result().

    def _process_unblocked_tasks(self):
        """Finds tasks whose dependencies are met and spawns agents for them."""
        ready_tasks = self.task_graph.get_ready_tasks()
        for task in ready_tasks:
            # Spawn a new worker if the ready task doesn't have an agent assigned yet
            if getattr(task, 'agent_id', None) is None:
                self.spawn_agent(role=task.required_role, task_id=task.task_id)

    def _run_live_agents(self):
        """
        The missing heartbeat: drives Agent.think()/decide()/execute() for
        every agent whose task is currently running.
        """
        available_roles = ["decomposer", "executor", "verifier"]
        # A8 STEP 1 (temporary experiment -- REVERT TO ToolRegistry.list_tools()
        # AFTERWARDS): tools disabled entirely to isolate whether TOOL is
        # part of what's blocking subtask completion. With this and tier 3
        # (see judge.decide() call below) both off, the only paths an agent
        # has left are THINK -> REPORT -> tier 1/2 -- the maximum-isolation
        # run. request_tool() now enforces this (not just the prompt text),
        # so a TOOL action can't sneak through even if the model emits one.
        available_tools = []

        for agent_id, live_agent in list(self.live_agents.items()):
            task_id = live_agent.task_id
            task_node = self.task_graph.tasks.get(task_id)
            if task_node is None or task_node.status != 1:
                # Only tick agents actively bound to a running task. This
                # also naturally covers a task waiting on unmet dependencies
                # (status stays 0 until TaskGraph.assign_agent lets it become
                # 1 -- see the fix there) -- such an agent already exists in
                # live_agents (spawn_agent creates it eagerly) but is simply
                # never ticked until its task is actually ready to run.
                continue

            if live_agent.awaiting is not None:
                continue

            # Belt and braces for handle_parent_notification's gate: whatever
            # cleared `awaiting`, a decomposer is not ticked while any child
            # is unfinished -- it cannot REPORT a complete result, and each
            # tick spends one of its capped cycles. Re-set rather than just
            # skipped, so the deadlock watchdog keeps treating it as waiting
            # instead of as silent.
            if getattr(live_agent, "role", None) == "decomposer" and self._open_child_task_ids(agent_id):
                live_agent.awaiting = "children"
                continue

            agent_node = self.colony.get_agent(agent_id)
            prev_len = len(live_agent.thought_process)
            was_capped = getattr(live_agent, "cycles_capped", False)
            # SPAWN closed for lack of energy runs the same reduced-menu
            # decide() as the cap, and its overrides land in
            # cycle_cap_coerced -- so it has to count as a decision too, or
            # "forced" can exceed "decisions" in the ledger.
            was_final = getattr(live_agent, "final_only", was_capped)
            try:
                action = live_agent.run(available_roles, available_tools, requirements=None)
                # Tallied so the final report shows whether the cycle cap
                # is firing, and how often the model ignored its menu.
                if getattr(live_agent, "cycles_capped", False) and not was_capped:
                    self.colony.record_verdict("cycle_cap_reached")
                # A run() that STARTED capped is a decision made with the
                # reduced menu -- run() skips think() and goes straight to a
                # decide() whose prompt no longer offers THINK/SPAWN. This is
                # the counter that shows enforcement working: cycle_cap_coerced
                # below counts only the decisions where the model ignored the
                # reduced menu anyway, so it reads 0 exactly when the strip
                # does its job.
                if was_final:
                    self.colony.record_verdict("cycle_cap_decisions")
                if getattr(live_agent, "cap_coerced_last_run", False):
                    self.colony.record_verdict("cycle_cap_coerced")
                # Observation only: pseudocode in an agent's own reasoning
                # is the reflex showing up before any action, and nothing
                # else would count it.
                new_thoughts = str(getattr(live_agent, "thought_process", "") or "")[prev_len:]
                if self._software_framing_reason(new_thoughts) is not None:
                    self.colony.record_verdict("software_framing_think_detected")
            except Exception as e:
                print(f"Agent {agent_id} crashed during run(): {e}")
                print(traceback.format_exc())

                fail_reason = f"CRASH in run(): {e}"
                give_up = True

                if agent_node:
                    # A crash still counts as activity -- without this the
                    # deadlock watchdog sees a stale last_active and piles a
                    # second, conflicting kill/respawn on top of this one.
                    agent_node.last_active = time.time()

                    crash_cost = self.energy_when_new_by_role.get(live_agent.role, self.energy_when_new)
                    self.colony.debit_energy(agent_id, crash_cost, category="crash")

                    agent_node.fail_reason = fail_reason
                    agent_node.crash_count += 1
                    give_up = agent_node.crash_count >= self.MAX_CONSECUTIVE_CRASHES
                    crash_label = f"{agent_node.crash_count} times in a row"
                else:
                    # No colony node: there's nowhere to keep a crash
                    # count (the debit itself is no longer free -- it would
                    # book under "orphaned" -- but an uncounted crash-loop
                    # would still retry forever). Route it out on the first
                    # crash.
                    crash_label = "with no registered colony node"

                if give_up:
                    print(f"Agent {agent_id} crashed {crash_label} -- "
                          f"routing through kill/respawn instead of retrying every tick.")
                    self.messenger.push_event(
                        "failure_request",
                        agent_id,
                        {
                            "task_id": task_id,
                            "role": live_agent.role,
                            "parent_id": live_agent.parent_id,
                            "result": fail_reason,
                        }
                    )
                # A crash is billed like a spawn, and a crash-loop under
                # MAX_CONSECUTIVE_CRASHES reaches no respawn decision at
                # all -- so it needs the same per-cycle boundary the
                # normal path gets below.
                self._enforce_task_energy_ceiling(agent_id, task_id)
                continue

            if agent_node:
                agent_node.last_active = time.time()
                agent_node.crash_count = 0

            new_chars = len(live_agent.thought_process) - prev_len
            cost = max(1, new_chars // 100)
            self.colony.debit_energy(agent_id, cost, category="think_tick")

            # The ceiling, checked every cycle rather than only where
            # MAX_TASK_ATTEMPTS is. This covers the cycles that reach no
            # terminal action -- a THINK, a refused TOOL/SPAWN -- and so
            # produce no respawn decision for the old check to fire on.
            #
            # Skipped for the cycle that just produced a REPORT or DIE, or
            # that handed off a SPAWN/TOOL: that work is queued for the next
            # tick's _route_events and has not been judged yet, so killing
            # the agent here would spend the energy and then throw the result
            # away -- the opposite of what the ceiling is for. It is not a
            # hole, because every way that adjudication can end also checks:
            # promote completes the task, execute/DIE and the two warn
            # short-circuits route through _kill_and_respawn, and a plain
            # WARN -- the loop that actually overspends, since every pass
            # through it is a REPORT cycle -- checks in handle_completion
            # before sending the agent back. That is where the REPORT is
            # salvaged rather than discarded: by then it has been rejected
            # and recorded in last_partial_result.
            # The same predicate Agent.run uses to decide whether the cycle
            # counted as non-terminal, so the rule here is exactly "a cycle
            # that was charged is a cycle that is checked". A dispatched TOOL
            # sets no `awaiting` but is still work in flight -- its result
            # arrives before the next tick -- so retiring the agent on it
            # would strand the call.
            handed_off = (getattr(live_agent, "awaiting", None) is not None
                          or (action == "TOOL"
                              and getattr(live_agent, "fail_reason", None) is None))
            if action not in ("REPORT", "DIE") and not handed_off:
                self._enforce_task_energy_ceiling(agent_id, task_id)

    def initialize_colony(self, problem_spec: str):
        """System entry point. Bootstraps the first task and the root agent."""
        # PATCH 33: the success cache lives for ONE colony run. The notebook
        # builds one MemoryStore and reuses it for every new_colony(), so
        # without this a later prompt could be served accepted entries from
        # an earlier one. Ghosts are untouched: they persist by design.
        # getattr: test fakes of the store need not implement it.
        if self.memory_store is not None:
            clear_session = getattr(self.memory_store, "clear_session", None)
            if callable(clear_session):
                clear_session()
        self.root_task_id = "root_task_0"

        if self.phaser is not None:
            spec = self.phaser.parse_problem(problem_spec)
            spec = self.phaser.estimate_complexity(spec)
        else:
            spec = {
                "raw_text": problem_spec,
                "goal": problem_spec,
                "goal_vector": None,
                "requirement": [],
                "domain": "General Discourse",
                "colony_budget": self.colony.budget_remaining or 100,
            }
        self.spec = spec

        self.colony.budget_remaining = spec.get("colony_budget", self.colony.budget_remaining)
        # The phaser's budget replaces whatever ColonyState was constructed
        # with, so the ledger's reconciliation baseline has to move with it.
        # Safe here: no debit has landed yet (the root spawn is below).
        self.colony.starting_budget = self.colony.budget_remaining

        if self.budget_override is not None:
            # starting_budget has to move with budget_remaining: the ledger's
            # reconciliation check compares starting_budget - debits + credits
            # against budget_remaining, and would report a phantom leak of
            # exactly the override delta if only one of the two changed.
            print(f"[initialize_colony] BUDGET OVERRIDE ACTIVE: phaser proposed "
                  f"{self.colony.budget_remaining}, using {self.budget_override}. "
                  f"This run is NON-STANDARD.")
            self.colony.budget_remaining = self.budget_override
            self.colony.starting_budget = self.budget_override

        # Printed unconditionally, on both paths. The override branch above
        # only announces itself when it fires, which made "the override did
        # not land" and "the override landed on the same number the phaser
        # proposed" look identical in a log. This line is the one to read.
        print(f"[initialize_colony] EFFECTIVE COLONY BUDGET: "
              f"{self.colony.budget_remaining} "
              f"(source={'override' if self.budget_override is not None else 'phaser'}, "
              f"phaser proposed {spec.get('colony_budget')}, "
              f"starting_budget={self.colony.starting_budget})")

        self.colony.goal_embedding = spec.get("goal_vector")
        self.goal_clause_texts = list(spec.get("goal_clauses") or [])
        self.goal_clause_embeddings = list(spec.get("goal_clause_vectors") or [])

        goal_text = spec.get("goal", problem_spec)
        print(f"[initialize_colony] goal ({len(goal_text)} chars): {goal_text}")
        print(f"[initialize_colony] software framing guard: "
              f"{'ON' if self._software_framing_guard_active else 'OFF'}")

        root_node = TaskNode(
            task_id=self.root_task_id,
            description=goal_text,
            required_role="decomposer",
            requirements=spec.get("requirement", []),
        )
        self.task_graph.add_task(root_node)
        # Problem_Phaser dedupes/caps the goal before returning it in spec --
        # the root task description (what the judge prints on WARN) must be
        # that exact same string, not a re-derived one, since goal_vector
        # was embedded from it. A mismatch here means a future edit split
        # the description assignment from the phaser's cleaned goal again.
        assert self.task_graph.tasks[self.root_task_id].description == goal_text

        print(f"Colony initialized. Bootstrapping root task: {self.root_task_id}")
        if (self.spawn_agent(role="decomposer", task_id=self.root_task_id) is None
                and "admission_root_refused" in self.colony.verdict_counts):
            # Nothing smaller than root + one decomposer + one executor can
            # answer anything. Closed now rather than left pending, which
            # the deadlock watchdog would retry every tick to MAX_TICKS.
            self._abandon_task(
                self.root_task_id, None, None, 0,
                reason=("the budget is below the cheapest possible "
                        "decomposition of the task"),
                reason_label="admission",
            )

    def spawn_agent(self, role: str, task_id: str, parent_id: Optional[str] = None, ghost_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """SPAWN: Creates an agent, assigns it to a task, and deducts energy.

        NOTE on dependency gating: this still eagerly creates a live Agent
        and debits spawn energy even if the task isn't ready yet (unmet
        dependencies). That's an intentional, smaller-footprint choice --
        see TaskGraph.assign_agent's fix -- rather than deferring agent
        creation itself. _run_live_agents already refuses to tick an agent
        whose task_node.status != 1, so a dependency-gated agent simply sits
        idle in self.live_agents until TaskGraph flips its task to running.
        """
        spawn_cost = self.energy_when_new_by_role.get(role, self.energy_when_new)
        if not self.colony.can_spawn(spawn_cost):
            print(
                f"Cannot spawn {role} for task {task_id}: insufficient energy "
                f"(budget={self.colony.budget_remaining}, cost={spawn_cost})."
            )
            return None

        # Admission. A task's first agent was reserved when its parent's SPAWN
        # batch was admitted, so only the root's bootstrap is checked here. A
        # TASK TOO LARGE conversion was admitted in handle_failure. Any other
        # spawn onto a task that already had an agent is a respawn: new work,
        # so it reserves a whole attempt again (sunk := spend so far) and must
        # fit like anything else.
        reservation = self._reservation(task_id)
        if reservation.attempts == 0:
            if task_id == self.root_task_id and not self._admits():
                print(f"  [admission] root refused: its cheapest decomposition "
                      f"({self._remaining_reservation(task_id):.0f}) does not fit "
                      f"budget {self.colony.starting_budget} minus the "
                      f"{self.ADMISSION_FLOOR} floor.")
                self.colony.record_verdict("admission_root_refused")
                return None
        elif task_id in self._admitted_conversions:
            self._admitted_conversions.discard(task_id)
        else:
            saved = (reservation.k, reservation.spawned, reservation.sunk)
            if role == "decomposer":
                if reservation.spawned:
                    # The ceiling keeps pricing the batch this task already
                    # paid for; only admission re-plans from k=0.
                    reservation.k_peak = max(reservation.k_peak, reservation.k)
                reservation.k, reservation.spawned = 0, False  # it plans again
            reservation.sunk = self.colony.task_energy_spent.get(task_id, 0)
            if not self._admits():
                # Enough numbers to tell real exhaustion (spent + attempt
                # alone over the limit) from other open work's reservations.
                attempt = self._remaining_reservation(task_id)
                committed = self.committed_energy()
                spent = self.colony.starting_budget - self.colony.budget_remaining
                reservation.k, reservation.spawned, reservation.sunk = saved
                print(f"  [admission] respawn on {task_id} refused: another "
                      f"attempt does not fit the uncommitted budget (attempt "
                      f"{attempt:.0f}, spent {spent}, committed with it "
                      f"{committed:.0f}, limit "
                      f"{self.colony.starting_budget - self.ADMISSION_FLOOR}).")
                self.colony.record_verdict("admission_respawn_refused")
                return None

        agent_id = self._generate_id()

        task_node = self.task_graph.tasks.get(task_id)
        description = task_node.description if task_node else task_id
        requirements = task_node.requirements if task_node else []

        if ghost_context is None and self.spec:
            overall_goal = self.spec.get("goal") or self.spec.get("raw_text")
            if overall_goal:
                overall_goal = _dedupe_repeated_sentences(overall_goal, max_chars=300)
                ghost_context = f"[Project goal] {overall_goal}"

        parent_node = self.colony.get_agent(parent_id) if parent_id else None
        generation = (parent_node.generation + 1) if parent_node else 0

        new_agent = AgentNode(
            agent_id=agent_id,
            role=role,
            status="running",
            parent_id=parent_id,
            task=description,
            task_id=task_id,
            ghost_context=ghost_context,
            requirements=requirements,
            generation=generation,
            # PATCH 8. Set only on a task the requirements filter emptied, so
            # every agent that runs it -- first attempt, respawn, or a TASK
            # TOO LARGE conversion, all of which come through here -- gets the
            # project referent. Read off the TaskNode rather than recomputed,
            # so a conversion cannot lose it the way the ghost-context path
            # loses the project goal line on attempt 2.
            goal_referent=getattr(task_node, "goal_referent", None),
            # PATCH 29. Every agent, whatever its role or attempt.
            request_text=self._request_text(),
        )

        self.colony.register_agent(new_agent)
        self.task_graph.assign_agent(task_id, agent_id)
        reservation.attempts += 1
        # PATCH 29 ledger row: the log prints a full prompt only on DIE, so
        # "did the fee agent see 9,000" is otherwise unanswerable from it.
        self.request_prompt_counts["agents"] += 1
        if new_agent.request_text:
            self.request_prompt_counts["with_request"] += 1
        if parent_node is not None:
            # Read by Agent.final_actions: a decomposer with no child yet
            # finishes by SPAWNing, one whose children exist by REPORTing.
            parent_node.has_spawned = True
        
        # Debit the initialization energy cost from the colony budget
        self.colony.debit_energy(agent_id, spawn_cost, category="spawn")
        self.agent_counter += 1

        if self.model is not None and self.tokeniser is not None:
            live_agent = Agent(self.tokeniser, self.model, self.messenger, new_agent)
            # Read at prompt time, so a decomposer's prompt states its
            # subtasks' current status from the graph, not from whatever
            # child-result injections are still inside its thought window.
            live_agent.child_status_fn = (
                lambda tid=task_id: self._direct_child_statuses(tid))
            self.live_agents[agent_id] = live_agent

        return agent_id

    _SUBTASK_KEY_SYNONYMS = {
        "depends_on": "dependencies",
        "deps": "dependencies",
        "desc": "description",
    }
    _SUBTASK_VALID_KEYS = ("role", "task", "label", "dependencies", "description")

    @classmethod
    def _normalize_subtask_keys(cls, sub: dict) -> dict:
        """Returns a copy of sub with keys cleaned up: whitespace stripped,
        known synonyms mapped to their canonical name, and anything left
        over (a typo like "rolle"/"taask") fuzzy-matched against the valid
        key set."""
        normalized = {}
        for key, value in sub.items():
            clean_key = key.strip() if isinstance(key, str) else key
            if clean_key in cls._SUBTASK_KEY_SYNONYMS:
                # Exact synonyms are checked first and win outright -- a
                # known synonym like "depends_on" must never fall through
                # to the fuzzy pass below and risk resolving to some other
                # valid key instead of its intended canonical name.
                normalized[cls._SUBTASK_KEY_SYNONYMS[clean_key]] = value
            elif isinstance(clean_key, str) and clean_key not in cls._SUBTASK_VALID_KEYS:
                fuzzy_key = normalize_identifier(clean_key, cls._SUBTASK_VALID_KEYS, cutoff=0.75)
                normalized[fuzzy_key or clean_key] = value
            else:
                normalized[clean_key] = value
        return normalized

    _REQ_FILTER_STOPWORDS = frozenset({
        "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at",
        "must", "should", "with", "without", "against", "as", "is", "are",
        "be", "by", "that", "this", "it", "its", "into", "than", "not",
        "specify", "design", "estimate", "under", "since", "base", "cannot",
        "survive", "remain", "via", "show", "just", "but", "final",
        # PATCH 9. Words that carry no domain meaning in a requirement
        # sentence, so an overlap on one of them is noise, not relevance.
        # Measured: task_b029d4e5 ("Group records by their normalized
        # title-author pairings...") qualified for tier 1 -- the strongest
        # tier -- on "per" and "one" alone, against "Answers must remain
        # within one paragraph per question."
        #
        # Three groups, all of them structural rather than topical:
        #   quantifiers and determiners,
        #   obligation verbs (every requirement is phrased as one, so they
        #     match everything and distinguish nothing),
        #   and the meta-vocabulary the phaser uses to talk ABOUT answers
        #     rather than about the domain.
        # Deliberately NOT added: "books", "titles", "votes", "members",
        # "readings" -- those are this project's actual subject matter and
        # an overlap on them is real signal.
        "per", "one", "two", "three", "each", "any", "all",
        "need", "needs", "needed", "required", "require", "requires",
        "may", "can", "will", "shall",
        "answer", "answers", "question", "questions", "response",
        "responses", "output", "outputs", "result", "results",
        "solution", "solutions", "within", "prior", "set",
    })
    # PATCH 9. Tier 1 kept every requirement sharing >=1 exact significant
    # word, with no cap -- a strictly weaker test than tier 2 below it,
    # which requires a stem match AND returns at most two. A single shared
    # token is noise at this corpus size. Tier 1 now needs two shared words
    # and is capped like tier 2; a requirement that reaches only one word
    # falls through to the stem path rather than being promoted on it.
    REQ_EXACT_MIN_WORDS = 2
    REQ_MAX_INHERITED = 2

    @classmethod
    def _significant_words(cls, text: str) -> set:
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
        return {w for w in words if w not in cls._REQ_FILTER_STOPWORDS}

    # Longest first: the first suffix that fits wins.
    _STEM_SUFFIXES = (
        "ations", "ation", "ments", "ment", "ings", "ing", "ers", "ies",
        "ied", "ed", "er", "es", "ly", "s",
    )

    @classmethod
    def _stem(cls, word: str) -> str:
        """
        Light suffix-stripping stem, capped at five characters.

        The plain five-character prefix this replaces could never make
        'vote'/'votes'/'voting'/'voters' meet ('votes' vs 'votin' vs
        'voter'), and skipped words under five letters entirely, so 'book'
        and 'books' could only ever meet as an exact match. A subtask
        about counting book votes then shared "no vocabulary" with a goal
        about choosing books by vote, and the derived-subtask guard threw
        the batch away.
        """
        for suffix in cls._STEM_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                break
        if word.endswith("e") and len(word) > 3:
            word = word[:-1]
        return word[:5]

    @classmethod
    def _stems(cls, words: set) -> set:
        """Stems for a word set, so 'vote'/'votes'/'voting' and
        'cooled'/'cooling' can still meet."""
        return {cls._stem(w) for w in words}

    _CODE_HINT_RE = re.compile(
        r"^\s*(def |class |import |from \w+ import |return |for \w+ in |if .+:|"
        r"[\w\.\[\]]+\s*=\s*\S)",
        re.MULTILINE,
    )

    @classmethod
    def _looks_like_code(cls, output) -> bool:
        """
        True when this output is source code rather than prose, so tier 2's
        prose-vs-prose similarity score can be skipped (see the call site in
        handle_completion for why the score is meaningless on code).

        Deliberately conservative: a fenced block or a clean ast.parse is
        proof; otherwise several code-shaped lines are required, so an
        English answer that happens to contain one `x = 3` is not mistaken
        for a program.
        """
        text = str(output or "").strip()
        if not text:
            return False
        if "```" in text:
            return True
        try:
            tree = ast.parse(text)
        except SyntaxError:
            pass
        else:
            # A bare sentence parses as an Expr of a Name/Call chain often
            # enough that "it parsed" alone is not evidence. Require at least
            # one real statement.
            if any(not isinstance(node, ast.Expr) for node in tree.body):
                return True
        return len(cls._CODE_HINT_RE.findall(text)) >= 3

    def _filter_requirements_for_task(self, description: str, requirements: list) -> list:
        """
        The constraints a child task actually inherits.

        FIX (inverted fallback): "no keyword overlap" used to mean "inherit
        EVERY requirement" -- exactly backwards. A subtask that shares no
        vocabulary with a requirement is the subtask that requirement has
        nothing to do with. That fallback is how a task like "implement the
        binomial PMF" ended up carrying "operating temperature must exceed
        1450C" and DIEing on a contradiction it was never asked to satisfy.

        No overlap now yields nothing. Between the two, a softer prefix-stem
        pass catches morphological near-misses ("cooling" vs "coolant") and
        returns at most the two best-scoring requirements -- enough to keep a
        genuinely-related constraint from being dropped over a suffix, not
        enough to reinstate the dump-everything behaviour.

        PATCH 9 (tier 1). The exact pass used to keep every requirement
        sharing ONE word, unbounded -- weaker than the stem tier beneath it,
        which demands a match and caps the result at two. So the "strongest"
        tier was the one most easily satisfied by noise, and run 4's
        task_b029d4e5 inherited a constraint on the strength of "per" and
        "one". It now takes REQ_EXACT_MIN_WORDS shared words, returns the
        REQ_MAX_INHERITED best, and falls through to the stem pass when no
        requirement reaches the bar instead of settling for a one-word hit.

        PATCH 9 (logging). Every tier logs now, tier 1 included. It was the
        only silent one, so a run's log showed the two degraded tiers and
        said nothing about the 6 tasks on the nominally-good path -- the
        tier distribution could not be read off a log at all.
        """
        if not requirements:
            return []
        task_words = self._significant_words(description)
        if not task_words:
            print("  [requirements] task description has no significant words -- "
                  "inheriting no requirements.")
            return []

        scored_exact = []
        for r in requirements:
            shared = task_words & self._significant_words(r)
            if len(shared) >= self.REQ_EXACT_MIN_WORDS:
                scored_exact.append((len(shared), r, shared))
        if scored_exact:
            scored_exact.sort(key=lambda triple: -triple[0])
            kept = [r for _, r, _ in scored_exact[:self.REQ_MAX_INHERITED]]
            shared_words = sorted(set().union(*[s for _, _, s in scored_exact[:self.REQ_MAX_INHERITED]]))
            print(f"  [requirements] tier 1 (exact) for '{description[:60]}' -- "
                  f"inheriting {len(kept)} of {len(requirements)} requirement(s) "
                  f"on shared words {shared_words}"
                  + (f"; {len(scored_exact) - len(kept)} more qualified and were "
                     f"capped at {self.REQ_MAX_INHERITED}"
                     if len(scored_exact) > len(kept) else ""))
            return kept

        task_stems = self._stems(task_words)
        scored = []
        for r in requirements:
            overlap = len(task_stems & self._stems(self._significant_words(r)))
            if overlap:
                scored.append((overlap, r))
        if scored:
            scored.sort(key=lambda pair: -pair[0])
            partial = [r for _, r in scored[:self.REQ_MAX_INHERITED]]
            print(f"  [requirements] tier 2 (stem) -- no requirement reached "
                  f"{self.REQ_EXACT_MIN_WORDS} exact shared words for "
                  f"'{description[:60]}' -- inheriting the {len(partial)} "
                  f"closest requirement(s) by stem overlap only.")
            return partial

        print(f"  [requirements] tier 3 -- no keyword overlap at all for "
              f"'{description[:60]}' -- inheriting NO requirements (previously "
              f"this inherited all of them, which is what forced unrelated "
              f"constraints onto narrow subtasks). PATCH 8 attaches the "
              f"project goal as this task's referent instead.")
        return []

    _VALID_ROLES = ("decomposer", "executor", "verifier")

    def _enforce_child_role(self, requested_role: str, parent_id: Optional[str]) -> str:
        normalized_role = normalize_identifier(requested_role, self._VALID_ROLES, cutoff=0.75)
        if normalized_role:
            requested_role = normalized_role

        parent_node = self.colony.get_agent(parent_id) if parent_id else None
        if parent_node is None or parent_node.role != "decomposer":
            return requested_role
        if requested_role == "verifier":
            return requested_role
        return "decomposer" if parent_node.generation == 0 else "executor"

    # Stands in for a result that scrub_result cut to nothing. Shaped like
    # the [ABANDONED ...] and [NOT STARTED ...] markers so the parent can
    # see there is a gap, and never "" or the degenerate text.
    DEGENERATE_RESULT_MARKER = (
        "[NO USABLE OUTPUT -- this subtask's result was degenerate from its "
        "first sentence ({reason}), so nothing from it was kept.]"
    )

    def _scrub_result(self, result, where: str, task_id: Optional[str] = None):
        """
        Returns `result` with its degenerate parts removed (text_utils.scrub_result).

        Called at promotion and again at each injection into another
        agent's prompt, so all consumers get the same cleaned text. The
        promotion call does the actual work. The injection calls are no-ops
        on text that was already scrubbed (scrub_result is idempotent), so
        they only fire for results that reached storage another way, such as
        an abandoned task's salvaged partial. The counters record which of
        the two did the cutting.

        Source code is returned unchanged, the same exemption the report-trim
        uses. So is anything that is not a non-empty string.
        """
        if not isinstance(result, str) or not result.strip() or self._looks_like_code(result):
            return result
        kept, reasons = scrub_result(result, exemplars=PROMPT_EXEMPLARS,
                                     grounding=self._grounding_for(task_id))
        if not reasons:
            return result

        label = task_id or "?"
        self.colony.record_verdict(f"result_scrubbed_at_{where}")
        print(f"  [result-scrub:{where}] {label}: removed {', '.join(reasons)} "
              f"({len(result)} -> {len(kept)} chars).")
        if kept.strip():
            return kept

        self.colony.record_verdict(f"result_empty_after_scrub_at_{where}")
        print(f"  [result-scrub:{where}] {label}: nothing survived -- replaced "
              f"with an explicit no-usable-output marker.")
        return self.DEGENERATE_RESULT_MARKER.format(reason=reasons[0])

    def _grounding_for(self, ref: Optional[str]) -> str:
        """What a task's result was allowed to draw on: the project goal, the
        task's own description and requirements, and its prerequisites'
        results. The status-report trim keeps a timestamp or metric found in
        here and cuts one that is not. `ref` is a task_id, or an agent_id
        (handle_parent_notification labels by child agent), resolved to its
        task. Same text at promotion and at every injection of one task, so
        the scrub stays idempotent across them."""
        parts = []
        if self.spec:
            parts += [str(self.spec.get("raw_text") or ""), str(self.spec.get("goal") or "")]
        task = self.task_graph.tasks.get(ref) if ref else None
        if task is None and ref:
            agent = self.colony.get_agent(ref)
            task = self.task_graph.tasks.get(getattr(agent, "task_id", None)) if agent else None
        if task is not None:
            parts.append(str(task.description or ""))
            parts += [str(r) for r in (task.requirements or [])]
            for dep_id in task.dependencies or []:
                dep = self.task_graph.tasks.get(dep_id)
                if dep is not None and dep.result is not None:
                    parts.append(str(dep.result))
        return "\n".join(parts)

    def _build_dependency_context(self, dependencies: Optional[list]) -> str:
        if not dependencies:
            return ""
        lines = []
        for dep_id in dependencies:
            dep_task = self.task_graph.tasks.get(dep_id)
            if dep_task is None:
                continue
            dep_result = dep_task.result
            if dep_result is None:
                dep_result = self.colony.results.get(dep_id)
            if dep_result is None:
                continue
            # Scrubbed before the cap, so a degenerate tail cannot fill the
            # 400 characters a sibling actually sees.
            snippet = self._scrub_result(
                str(dep_result).strip(), "dependency_injection", task_id=dep_id
            )
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            lines.append(f'- From completed step "{dep_task.description}": {snippet}')
        if not lines:
            return ""
        return (
            "Shared state from completed prerequisite steps -- USE THESE "
            "EXACT VALUES, do not re-derive or invent your own numbers for "
            "anything already computed here:\n" + "\n".join(lines) + "\n"
        )

    def _spawn_child_task(self, description: str, role: str, parent_id: Optional[str],
                           dependencies: Optional[list] = None, task_id: Optional[str] = None,
                           label: Optional[str] = None):
        if task_id is None:
            task_id = self._generate_task_id()
        full_requirements = self.spec.get("requirement", []) if self.spec else []

        if role == "decomposer":
            requirements = full_requirements
        else:
            requirements = self._filter_requirements_for_task(description, full_requirements)

        # PATCH 8. The filter/guard band, closed. _filter_requirements_for_task
        # compares a task against each requirement individually;
        # _has_no_requirement_overlap (the derived-subtask guard) compares it
        # against the UNION of requirements, goal, raw_text and parent task.
        # The guard's corpus is a strict superset, so a task can be stripped
        # to zero requirements by the filter and still sail past the guard on
        # a word it shares with the goal -- which is exactly where run 4's
        # drift lived. A task landing there now carries the goal text as a
        # referent so it is not left with no project-level anchor anywhere.
        # Background only: it is NOT appended to `requirements`, so nothing
        # asks this agent to satisfy the project, and the inverted-fallback
        # bug the filter exists to prevent stays fixed.
        goal_referent = None
        if role != "decomposer" and not requirements:
            goal_referent = (self.spec.get("goal") or self.spec.get("raw_text")) if self.spec else None
            if goal_referent:
                goal_referent = _dedupe_repeated_sentences(str(goal_referent), max_chars=300)
                self.colony.record_verdict("goal_referent_attached")

        parent_node = self.colony.get_agent(parent_id) if parent_id else None
        child_node = TaskNode(
            goal_referent=goal_referent,
            task_id=task_id,
            description=description,
            dependencies=dependencies or [],
            required_role=role,
            requirements=requirements,
            # The spawning agent's TASK, fixed here. parent_id is an agent id:
            # that agent can be retired and its children handed to its own
            # parent (ColonyState.unregister_agent). The task it was working
            # on does not change.
            parent_task_id=getattr(parent_node, "task_id", None),
            label=self._short_label(label),
        )
        # N2b: embed this task's own description once, at spawn time, so
        # judge.decide's tier-2 check has a target that actually matches
        # what this task's agent was asked to do -- not the colony's overall
        # goal, which is what every child was being scored against before.
        # PATCH 12: when this task came through a SPAWN batch the goal-drift
        # gate has already encoded it. Popped rather than read, so the cache
        # cannot grow across a run; anything the gate never saw (the
        # bootstrap root, an overflow drain re-entering here) still falls
        # through to encoding it now.
        if task_id in self._gate_description_embeddings:
            child_node.description_embedding = self._gate_description_embeddings.pop(task_id)
        elif self.embed_model is not None:
            child_node.description_embedding = self._embed_description(description)

        self._measure_spawn_drift(child_node, parent_task_id=child_node.parent_task_id)
        self._flag_software_shaped_task(child_node)
        self._flag_artifact_shaped_task(child_node.task_id, child_node.description)
        self._flag_unsupplied_figures(child_node.task_id, child_node.description)
        self.task_graph.add_task(child_node)

        if self.spawn_agent(role=role, task_id=task_id, parent_id=parent_id,
                            ghost_context=self._child_ghost_context(dependencies)) is None:
            self._close_unstartable_child(task_id, parent_id)

    def _project_goal_text(self) -> Optional[str]:
        """PATCH 8. The goal as one short line, for the tier-3 prompt and the
        zero-requirement referent. Deduped and capped the same way
        _child_ghost_context caps it, so a phaser goal that looped on itself
        cannot inflate every critique prompt for the rest of the run."""
        if not self.spec:
            return None
        goal_text = self.spec.get("goal") or self.spec.get("raw_text")
        if not goal_text:
            return None
        return _dedupe_repeated_sentences(str(goal_text), max_chars=300)

    def _print_clause_reference_comparison(self, samples) -> None:
        """PATCH 17 ledger section. Whole-goal and max-over-clauses scored on
        the same subtasks, side by side: who sets each ceiling, and who falls
        under GOAL_DRIFT_GATE_FRACTION of each. Measure-only -- the gate
        still reads goal_drift, and is itself measure-only."""
        both = [s for s in samples
                if s.get("goal_drift") is not None and s.get("clause_drift") is not None]
        print(f"    PATCH 17 -- whole goal vs max over "
              f"{len(self.goal_clause_embeddings)} clause(s), measure-only:")
        for i, text in enumerate(self.goal_clause_texts, 1):
            print(f"      c{i}: {text}")
        if not both:
            print("      (no subtask has both scores -- clause vectors missing?)")
            return

        def _short(s):
            d = " ".join(str(s["description"] or "").split())
            return d[:52] + "..." if len(d) > 55 else d

        below = {}
        for key, label in (("goal_drift", "whole-goal"), ("clause_drift", "max-clause")):
            ceiling = max(both, key=lambda s: s[key])
            threshold = GOAL_DRIFT_GATE_FRACTION * ceiling[key]
            below[key] = {s["task_id"] for s in both if s[key] < threshold}
            print(f"      {label}: ceiling {ceiling[key]:.3f} set by "
                  f"{ceiling['task_id']} ({_short(ceiling)}); "
                  f"{len(below[key])} of {len(both)} under "
                  f"{GOAL_DRIFT_GATE_FRACTION:g} x max = {threshold:.3f}")
        print("      lowest 8 by max-clause (* = under that reference's threshold):")
        for s in sorted(both, key=lambda s: s["clause_drift"])[:8]:
            w = "*" if s["task_id"] in below["goal_drift"] else " "
            c = "*" if s["task_id"] in below["clause_drift"] else " "
            print(f"        goal={s['goal_drift']:.3f}{w} "
                  f"clause={s['clause_drift']:.3f}{c}(c{s['clause_index']})  "
                  f"{s['task_id']}  {_short(s)}")

    def _clause_drift(self, embedding):
        """PATCH 17 (measure-only). This embedding's cosine to the goal
        clause it is CLOSEST to, and that clause's 1-based index, as
        (score, index) -- or (None, None) when no clause is measurable.

        Why max over clauses: the goal is a conjunction and its embedding
        sits in the average of its clauses, so a subtask serving exactly one
        clause scores low against the whole goal by construction, while the
        run's ceiling is set by whichever subtask straddles two (run 6:
        0.658, run 7: 0.793). See goal_clauses in text_utils.
        """
        best, best_i = None, None
        for i, clause_vec in enumerate(self.goal_clause_embeddings):
            score = _cosine(embedding, clause_vec)
            if score is not None and (best is None or score > best):
                best, best_i = score, i + 1
        return best, best_i

    def _measure_spawn_drift(self, child_node: TaskNode, parent_task_id: Optional[str]):
        """
        PATCH 7. Score every spawned subtask against the colony goal and
        against its own parent task, store both on the TaskNode, and log them
        unconditionally.

        Runs on tasks that are ALREADY going to exist. PATCH 12's gate acts
        earlier, on the prepared SPAWN batch, so anything reaching here was
        kept -- and the gate records its own dropped subtasks straight into
        goal_drift_samples so the distribution below stays complete.
        parent_drift remains measure-only and nothing is gated on it: run 5
        showed it both misses the gradual cases and false-alarms on good ones
        (see _screen_goal_drift_batch). The point of the distribution is
        unchanged: run 4's ISBN subtree walked from "Count total number of distinct books available"
        to "Remove duplicate entries where titles and authors exactly match
        case-insensitively" one plausible hop at a time, and until there are
        real numbers for both kinds of distance there is no defensible place
        to put a threshold.

        The two are different questions and are reported separately:
          goal_drift   -- CUMULATIVE. How far from the project this task sits,
                          however many hops it took to get there.
          parent_drift -- PER-HOP. How far this one SPAWN moved.
        If per-hop stays high while goal drift decays, no single decomposer
        did anything obviously wrong and only the sum is bad -- which is a
        different fix (a cumulative budget) from one bad hop (a spawn gate).

        Both vectors already existed and were never multiplied: goal_vector
        is computed at problem_phaser.py:332 and read only by the phaser's own
        budget multiplier, and description_embedding has been set at spawn
        since the N2b fix purely as a tier-2 target.
        """
        goal_drift = _cosine(child_node.description_embedding, self.colony.goal_embedding)
        parent_task = self.task_graph.tasks.get(parent_task_id) if parent_task_id else None
        parent_drift = _cosine(
            child_node.description_embedding,
            getattr(parent_task, "description_embedding", None),
        )
        clause_drift, clause_index = self._clause_drift(child_node.description_embedding)
        child_node.goal_drift = goal_drift
        child_node.parent_drift = parent_drift
        self.goal_drift_samples.append({
            "task_id": child_node.task_id,
            "description": child_node.description,
            "goal_drift": goal_drift,
            "parent_drift": parent_drift,
            "clause_drift": clause_drift,
            "clause_index": clause_index,
            "gated": child_node.task_id in self._gate_would_drop,
        })

        # Why a number is missing matters as much as the number. A zero-norm
        # goal vector means the phaser fell back (problem_phaser.py:601/:636)
        # and EVERY drift figure this run is absent for the same reason; a
        # missing parent embedding is routine on a root child, which has no
        # embedded parent description.
        if goal_drift is None:
            self.colony.record_verdict("spawn_drift_goal_unmeasurable")
        if parent_drift is None:
            self.colony.record_verdict("spawn_drift_parent_unmeasurable")

        def _fmt(value):
            return f"{value:.3f}" if value is not None else "n/a"

        kept = ("kept, GATE WOULD HAVE DROPPED"
                if child_node.task_id in self._gate_would_drop else "kept")
        clause = (f"{_fmt(clause_drift)}(c{clause_index})"
                  if clause_index is not None else "n/a")
        print(f"  [spawn-drift] {child_node.task_id} goal={_fmt(goal_drift)} "
              f"clause={clause} parent={_fmt(parent_drift)} ({kept}): "
              f"{str(child_node.description)[:70]!r}")

    def _flag_software_shaped_task(self, child_node: TaskNode):
        """
        PATCH 10 (measure-only). Ask the software-request question of a
        SUBTASK, not just of the user's original request.

        _SOFTWARE_REQUEST_RE already lists "database", "script", "sql",
        "dataset", "regex", "json" and "csv", but asks_for_software runs it on
        spec["raw_text"] alone, to decide whether the framing guard is on at
        all. Nothing asked it of a subtask, so run 4 spawned "Query the
        catalog database to extract records" and "Write a script that
        processes raw input rows" untouched -- and the first became the single
        most expensive task of the run at 85/108, then promoted.

        This is a different question from the two checks that already run on a
        subtask. software_artifact_reason asks "is this text written as code"
        (syntax); plain_register asks "does this text use a code-coded word"
        (vocabulary, and it rewords rather than judges). This asks "is this
        task ASKING for software" (substance) -- the question the scan found
        nothing was asking.

        Counted and logged only. No reject, no reword, no respawn: the whole
        point of this round is to find out how many subtasks it would fire on
        before it is allowed to cost anything.
        """
        if not self._software_framing_guard_active:
            return
        hits = software_request_words(child_node.description)
        if not hits:
            return
        self.colony.record_verdict("software_shaped_task_detected")
        print(f"  [software-shaped task] {child_node.task_id} asks for software "
              f"{hits} on a project that never did (measure-only, not blocked): "
              f"{str(child_node.description)[:70]!r}")

    def _child_ghost_context(self, dependencies: Optional[list]) -> Optional[str]:
        """A child's starting ghost context: the project goal as background,
        plus whatever its prerequisites have produced so far."""
        goal_text = (self.spec.get("goal") or self.spec.get("raw_text")) if self.spec else None
        if goal_text:
            goal_text = _dedupe_repeated_sentences(goal_text, max_chars=300)
        goal_line = (
            f"[Project goal -- for background only, NOT your task's scope: "
            f"your own task and constraints above are the ONLY things you "
            f"need to satisfy] {goal_text}\n"
        ) if goal_text else ""
        dep_context = self._build_dependency_context(dependencies)
        return (goal_line + dep_context) or None

    def _thread_results_to_unblocked_dependents(self, task_node: Optional[TaskNode]):
        """Hand a closed task's result to the dependents it just unblocked.

        A dependent's context is built when it is spawned, which for a
        sibling in the same SPAWN batch (the sequential default makes every
        child after the first one) is BEFORE its prerequisite has a result.
        Its agent then waits at status 0 and starts at status 1 with the
        "Shared state" block that was empty when it was built. The block has
        to be rebuilt when the prerequisite closes, and this is where the
        prerequisite's result actually reaches the sibling's prompt.
        _build_dependency_context scrubs and caps it here.

        Called after the result is stored, from every path that closes a
        task and releases its dependents: promotion, abandonment and a
        success-cache hit. Only a dependent whose LAST prerequisite this
        was (in_degree 0) and whose agent has not ticked yet is touched, so
        a running agent's prompt never changes under it.
        """
        if task_node is None:
            return
        for dependent_id in task_node.dependents:
            dep_task = self.task_graph.tasks.get(dependent_id)
            if dep_task is None or dep_task.in_degree != 0 or dep_task.agent_id is None:
                continue
            live_agent = self.live_agents.get(dep_task.agent_id)
            if (live_agent is None or getattr(live_agent, "thought_process", "")
                    or getattr(live_agent, "non_terminal_cycles", 0)):
                continue
            context = self._child_ghost_context(dep_task.dependencies)
            if context and context != getattr(live_agent, "ghost_context", None):
                live_agent.ghost_context = context
                self.colony.record_verdict("dependency_context_threaded")
                print(f"  [dependency-context] {dep_task.agent_id}/{dependent_id} "
                      f"unblocked by {task_node.task_id} -- prerequisite results "
                      f"threaded into its context.")

    def _close_unstartable_child(self, task_id: str, parent_id: Optional[str]):
        """
        spawn_agent refused this child for lack of energy after its TaskNode
        was already in the graph. Left alone, that task sat at pending with
        no agent, so _process_unblocked_tasks (or the watchdog) started it
        later with no parent_id: its result went nowhere, the parent never
        heard back, and anything depending on it stayed blocked. Retrying
        with the right parent would not help either -- energy only goes
        down (the one credit path, consolidation, never fires), so a spawn
        refused now is refused on every later tick.

        So the task is closed on the spot, the same way _abandon_task closes
        one: marked failed with a result saying it never started, its
        dependents released, and the parent notified.
        """
        self.unstarted_tasks.add(task_id)
        task_node = self.task_graph.tasks.get(task_id)
        description = task_node.description if task_node is not None else "<unknown task>"
        result = (
            "[NOT STARTED -- the colony did not have enough energy left to "
            "start an agent for this subtask, so no work was done on it.]"
        )
        print(f"  [spawn] {task_id} (\"{description[:60]}\") could not be started "
              f"for lack of energy -- closing it instead of leaving it pending "
              f"with no parent.")
        if task_node is not None:
            task_node.status = 3
            task_node.result = result
        self.colony.store_result(task_id, result)
        self._release_dependents(task_node)
        self._thread_results_to_unblocked_dependents(task_node)
        self._drain_pending_overflow(parent_id)
        if parent_id:
            self.messenger.push_event(
                "parent_notification",
                "orchestrator",
                {"parent_id": parent_id, "child_id": task_id, "task_id": task_id,
                 "result": result}
            )

    def _release_dependents(self, task_node: Optional[TaskNode]):
        """Unblocks the dependents of a task closed as failed. complete_task()
        does this as a side effect of marking status=2, which would be a lie
        here, so the in_degree bookkeeping is repeated explicitly."""
        if task_node is None:
            return
        for dependent_id in task_node.dependents:
            dep_task = self.task_graph.tasks.get(dependent_id)
            if dep_task is None:
                continue
            dep_task.in_degree -= 1
            if dep_task.in_degree == 0 and dep_task.agent_id is not None:
                dep_task.status = 1

    def _drain_pending_overflow(self, parent_id: Optional[str]):
        """
        Spawns exactly one queued overflow subtask (if any) for this
        decomposer -- called whenever one of its children genuinely
        completes and frees a fan-out slot. See pending_overflow's
        docstring in __init__ for why this replaces the old
        bundle-into-one-oversized-child approach.

        Draining exactly one per freed slot maintains the same "at most
        `cap` concurrently in-flight children" invariant the batch was
        capped to in the first place, without needing to separately track
        how many of this parent's children are currently live.
        """
        if not parent_id:
            return
        queue = self.pending_overflow.get(parent_id)
        if not queue:
            return
        next_spawn_kwargs = queue.pop(0)
        if not queue:
            del self.pending_overflow[parent_id]
        print(
            f"  [handle_spawn] fan-out slot freed under decomposer "
            f"{parent_id} -- draining next queued subtask "
            f"({len(queue)} still waiting)."
        )
        self._spawn_child_task(**next_spawn_kwargs)

    # How close a spawned subtask's description has to be to one of the
    # prompt's own examples before it is treated as copied rather than
    # written. Deliberately high: this rejects an entire batch, so it has to
    # mean "this is the example text", not "this is on a similar topic".
    _EXEMPLAR_MATCH_CUTOFF = 0.85
    # How much longer than the exemplar a containing description may be and
    # still count as a copy. Some exemplars are short and generic ("Select
    # the base material"), and a real subtask legitimately extends one of
    # those into project vocabulary ("Select the base material for the
    # turbine blade given the 1450C floor") -- that is a decomposition, not
    # a copy, and rejecting it would throw away a valid plan. A copy that
    # merely has a few words of connective tissue bolted on stays well
    # inside this bound; an elaboration that adds real content does not.
    _EXEMPLAR_PADDING_ALLOWANCE = 1.5

    @staticmethod
    def _normalize_for_exemplar_match(text: str) -> str:
        """Casefolded, punctuation-free, whitespace-collapsed view of a task
        description -- so "Pick a color palette for the newsletter template."
        and "pick a colour  palette for the newsletter template" compare as
        the same string modulo the ratio below."""
        return normalize_words(text)

    @classmethod
    def _matching_exemplar(cls, description: str) -> Optional[str]:
        """
        The prompt exemplar this subtask description was copied from, or None.

        The exemplars come from agent_node.EXEMPLAR_SUBTASK_DESCRIPTIONS,
        which is the same tuple the prompt's SPAWN examples are built from --
        that shared constant is the whole point, since a hardcoded copy here
        would silently stop matching the moment someone reworded an example.
        """
        normalized = cls._normalize_for_exemplar_match(description)
        if not normalized:
            return None
        for exemplar in EXEMPLAR_SUBTASK_DESCRIPTIONS:
            exemplar_norm = cls._normalize_for_exemplar_match(exemplar)
            if not exemplar_norm:
                continue
            if normalized == exemplar_norm:
                return exemplar
            # Containment either way: the model routinely emits the example
            # sentence with a few words of its own bolted on either end,
            # which drops the plain ratio below the cutoff while leaving the
            # copied text fully intact. Bounded by the padding allowance
            # above so that genuinely extending a short generic exemplar
            # into this project's vocabulary is not read as copying it.
            if normalized in exemplar_norm:
                return exemplar
            if (exemplar_norm in normalized
                    and len(normalized) <= len(exemplar_norm) * cls._EXEMPLAR_PADDING_ALLOWANCE + 12):
                return exemplar
            if difflib.SequenceMatcher(
                None, normalized, exemplar_norm
            ).ratio() >= cls._EXEMPLAR_MATCH_CUTOFF:
                return exemplar
        return None

    def _has_no_requirement_overlap(self, description: str,
                                    parent_task: Optional[str] = None) -> bool:
        """
        True when this description shares no vocabulary at all -- neither a
        significant word nor a stem -- with anything this project is about:
        its requirements, its goal, the user's own request, or the task of
        the decomposer that spawned it.

        This used to read the requirements list alone. The phaser writes
        those as abstract, domain-free sentences ("The selection process
        requires defined criteria or voting method"), so a subtask worded in
        the goal's own nouns ("Count the votes for each book candidate")
        shared nothing with them and the whole batch was thrown away -- six
        times in one run, on plainly on-topic plans. A copied exemplar (a
        newsletter palette in a turbine colony) shares nothing with the goal
        or the parent task either, so the wider corpus still catches it.

        Inert without requirements, as before: no phaser spec means no
        trustworthy picture of the project to judge a subtask against.
        """
        spec = self.spec or {}
        requirements = spec.get("requirement", [])
        if not requirements:
            return False
        task_words = self._significant_words(description)
        if not task_words:
            return False
        corpus_words = set()
        for text in (*requirements, spec.get("goal"), spec.get("raw_text"), parent_task):
            if isinstance(text, str):
                corpus_words |= self._significant_words(text)
        if task_words & corpus_words:
            return False
        return not (self._stems(task_words) & self._stems(corpus_words))

    @property
    def _software_framing_guard_active(self) -> bool:
        """
        True when the user's own request never asked for software, so a
        subtask or answer shaped like code is the model's reflex rather than
        the job. Read from raw_text only, never the phaser's goal: the goal
        is model output and can itself be where "Generate a script to..."
        came from. No spec (or no raw text) leaves the guard off.
        """
        raw_text = (self.spec or {}).get("raw_text")
        return bool(raw_text) and not asks_for_software(raw_text)

    def _software_framing_reason(self, text) -> Optional[str]:
        """Why this text is a software deliverable in disguise, or None
        (always None when the project really is about software)."""
        if not self._software_framing_guard_active:
            return None
        return software_artifact_reason(text)

    def _plain_subtask_description(self, description: str) -> str:
        """
        A SPAWNed subtask description with code-coded words reworded
        ("Implement the voting logic" -> "Work out the voting rules").

        Reworded rather than rejected: the words are ordinary English and
        the plan behind them is usually fine, but the child reads its task
        text (and its own children inherit it) as a request for code -- the
        suspected path by which one early "implement" spreads through a
        colony. Rejecting would cost a respawn per word.

        Counted whether or not the "reword" lever is on, so a baseline run
        still shows how often subtasks arrive in code-coded wording.
        """
        if not self._software_framing_guard_active or not isinstance(description, str):
            return description
        plain, replaced = plain_register(description)
        if not replaced:
            return description
        self.colony.record_verdict("software_framing_subtask_codeword")
        if "reword" not in self.framing_levers:
            print(f"  [software-framing] subtask uses code-coded wording {replaced} "
                  f"(reword lever off): {description[:80]!r}")
            return description
        print(f"  [software-framing] reworded subtask {replaced}: "
              f"{description[:80]!r} -> {plain[:80]!r}")
        self.colony.record_verdict("software_framing_subtask_reworded")
        return plain

    def _screen_derived_subtask_batch(self, event: Event, descriptions: list,
                                      originals: Optional[list] = None):
        """
        Returns (proceed, dropped): proceed is False if the whole batch was
        rejected and the spawner rerouted to a respawn; otherwise dropped is
        the indices of individual subtasks to leave out.

        `originals` is each description as the model wrote it, before the
        software-framing reword -- checked alongside the reworded text so the
        reword can neither hide a copied exemplar nor cost a subtask the
        overlap it had.

        A subtask that merely shares no vocabulary with the project is
        dropped on its own, and the rest of the batch proceeds: that check
        is a vocabulary heuristic, and one badly worded subtask is not
        evidence the whole plan was copied. Only when EVERY subtask in the
        batch fails it is the batch rejected.

        A near-copy of a prompt exemplar, or a software-shaped subtask on a
        non-software project, still rejects the WHOLE batch, in code rather
        than in the prompt. Three separate rewordings of the
        SPAWN examples have failed to stop a decomposer under load from
        lifting an example's task text straight into its own payload -- the
        colony then spends real agents, real energy and real wall-clock time
        building a newsletter template for a turbine-blade problem, and every
        downstream agent that depends on that subtask dies on the
        contradiction. A batch with one copied subtask is a batch the model
        was pattern-matching rather than decomposing, so the plan is thrown
        away whole and the decomposer is respawned to produce a real one.
        """
        spawner_id = event.from_agent
        spawner = self.colony.get_agent(spawner_id) if spawner_id else None
        # Orchestrator-originated spawns (bootstrap, overflow drains) are
        # ours, not a model's -- nothing to reject and nobody to respawn.
        if spawner is None:
            return True, []
        if originals is None or len(originals) != len(descriptions):
            originals = list(descriptions)

        offending = None
        advice = (
            "The examples in your instructions show "
            "the FORMAT only -- their task text belongs to a different, "
            "made-up project and must never appear in your payload. Every "
            "subtask you spawn must be a piece of THIS project's task, "
            "worded in this project's own vocabulary."
        )
        agent_offending = None
        rejection = ("REJECTED: decomposition copied its subtasks from "
                     "the prompt's worked examples instead of "
                     "decomposing the assigned task.")
        # Software framing. Every code-shaped subtask in the batch is counted
        # BEFORE deciding anything, so the count means the same thing with
        # the "block" lever on (batch rejected at the first one) or off.
        software_hits = []
        for description in descriptions:
            software_reason = self._software_framing_reason(description)
            if software_reason is not None:
                self.colony.record_verdict("software_framing_spawn_detected")
                software_hits.append((description, software_reason))
        if software_hits and "block" not in self.framing_levers:
            for description, software_reason in software_hits:
                print(f"  [software-framing] subtask {software_reason} "
                      f"(block lever off): {description[:80]!r}")
        elif software_hits:
            description, software_reason = software_hits[0]
            offending = (
                f"subtask '{description[:80]}' {software_reason}, but "
                f"this project never asked for software"
            )
            # The spawner's respawn reads fail_reason as ghost context, so
            # it must not quote the file/function names back to it.
            agent_offending = (
                f"a subtask {_without_quoted_example(software_reason)}, but "
                f"this project never asked for software"
            )
            advice = (
                "Word every subtask as the decision, list, plan or "
                "piece of writing it really is, in plain sentences -- "
                "no file names, function names or code."
            )
            rejection = ("REJECTED: decomposition turned a non-software "
                         "task into software subtasks.")
            self.colony.record_verdict("software_framing_spawn_rejected")

        off_topic = []
        parent_task = getattr(spawner, "task", None)
        for i, description in enumerate([] if offending is not None else descriptions):
            original = originals[i]
            exemplar = (self._matching_exemplar(original)
                        or self._matching_exemplar(description))
            if exemplar is not None:
                offending = (
                    f"subtask '{description[:80]}' is a near-copy of the "
                    f"prompt's own worked example '{exemplar}', which is "
                    f"illustration text, not work belonging to this project"
                )
                break
            if self._has_no_requirement_overlap(f"{original} {description}",
                                                parent_task=parent_task):
                off_topic.append(i)

        if offending is None and off_topic and len(off_topic) == len(descriptions):
            offending = (
                f"subtask '{descriptions[off_topic[0]][:80]}' shares no "
                f"vocabulary at all with this project's goal, requirements "
                f"or the task being decomposed -- and neither does any other "
                f"subtask in the batch, which is what a plan copied from an "
                f"example looks like"
            )

        if offending is None:
            return True, off_topic

        print(f"REJECT (derived subtask) on {spawner_id}: {offending} -- "
              f"discarding all {len(descriptions)} subtask(s) in this batch "
              f"and respawning the decomposer.")

        spawner.fail_reason = (
            "Previous attempt was REJECTED and none of its subtasks were "
            f"created: {agent_offending or offending}. {advice}"
        )
        self.messenger.push_event(
            "failure_request",
            spawner_id,
            {
                "task_id": spawner.task_id,
                "role": spawner.role,
                "parent_id": spawner.parent_id,
                "result": rejection,
                # No agent executed anything: handle_failure respawns this
                # without spending one of the task's MAX_TASK_ATTEMPTS.
                "derived_subtask_rejection": True,
            },
        )
        return False, []

    def _drop_off_topic_subtasks(self, parent_id: str, dropped: list) -> None:
        """Tell the decomposer which of its subtasks were left out as sharing
        no vocabulary with the project, so its roll-up says what was not
        covered instead of presenting a partial plan as the whole one.
        Delivered and billed like _tell_parent_about_dropped_subtasks."""
        for _ in dropped:
            self.colony.record_verdict("derived_subtask_dropped")
        names = "; ".join(f'"{d[:80]}"' for d in dropped)
        print(f"  [derived subtask] {parent_id}: dropped {len(dropped)} subtask(s) "
              f"sharing no vocabulary with this project; the rest of the batch "
              f"proceeds: {names}")
        live_parent = self.live_agents.get(parent_id)
        if live_parent is None:
            return
        note = (f"\n[REVIEW] These subtasks were NOT started because they do "
                f"not mention anything from this project: {names}. Your final "
                f"REPORT must say they were not covered.\n")
        live_parent.thought_process += note
        self.colony.debit_energy(parent_id, max(1, len(note) // 100), category="injection")

    # ------------------------------------------------------------------
    # PATCH 12 -- the goal-drift gate
    # ------------------------------------------------------------------

    def _embed_description(self, description: str):
        """This task description as a vector, or None. The one place a
        subtask description is embedded: the gate calls it at batch-screen
        time and caches the result for _spawn_child_task, so a subtask that
        survives the gate is not encoded a second time."""
        if self.embed_model is None:
            return None
        try:
            return self.embed_model.encode(description, convert_to_numpy=True)
        except Exception as e:
            print(f"Warning: failed to embed task description: {e}")
            return None

    def _goal_drift_gate_threshold(self, batch_scores) -> Optional[float]:
        """The cutoff this batch is judged against, or None when the run has
        not seen enough measured goal-cosines to have a meaningful maximum.

        The run's observed max INCLUDES this batch's own scores. A batch that
        is itself the highest-scoring thing the run has produced must not be
        gated against a ceiling it just raised -- computing the max first and
        comparing after is what makes the gate scale-free rather than
        retroactive.
        """
        observed = [s["goal_drift"] for s in self.goal_drift_samples
                    if s.get("goal_drift") is not None]
        observed += [d for d in batch_scores if d is not None]
        if len(observed) < GOAL_DRIFT_GATE_MIN_SAMPLES:
            return None
        ceiling = max(observed)
        if ceiling <= 0.0:
            return None
        return GOAL_DRIFT_GATE_FRACTION * ceiling

    def _screen_goal_drift_batch(self, prepared) -> list:
        """PATCH 12. Score every subtask in a prepared SPAWN batch against the
        colony goal and return the indices of the ones too far from it to
        start. Nothing is embedded twice: the vectors computed here are handed
        to _spawn_child_task through _gate_description_embeddings.

        `prepared` is handle_spawn's list of (sub, description, task_id).

        CUMULATIVE distance only. Run 5 settled the per-hop question: the
        gradual cases are invisible to it (task_2239fb25 sat at goal 0.439
        behind a perfectly ordinary parent hop of 0.638) and it false-alarms
        on fine ones (task_aeab53e4, goal 0.503, arrived on a 0.256 hop and
        was a legitimate piece of the project). parent_drift stays measured
        and reported; nothing is gated on it.

        PER-SUBTASK DROP, never a batch reject. Same shape as the
        derived-subtask guard's off-topic drop, for the same reason: one
        subtask that wandered is not evidence the plan was copied, and the
        siblings around it were fine. The decomposer is told what was left
        out so its roll-up does not present a partial plan as the whole one.

        A batch where EVERY subtask trips is left ALONE and counted. Dropping
        all of them starves the spawner of children, which is the DIE path,
        not a drop -- and a whole batch under the threshold is far more likely
        to mean the ceiling is wrong for this corpus than that a decomposer
        produced nothing usable. That case is the one worth looking at in the
        ledger before acting on it, which is what the counter is for. It also
        means a lone subtask is never gated, since a batch of one is always
        all-or-nothing.
        """
        scores = []
        for _, description, task_id in prepared:
            embedding = self._embed_description(description)
            self._gate_description_embeddings[task_id] = embedding
            scores.append(_cosine(embedding, self.colony.goal_embedding))

        threshold = self._goal_drift_gate_threshold(scores)
        if threshold is None:
            return []

        tripped = [i for i, drift in enumerate(scores)
                   if drift is not None and drift < threshold]
        if not tripped:
            return []
        if len(tripped) == len(prepared):
            self.colony.record_verdict("goal_drift_gate_whole_batch_spared")
            print(f"  [goal-drift gate] every subtask in this batch of "
                  f"{len(prepared)} scores below {threshold:.3f} "
                  f"(={GOAL_DRIFT_GATE_FRACTION:g} x the run's observed max) -- "
                  f"spared and counted rather than dropped, since dropping all "
                  f"of them leaves the decomposer with no children at all.")
            return []

        for i in tripped:
            self.colony.record_verdict("goal_drift_gate_dropped")
            print(f"  [goal-drift gate] "
                  f"{'DROP' if GOAL_DRIFT_GATE_ACTS else 'WOULD DROP'} "
                  f"{prepared[i][2]} "
                  f"goal={scores[i]:.3f} < {threshold:.3f} "
                  f"(={GOAL_DRIFT_GATE_FRACTION:g} x observed max): "
                  f"{str(prepared[i][1])[:70]!r}")
            # The distribution is the ledger's, gated or not. A dropped
            # subtask never reaches _measure_spawn_drift, so its score is
            # recorded here or nowhere -- and a ledger that silently omits
            # exactly the tasks the gate acted on is a ledger that cannot be
            # used to check whether the threshold was right. parent_drift is
            # None rather than measured: the gate does not read it, and
            # computing it would mean resolving a parent task for a node that
            # is never going to exist.
            self._flag_artifact_shaped_task(
                prepared[i][2], prepared[i][1],
                drift_dropped=GOAL_DRIFT_GATE_ACTS)
            if GOAL_DRIFT_GATE_ACTS:
                clause_drift, clause_index = self._clause_drift(
                    self._gate_description_embeddings.get(prepared[i][2]))
                self.goal_drift_samples.append({
                    "task_id": prepared[i][2],
                    "description": prepared[i][1],
                    "goal_drift": scores[i],
                    "parent_drift": None,
                    "clause_drift": clause_drift,
                    "clause_index": clause_index,
                    "gated": True,
                })
            else:
                # Measure-only: this subtask IS going to spawn, so
                # _measure_spawn_drift will file its ledger entry in a
                # moment -- with a real parent_drift, which this path
                # cannot compute. Appending here too would double-count it
                # and break the run's own n. The task_id is remembered
                # instead, and _measure_spawn_drift marks that entry
                # "gated", so the ledger still shows exactly which subtasks
                # the gate would have refused.
                self._gate_would_drop.add(prepared[i][2])
        if not GOAL_DRIFT_GATE_ACTS:
            print(f"  [goal-drift gate] measure-only "
                  f"(GOAL_DRIFT_GATE_ACTS=False): {len(tripped)} subtask(s) "
                  f"scored below threshold and were started anyway.")
            return []
        return tripped

    def _drop_drifted_subtasks(self, parent_id: str, dropped: list) -> None:
        """Tell the decomposer which of its subtasks were left out as too far
        from the project goal. Delivered and billed like
        _drop_off_topic_subtasks, whose wording this deliberately does not
        reuse: "shares no vocabulary with this project" and "is too far from
        what this project is for" are different findings, and a decomposer
        handed the wrong one rewrites the wrong thing."""
        names = "; ".join(f'"{d[:80]}"' for d in dropped)
        print(f"  [goal-drift gate] {parent_id}: dropped {len(dropped)} subtask(s) "
              f"too far from the project goal; the rest of the batch proceeds: "
              f"{names}")
        live_parent = self.live_agents.get(parent_id)
        if live_parent is None:
            return
        note = (f"\n[REVIEW] These subtasks were NOT started because they are "
                f"too far from what this project is actually for: {names}. Do "
                f"not re-spawn them or invent deliverables nobody asked for. "
                f"Your final REPORT must say they were not covered.\n")
        live_parent.thought_process += note
        self.colony.debit_energy(parent_id, max(1, len(note) // 100), category="injection")

    # ------------------------------------------------------------------
    # PATCH 15 -- artifact manufacturing (measure-only)
    # ------------------------------------------------------------------

    @property
    def _artifact_guard_active(self) -> bool:
        """True when the user's own request never named a document or
        deliverable, so a subtask asking for one is the model reaching for
        something to produce rather than the job.

        Read from raw_text only, never the phaser's goal -- the same rule and
        the same reason as _software_framing_guard_active: the goal is model
        output and can itself be where "Prepare a summary deck..." came from.
        No spec (or no raw text) leaves the guard off.
        """
        raw_text = (self.spec or {}).get("raw_text")
        return bool(raw_text) and not asks_for_artifact(raw_text)

    def _flag_artifact_shaped_task(self, task_id, description,
                                   drift_dropped: bool = False):
        """
        PATCH 15 (measure-only). Ask the artifact question of a SUBTASK.

        Counted and logged only. Nothing is dropped, reworded or respawned on
        it. PATCH 12's drift gate went live in the same run, and two new gates
        firing at once leaves no way to attribute what changed -- so this one
        watches while that one acts.

        `drift_dropped` records whether PATCH 12 already dropped this subtask,
        which is the number worth reading: the two signals are meant to be
        complementary, and the ledger shows the overlap directly instead of
        making it something to reconstruct from the log by hand. Called from
        the drift gate for the subtasks it drops and from _spawn_child_task
        for everything else, so the two sets are disjoint and nothing is
        counted twice.
        """
        if not self._artifact_guard_active:
            return
        hits = artifact_request_words(description)
        if not hits:
            return
        self.colony.record_verdict("artifact_shaped_task_detected")
        if drift_dropped:
            self.colony.record_verdict("artifact_shaped_task_also_drift_dropped")
        print(f"  [artifact-shaped task] {task_id} asks for {hits} on a project "
              f"that never did (measure-only, not blocked"
              f"{'; already dropped by the drift gate' if drift_dropped else ''}): "
              f"{str(description)[:70]!r}")

    # ------------------------------------------------------------------
    # PATCH 16 -- figures that do not survive a retry (measure-only)
    # ------------------------------------------------------------------

    def _request_text(self):
        """PATCH 29. The user's request as every agent prompt shows it:
        spec["raw_text"] with its whitespace collapsed, or None without a
        spec. Not deduped or capped like the goal referent -- it is the
        person's own words, and the phaser already caps it at 3000 chars."""
        raw_text = (self.spec or {}).get("raw_text")
        if not raw_text or not str(raw_text).strip():
            return None
        return " ".join(str(raw_text).split())

    def _flag_unsupplied_figures(self, task_id, description) -> None:
        """PATCH 22 (measure-only). A subtask description that states a
        figure the user's request does not.

        Run 8's root decomposer spawned "Allocate Rs. 9,600 total annual
        fees" on a request that said Rs. 9,000 -- copied from a phaser
        constraint, and from there into every descendant. Checked against
        raw_text only: the goal and constraints are model output and may be
        the very thing that is wrong. Only when the request supplies data;
        on a data-free problem PATCH 21 counts the answers instead.

        A figure here is not necessarily invented -- "450 per applicant" is
        9000/20 -- so this counts and lists, and does nothing else.
        """
        self.spawned_subtask_count += 1
        spec = self.spec or {}
        if not spec.get("supplies_data"):
            return
        extra, _ = figure_fidelity(spec.get("raw_text", ""), description,
                                   require_all=False)
        if not extra:
            return
        self.colony.record_verdict("subtask_states_unsupplied_figure")
        self.unsupplied_figure_subtasks.append((task_id, extra, description))
        print(f"  [data fidelity] {task_id} states figures the request does "
              f"not: {extra} (measure-only): {str(description)[:80]!r}")

    def _print_fidelity_report(self) -> None:
        """PATCH 22 ledger section: what the phaser did to the user's data,
        and how many spawned subtasks stated figures the user never gave."""
        spec = self.spec or {}
        print(chr(10) + "  PHASER DATA FIDELITY (PATCH 22)")
        goal = spec.get("goal_fidelity")
        if not goal:
            print("    goal : not checked (older spec, or the phaser fell back)")
        else:
            print(f"    goal : {goal['outcome']}  "
                  f"({len(goal['attempts'])} draw(s))")
            for i, attempt in enumerate(goal["attempts"], 1):
                problems = [f"figures not in request {attempt['extra']}" if attempt["extra"] else "",
                            f"request figures missing {attempt['missing']}" if attempt["missing"] else "",
                            f"{len(attempt['uncovered'])} part(s) not covered" if attempt["uncovered"] else ""]
                problems = "; ".join(p for p in problems if p) or "ok"
                print(f"      draw {i}: {problems}")
                for ask in attempt["uncovered"]:
                    print(f"        missing part: {ask[:90]!r}")
        constraints = spec.get("constraint_fidelity")
        if constraints:
            print(f"    constraints : {len(constraints['dropped'])} dropped for "
                  f"figures not in the request"
                  + (" (after one re-extraction)" if constraints["retried"] else ""))
            for item in constraints["dropped"]:
                print(f"      {item['extra']}  {item['constraint'][:80]!r}")
            # PATCH 30. .get: a spec from before the repair has no key.
            repaired = constraints.get("repaired", [])
            print(f"    constraints : {len(repaired)} repaired to the request's figure")
            for item in repaired:
                pairs = ", ".join(f"{a} -> {b}" for a, b in item["repairs"])
                print(f"      [{pairs}]  {item['repaired'][:80]!r}")
            if "extracted" in constraints:
                print(f"    constraints : {constraints['extracted']} extracted, "
                      f"{len(spec.get('requirement') or [])} kept (PATCH 31: the "
                      f"budget multiplier counts the extracted ones)")
        counts = self.request_prompt_counts
        print(f"    agents with the request verbatim in their prompt (PATCH 29) : "
              f"{counts['with_request']} of {counts['agents']}"
              + (f"  (request figures: {spec.get('supplied_figures')})"
                 if spec.get("supplied_figures") else ""))
        if spec.get("supplies_data"):
            hits = self.unsupplied_figure_subtasks
            print(f"    spawned subtasks stating figures the request does not : "
                  f"{len(hits)} of {self.spawned_subtask_count}  (measure-only)")
            for task_id, extra, description in hits:
                print(f"      {task_id}  {extra}  {str(description)[:60]!r}")

    def _promoted_reports_asserting_data(self):
        """PATCH 21 (measure-only). Every promoted REPORT -- root included --
        as (task_id, figures, identifiers), and the total promoted count.
        Read from the task graph at ledger time, so it is the text that was
        actually promoted and threaded onward, not a draft."""
        hits, total = [], 0
        for task_id, task in self.task_graph.tasks.items():
            if getattr(task, "status", None) != 2 or getattr(task, "result", None) is None:
                continue
            total += 1
            text = str(task.result)
            figs = asserted_figures(text)
            ids = asserted_identifiers(text)
            if figs or ids:
                hits.append((task_id, figs, ids))
        return hits, total

    def _print_data_free_report(self) -> None:
        """PATCH 21 ledger rows. Measure-only: nothing is rejected on it.

        When the request supplies no figure and no identifier, every
        specific number or ID in a promoted REPORT came from somewhere other
        than the user. Each hit's own figures are printed because some will
        be DERIVED from words ("more than half" -> 50%) rather than made up,
        and only a reader can tell those apart -- see supplied_data."""
        spec = self.spec or {}
        if "supplies_data" in spec:
            supplies = bool(spec.get("supplies_data"))
            source = "phaser"
        else:
            figs, ids = supplied_data(spec.get("raw_text", ""))
            supplies = bool(figs or ids)
            source = "raw_text, recomputed"
        print(chr(10) + "  DATA-FREE PROBLEM (PATCH 21, measure-only)")
        print(f"    problem supplies data : {'yes' if supplies else 'no'}  ({source})")
        if supplies:
            # PATCH 23 (measure-only). The request supplies data, so each
            # promoted REPORT's figures -- distinct values per REPORT --
            # are sorted into grounded / derivable / near-miss / neither
            # against raw_text. The same set PATCH 21 walks (status 2,
            # result not None, root and roll-ups included). Two passes: the
            # run's near-miss values feed the second one's trace annotation.
            from text_utils import (classify_figures, compact_figure,
                                    FIGURE_GROUNDED, FIGURE_DERIVABLE,
                                    FIGURE_NEAR_MISS, FIGURE_NEITHER)
            request = spec.get("raw_text", "")
            promoted = [(task_id, str(task.result))
                        for task_id, task in self.task_graph.tasks.items()
                        if getattr(task, "status", None) == 2
                        and getattr(task, "result", None) is not None]
            near = sorted({e["value"] for _, text in promoted
                           for e in classify_figures(request, text)
                           if e["class"] == FIGURE_NEAR_MISS})
            classes = (FIGURE_GROUNDED, FIGURE_DERIVABLE, FIGURE_NEAR_MISS,
                       FIGURE_NEITHER)
            counts = {c: 0 for c in classes}
            rows = {c: [] for c in classes}
            distinct = {c: set() for c in classes}
            with_figures = 0
            for task_id, text in promoted:
                entries = classify_figures(request, text, near_misses=near)
                with_figures += bool(entries)
                for e in entries:
                    counts[e["class"]] += 1
                    distinct[e["class"]].add(e["value"])
                    shown = compact_figure(e["value"])
                    if e["class"] == FIGURE_DERIVABLE:
                        rows[e["class"]].append(f"{task_id}  {e['detail']}")
                    elif e["class"] != FIGURE_GROUNDED:
                        rows[e["class"]].append(
                            f"{task_id}  {shown}"
                            + (f"  {e['detail']}" if e["detail"] else ""))
            print(f"    promoted REPORTs with figures : {with_figures} of {len(promoted)}")
            for c in classes:
                label = f"figures {c}".ljust(18)
                note = "   (not counting near-misses)" if c == FIGURE_NEITHER else ""
                print(f"    {label} : {counts[c]}{note}")
                for row in rows[c]:
                    print(f"        {row}")
            print("    run-level distinct values:")
            print("        " + "  ".join(
                f"{c} [{', '.join(compact_figure(v) for v in sorted(distinct[c]))}]"
                for c in classes))
            return
        hits, total = self._promoted_reports_asserting_data()
        print(f"    promoted REPORTs asserting figures or IDs anyway : "
              f"{len(hits)} of {total}")
        for task_id, figs, ids in hits:
            shown = (figs + ids)[:8]
            more = len(figs) + len(ids) - len(shown)
            print(f"      {task_id}  {shown}" + (f" +{more} more" if more > 0 else ""))

    def _print_final_answer_figures(self, final_answer) -> None:
        """PATCH 23 (measure-only). The FINAL ANSWER's figures, classified
        separately from the REPORTs: the text as shipped, after the exit
        guard and the partial banner. Called from terminate() on both exit
        paths -- not from _print_data_free_report, which runs before the
        partial-path synthesis exists. Silent on a data-free problem."""
        spec = self.spec or {}
        if "supplies_data" in spec:
            supplies = bool(spec.get("supplies_data"))
        else:
            figs, ids = supplied_data(spec.get("raw_text", ""))
            supplies = bool(figs or ids)
        if not supplies:
            return
        from text_utils import (classify_figures, compact_figure,
                                FIGURE_GROUNDED, FIGURE_DERIVABLE,
                                FIGURE_NEAR_MISS, FIGURE_NEITHER)
        print(chr(10) + "  FINAL ANSWER figures (PATCH 23):")
        if not isinstance(final_answer, str):
            print("      not a text answer, not classified")
            return
        distinct = {c: set() for c in (FIGURE_GROUNDED, FIGURE_DERIVABLE,
                                       FIGURE_NEAR_MISS, FIGURE_NEITHER)}
        for e in classify_figures(spec.get("raw_text", ""), final_answer):
            distinct[e["class"]].add(e["value"])
        print("      " + "  ".join(
            f"{c} [{', '.join(compact_figure(v) for v in sorted(vals))}]"
            for c, vals in distinct.items()))

    def _check_figure_divergence(self, task_id, agent_id, result):
        """
        PATCH 16 (measure-only). Compare this REPORT's figures against the
        last REPORT recorded for the SAME TASK, and log when almost none of
        them carried over.

        Keyed on the task, not the agent, and run at every REPORT rather than
        only at respawn -- which is what makes it see run 5's case at all.
        agent_ce16fd80 was not respawned between its three contradictory
        tallies for task_91dcc48e; it was WARNed twice by tier 2 and answered
        again as the same agent. A respawn-only check would have watched the
        agent that mattered and missed every comparison it made.

        Reads the UNTRIMMED payload, like the exemplar-echo and
        software-framing checks beside it: the 3-sentence report-trim keeps
        the answer and drops the tail, and run 5 put half of attempt 1's
        figures in the tail.

        Never fails a task, never respawns, never rejects. The point of this
        round is the count.
        """
        if not isinstance(result, str) or not task_id:
            return
        previous = self._task_last_report.get(task_id)
        self._task_last_report[task_id] = (agent_id, result)
        if previous is None:
            return
        divergence = figure_divergence(previous[1], result)
        if divergence is None:
            return
        self.colony.record_verdict("figure_divergence_detected")
        ratio = divergence["ratio"]
        print(f"  [figure divergence] {task_id}: this attempt ({agent_id}) "
              f"shares {divergence['overlap']:.0%} of its figures with the "
              f"previous one ({previous[0]}) -- same task, same inputs "
              f"(measure-only, nothing failed).")
        print(f"      was : {divergence['previous']}")
        print(f"      now : {divergence['current']}")
        print(f"      shared={divergence['shared'] or 'none'}"
              + (f"  largest-figure ratio={ratio:.1f}x" if ratio else ""))

    def _reject_spawn_from_non_decomposer(self, event: Event) -> bool:
        """
        Returns True if this SPAWN may proceed, False if it was rejected and
        rerouted. See handle_spawn for why this is enforced in code.
        """
        spawner_id = event.from_agent
        spawner = self.colony.get_agent(spawner_id) if spawner_id else None
        # Orchestrator-originated spawns (bootstrap, overflow drains) have no
        # agent record to check -- those are ours, not a model's request.
        if spawner is None or spawner.role == "decomposer":
            return True

        print(f"REJECT (structural) on {spawner_id}: a '{spawner.role}' agent "
              f"requested a SPAWN, which only a decomposer may do -- "
              f"rerouting to the TASK TOO LARGE respawn path.")

        spawner.fail_reason = (
            "Previous attempt was REJECTED: you issued a SPAWN, but only a "
            "decomposer may spawn subtasks. If the task is genuinely too big "
            "for one agent, DIE with a payload starting 'TASK TOO LARGE:' and "
            "it will be re-planned by a decomposer; otherwise do the work "
            "yourself with THINK/TOOL and REPORT the result."
        )
        self.messenger.push_event(
            "failure_request",
            spawner_id,
            {
                "task_id": spawner.task_id,
                "role": spawner.role,
                "parent_id": spawner.parent_id,
                "result": ("TASK TOO LARGE: the assigned agent tried to "
                           "decompose this task itself instead of executing "
                           "it."),
            },
        )
        return False

    def handle_spawn(self, event: Event):
        """Triggered when an existing agent requests sub-agents (children)."""
        unstarted_before = len(self.unstarted_tasks)
        if self._handle_spawn_request(event):
            self._release_spawner_if_nothing_started(
                event.payload.get("parent_id"),
                out_of_energy=len(self.unstarted_tasks) > unstarted_before,
            )

    def _release_spawner_if_nothing_started(self, parent_id: Optional[str],
                                            out_of_energy: bool = False):
        """
        Agent.request_spawn sets awaiting="children" as soon as a SPAWN has a
        plausible shape, before the orchestrator has created anything. If
        nothing was actually started -- every batch entry malformed and
        skipped, a single SPAWN with no task text, or no energy to start any
        child -- no child exists to ever send a parent_notification, and
        the deadlock watchdog skips awaiting agents, so the decomposer hung
        until the budget ran out. Hand it back with the reason instead, and
        charge the cycle: request_spawn's `awaiting` meant run() did not,
        so a decomposer repeating an unusable batch would otherwise loop
        outside the cycle cap.
        """
        live_parent = self.live_agents.get(parent_id) if parent_id else None
        if live_parent is None or live_parent.awaiting != "children":
            return
        if self._open_child_task_ids(parent_id):
            return

        parent_node = self.colony.get_agent(parent_id)
        parent_task_id = getattr(parent_node, "task_id", None)
        tally = self._direct_child_tally(parent_task_id)
        has_completed_child = self._has_completed_direct_child(parent_task_id)
        was_capped = getattr(live_parent, "cycles_capped", False)

        # Two cases where releasing it to try again cannot help, so it goes
        # down the normal DIE path (failure -> respawn or abandon) instead:
        #  * it was already at the cycle cap. Its SPAWN was its last allowed
        #    action; releasing it again let a batch that looks valid but
        #    starts nothing (e.g. {"subtasks": ["junk"]}) repeat forever
        #    past the cap.
        #  * no energy, and not one of its DIRECT subtasks completed. Another
        #    SPAWN is refused the same way, and there is nothing to REPORT:
        #    subtasks that were all abandoned or never started leave only
        #    [ABANDONED]/[NOT STARTED] markers, and a REPORT on those is the
        #    decomposer describing work that never happened. Run 3's root did
        #    exactly that with 0 of 3 subtasks completed, and was accepted.
        #    Counted from TaskGraph.direct_children -- not has_spawned (true
        #    once any child existed, whatever became of it), and not
        #    AgentNode.children (grandchildren adopted from a retired child
        #    decomposer land there).
        # `awaiting` stays set so it is not ticked before the failure event
        # is routed on the next tick.
        if was_capped or (out_of_energy and not has_completed_child):
            if not out_of_energy:
                why = "its final allowed SPAWN started no subtasks"
            elif tally["failed"]:
                why = (f"no energy left to start its subtasks, and none of its "
                       f"{tally['failed']} earlier subtask(s) completed")
            else:
                why = "no energy left to start its subtasks"
            if out_of_energy and not has_completed_child:
                self.colony.record_verdict("spawn_refused_no_completed_child")
            print(f"  [handle_spawn] {parent_id}'s SPAWN started no subtasks "
                  f"and it cannot usefully retry ({why}) -- routing to failure.")
            if parent_node is not None:
                parent_node.fail_reason = f"SPAWN Action Failed: {why}."
            self.messenger.push_event(
                "failure_request",
                parent_id,
                {
                    "task_id": live_parent.task_id,
                    "role": live_parent.role,
                    "parent_id": live_parent.parent_id,
                    "result": f"Cycle cap / spawn failure: {why}.",
                },
            )
            return

        print(f"  [handle_spawn] {parent_id}'s SPAWN started no subtasks -- "
              f"releasing it from awaiting instead of leaving it to hang.")
        if out_of_energy:
            # Asking for a corrected batch would only be refused again. Only
            # reached with at least one DIRECT subtask completed (status 2):
            # every out-of-energy case without one took the fail branch
            # above. So REPORT is on its menu and there is something real to
            # put in it; the YOUR SUBTASKS block in its prompt says which
            # subtasks completed and which did not.
            # Closed structurally, not just in the fail_reason: think() never
            # sees that text (it continues its KV cache) and decide() still
            # offered SPAWN, so the same batch came back at full think cost
            # every cycle until the cap. spawn_closed puts it on the capped
            # path now -- decide() only, REPORT/DIE.
            live_parent.fail_reason = (
                "SPAWN Action Failed: none of your requested subtasks were "
                "started because the colony is out of energy to start new "
                "agents. Do not SPAWN again. REPORT what your completed "
                "subtasks produced, and say plainly which did not complete."
            )
            if hasattr(live_parent, "spawn_closed"):
                live_parent.spawn_closed = True
                self.colony.record_verdict("spawn_closed_no_energy")
        else:
            live_parent.fail_reason = (
                "SPAWN Action Failed: none of your requested subtasks were "
                "started, so nothing is working on your behalf. Every subtask "
                "needs a non-empty \"task\" string; SPAWN a corrected batch, or "
                "DIE if the task cannot be decomposed."
            )
        live_parent.awaiting = None
        if hasattr(live_parent, "non_terminal_cycles"):
            live_parent.non_terminal_cycles += 1
            if live_parent.cycles_capped:
                self.colony.record_verdict("cycle_cap_reached")

    def _handle_spawn_request(self, event: Event) -> bool:
        """Body of handle_spawn. Returns False only when the request was
        rejected and the spawner already routed to a respawn, True in every
        other case (including when nothing ended up being spawned)."""
        payload = event.payload
        parent_id = payload.get("parent_id")

        # Only a decomposer may spawn. The prompt already says so in bold,
        # and _enforce_child_role only constrains WHICH role a child gets --
        # not whether this agent was entitled to ask for one. Prompt-only
        # enforcement of a structural rule does not hold: an executor that
        # decides its task is too big spawns instead of DIEing, and the
        # children it invents carry whatever text its generation happened to
        # be mid-sentence on. Converted here to the DIE / "TASK TOO LARGE"
        # path the prompt already describes, which handle_failure respawns
        # as a decomposer on the same task.
        if not self._reject_spawn_from_non_decomposer(event):
            return False

        subtasks = payload.get("subtasks")

        if subtasks:
            parent_node = self.colony.get_agent(parent_id) if parent_id else None
            cap = None
            if parent_node is not None and parent_node.role == "decomposer":
                cap = (
                    self.MAX_SUBTASKS_ROOT if parent_node.generation == 0
                    else self.MAX_SUBTASKS_NON_ROOT
                )

            prepared = []
            originals = []
            label_to_id = {}
            for i, sub in enumerate(subtasks):
                # FIX (#1, crash-the-whole-run bug): a malformed subtask can
                # arrive as a bare string (or any non-dict) instead of the
                # expected {"role":..., "task":...} object. Without this
                # guard, _normalize_subtask_keys's sub.items() call throws
                # and takes the entire tick (and therefore the whole colony
                # run) down with it.
                if not isinstance(sub, dict):
                    print(f"Warning: subtask entry at index {i} is not a dict ({sub!r}), skipping.")
                    continue
                sub = self._normalize_subtask_keys(sub)
                description = sub.get("task", sub.get("description", ""))
                if not description:
                    print("Warning: subtask entry missing 'task'/'description', skipping.")
                    continue
                original_description = description
                description = self._plain_subtask_description(description)
                sub["role"] = self._enforce_child_role(sub.get("role", "worker"), parent_id)
                task_id = self._generate_task_id()
                label = sub.get("label")
                label_to_id[str(i)] = task_id  # positional reference always works
                if label:
                    label_to_id[str(label)] = task_id
                prepared.append((sub, description, task_id))
                originals.append(original_description)

            # Derived-subtask guard, run on the whole prepared batch BEFORE
            # any child is created: one copied subtask invalidates the plan,
            # not just itself; a subtask that merely shares no vocabulary
            # with the project is left out on its own (see
            # _screen_derived_subtask_batch).
            proceed, off_topic = self._screen_derived_subtask_batch(
                event, [description for _, description, _ in prepared], originals
            )
            if not proceed:
                return False
            if off_topic:
                self._drop_off_topic_subtasks(
                    parent_id, [prepared[i][1] for i in off_topic])
                prepared = [entry for i, entry in enumerate(prepared)
                            if i not in set(off_topic)]

            # PATCH 12. Goal-drift gate, run on what is left of the batch and
            # BEFORE admission and the fan-out cap -- so a dropped subtask
            # never reserves budget, never becomes a queued overflow item,
            # and never becomes a dependency target that the loop below has
            # to unwire later. This is also the only point where an overflow
            # item can be gated at all: by the time it drains it has siblings
            # already depending on it.
            drifted = self._screen_goal_drift_batch(prepared)
            if drifted:
                self._drop_drifted_subtasks(
                    parent_id, [prepared[i][1] for i in drifted])
                for i in drifted:
                    self._gate_description_embeddings.pop(prepared[i][2], None)
                prepared = [entry for i, entry in enumerate(prepared)
                            if i not in set(drifted)]

            # Admission: the batch is cut to the largest k whose reservation
            # fits (queued overflow included -- it is reserved now, spawned
            # later). The rest is dropped, not bundled, and the decomposer is
            # told. Not even one fitting sends it down the DIE path.
            spawner = self.colony.get_agent(parent_id) if parent_id else None
            if spawner is not None and prepared:
                admitted = self._admit_spawn_batch(
                    spawner.task_id, [sub.get("role") for sub, _, _ in prepared])
                if admitted == 0:
                    self._refuse_spawn_for_budget(parent_id, spawner)
                    return False
                if admitted < len(prepared):
                    self._tell_parent_about_dropped_subtasks(
                        parent_id, [d for _, d, _ in prepared[admitted:]])
                    for _, _, dropped_id in prepared[admitted:]:
                        self._gate_description_embeddings.pop(dropped_id, None)
                    prepared = prepared[:admitted]
            admitted_ids = {task_id for _, _, task_id in prepared}

            # Fan-out cap: resolved against the FULL original batch's labels
            # above (so an item past the cap can still be a valid dependency
            # target/source), but only the first `cap` are spawned now --
            # the rest are queued (see pending_overflow's docstring in
            # __init__) instead of force-merged into one oversized child.
            if cap is not None and len(prepared) > cap:
                print(
                    f"  [handle_spawn] batch of {len(prepared)} subtasks "
                    f"from a generation-{parent_node.generation} decomposer "
                    f"exceeds the fan-out cap ({cap}) -- spawning the first "
                    f"{cap} now and queueing the remaining "
                    f"{len(prepared) - cap} to spawn one at a time as this "
                    f"decomposer's other children complete, instead of "
                    f"bundling them into one child that's reliably too "
                    f"large for a single agent."
                )

            for i, (sub, description, task_id) in enumerate(prepared):
                if "dependencies" in sub:
                    deps_raw = sub.get("dependencies") or []
                elif i > 0:
                    print(
                        f"  [handle_spawn] subtask '{description[:60]}...' "
                        f"omitted the required 'dependencies' field -- "
                        f"falling back to sequential-default (depends on "
                        f"the previous subtask in this batch)."
                    )
                    deps_raw = [str(i - 1)]
                else:
                    deps_raw = []

                resolved_deps = []
                for dep in deps_raw:
                    dep_id = label_to_id.get(str(dep))
                    if dep_id is None:
                        # FIX: a dependency label can arrive slightly mangled
                        # relative to how it was declared (e.g.
                        # "__coating_thicknes__" vs "__coating_thickness__").
                        # strip_chars="" here deliberately -- unlike an
                        # action/role/tool name, underscores in a label are
                        # part of its identity, not LLM decoration, so both
                        # sides are normalized (whitespace + casefold only)
                        # and compared as-is rather than trimmed.
                        normalized_dep = normalize_identifier(
                            str(dep), list(label_to_id.keys()), cutoff=0.75, strip_chars=""
                        )
                        if normalized_dep:
                            dep_id = label_to_id.get(normalized_dep)
                    if dep_id and dep_id not in admitted_ids:
                        # Its target was dropped (by admission, or as off-topic
                        # by the derived-subtask guard) and will never exist;
                        # waiting on it would leave this task pending.
                        print(
                            f"  [admission] subtask '{description[:60]}...' "
                            f"depended on a subtask that was dropped -- "
                            f"dependency removed."
                        )
                    elif dep_id:
                        resolved_deps.append(dep_id)
                    else:
                        print(
                            f"Warning: subtask '{description[:60]}...' lists "
                            f"dependency '{dep}' which doesn't match any "
                            f"label/index in this SPAWN batch -- dropped."
                        )

                spawn_kwargs = dict(
                    description=description,
                    role=sub.get("role", "worker"),
                    parent_id=parent_id,
                    dependencies=resolved_deps,
                    task_id=task_id,
                    label=sub.get("label"),
                )
                if cap is not None and i >= cap:
                    # Queued rather than spawned now. NOTE (known edge case,
                    # matching TaskGraph.add_task's own "depends on unknown
                    # task_id" limitation): if this item depends on ANOTHER
                    # still-queued overflow item rather than an
                    # already-kept/spawned one, its in_degree will never
                    # resolve until that sibling is drained first -- fine
                    # for the common case this replaces (independent
                    # overflow items), not handled for a genuine chain of
                    # dependent overflow items.
                    self.pending_overflow.setdefault(parent_id, []).append(spawn_kwargs)
                else:
                    self._spawn_child_task(**spawn_kwargs)
            return True

        # Single-subtask case.
        description = payload.get("task_id") or payload.get("description")
        role = payload.get("role", "worker")
        role = self._enforce_child_role(role, parent_id)

        if not description:
            print("Warning: Spawn request ignored due to missing task description in payload.")
            return True

        original_description = description
        description = self._plain_subtask_description(description)
        # A lone subtask that fails the overlap check is the whole batch, so
        # _screen_derived_subtask_batch rejects it outright; `dropped` is
        # always empty here.
        proceed, _ = self._screen_derived_subtask_batch(
            event, [description], [original_description])
        if not proceed:
            return False

        spawner = self.colony.get_agent(parent_id) if parent_id else None
        if spawner is not None and self._admit_spawn_batch(spawner.task_id, [role]) == 0:
            self._refuse_spawn_for_budget(parent_id, spawner)
            return False

        self._spawn_child_task(
            description=description,
            role=role,
            parent_id=parent_id,
            dependencies=payload.get("dependencies", []),
        )
        return True

    def handle_parent_notification(self, event: Event):
        """
        Delivers a completed child's result back into the parent's own live
        reasoning context, and clears the parent's `awaiting` gate so
        orchestrator._run_live_agents() resumes ticking it.
        """
        payload = event.payload
        parent_id = payload.get("parent_id")
        child_id = payload.get("child_id")
        result = payload.get("result")

        live_parent = self.live_agents.get(parent_id)
        if live_parent is None:
            return

        # FIX (#4, free-energy bug): this injection happens outside any
        # _run_live_agents() tick, so the per-tick energy debit (which only
        # measures thought_process growth across a single run() call) never
        # sees it -- confirmed via repro that injecting 5,000 characters
        # cost 0 energy. Capped at the same 400-char length
        # _build_dependency_context uses (consistency with the rest of the
        # "shared state" mechanism) and now explicitly debited using the
        # same chars-per-100 proxy used everywhere else in the energy model.
        # Scrubbed at injection too, like _build_dependency_context. A
        # promoted result is already clean. An abandoned task's salvaged
        # partial never passed through promotion, and this is where it would
        # otherwise enter the parent's prompt uncleaned.
        # Grounded by the child's TASK, carried in the payload: on the
        # success-cache and abandonment paths the child agent is already
        # unregistered, and resolving through it fell back to the goal alone
        # -- cutting a timestamp the task itself gave, which promotion kept.
        result_str = self._scrub_result(
            str(result).strip(), "parent_injection",
            task_id=payload.get("task_id") or child_id,
        )
        if len(result_str) > 400:
            result_str = result_str[:400] + "..."
        injected_text = f"\n[CHILD RESULT - {child_id}]: {result_str}\n"
        live_parent.thought_process += injected_text

        injection_cost = max(1, len(injected_text) // 100)
        self.colony.debit_energy(parent_id, injection_cost, category="injection")

        # The ceiling, checked at bill time. An injection is billed to a
        # parent that is `awaiting` its children, and _run_live_agents skips
        # an awaiting agent -- so a decomposer's task could be pushed over
        # its ceiling by its children's results with nothing consulting the
        # line until the parent's own next cycle. Checked here, the task
        # stops on the debit that crossed it.
        parent_node = self.colony.get_agent(parent_id)
        parent_task_id = getattr(parent_node, "task_id", None) if parent_node else None
        if self._enforce_task_energy_ceiling(parent_id, parent_task_id):
            return

        # Resume the parent only once EVERY child is done. Clearing the gate
        # on the first result ticked a decomposer every heartbeat while its
        # siblings were still running -- a THINK per tick that used to only
        # waste energy, and with the per-agent cycle cap
        # (Agent.MAX_NON_TERMINAL_CYCLES) forces it onto REPORT with half its
        # children's results, which for the root ends the run early.
        still_open = self._open_child_task_ids(parent_id)
        if still_open:
            print(f"  [parent_notification] {parent_id} received {child_id}'s "
                  f"result; still waiting on {len(still_open)} child task(s).")
            return
        live_parent.awaiting = None

    def _open_child_task_ids(self, parent_id: str) -> list:
        """Task ids still IN FLIGHT under this agent -- status 0 (pending) or
        1 (running) -- plus overflow subtasks queued for it but not spawned.

        "Open" means "may still send this agent a parent_notification", and
        nothing more. A child that is not open is CLOSED, and closed is
        either completed (2) or failed / abandoned / never started (3). So
        "no open children" does NOT mean "the children succeeded": anything
        that needs to know whether something succeeded reads
        _direct_child_tally's `completed`, as the SPAWN-refusal fail
        predicate does. Status 3 stays closed rather than open on purpose --
        counted as open, a decomposer would wait forever on a child that
        will never report again.

        Two sources, unioned, so this and the fail predicate cannot disagree
        about a direct child:
          * the agent's task's direct children (TaskGraph.direct_children),
            the view the fail predicate counts;
          * the agent-level `children` list, which also holds grandchildren
            adopted when a child decomposer was retired
            (ColonyState.unregister_agent). Those report to this agent now,
            so it keeps waiting on them as before.
        A direct child from an earlier agent on the same task cannot be open
        here: a decomposer is never ticked -- so cannot REPORT or DIE --
        while a child is open, so its successor starts after they closed.
        """
        open_ids = []
        parent = self.colony.get_agent(parent_id)
        if parent is not None:
            for child in self.task_graph.direct_children(parent.task_id):
                if child.status in (0, 1):
                    open_ids.append(child.task_id)
            for child_id in parent.children:
                child = self.colony.get_agent(child_id)
                if child is None:
                    continue
                task = self.task_graph.tasks.get(child.task_id)
                if (task is not None and task.status in (0, 1)
                        and child.task_id not in open_ids):
                    open_ids.append(child.task_id)
        for queued in self.pending_overflow.get(parent_id) or []:
            open_ids.append(queued.get("task_id"))
        return open_ids

    def _direct_child_tally(self, task_id: Optional[str]) -> Dict[str, int]:
        """How this task's DIRECT children stand: open (status 0/1),
        completed (2), failed (3: abandoned, never started, or failed by a
        cascade). The three always add up to len(direct_children)."""
        tally = {"open": 0, "completed": 0, "failed": 0}
        for child in self.task_graph.direct_children(task_id):
            if child.status in (0, 1):
                tally["open"] += 1
            elif child.status == 2:
                tally["completed"] += 1
            else:
                tally["failed"] += 1
        return tally

    def _has_completed_direct_child(self, task_id: Optional[str]) -> bool:
        """The one predicate shared by the SPAWN-refusal fail branch and the
        decomposer REPORT gate: did at least one DIRECT subtask complete?"""
        return self._direct_child_tally(task_id)["completed"] > 0

    def _direct_child_status_label(self, child: TaskNode) -> str:
        if child.status == 2:
            return "completed"
        if child.status == 1:
            return "running"
        if child.status == 0:
            return "pending"
        if child.task_id in self.unstarted_tasks:
            return "not started"
        if child.task_id in self.abandoned_tasks:
            return "abandoned"
        return "failed"

    # How much of a completed subtask's result the YOUR SUBTASKS block shows.
    # Enough to name what it decided; the full result is in the injection.
    CHILD_EXCERPT_CHARS = 140

    @staticmethod
    def _short_label(label) -> Optional[str]:
        """A SPAWN label worth showing back to its decomposer: a short
        string. Anything else (missing, a dict, a paragraph) is dropped."""
        if label is None or isinstance(label, (dict, list)):
            return None
        label = " ".join(str(label).split())
        return label if 0 < len(label) <= 40 else None

    def _direct_child_statuses(self, task_id: Optional[str]) -> list:
        """One row per DIRECT child of this task, read fresh from the graph:
        what Agent.child_status_fn hands a decomposer's prompt, so what it is
        told about its subtasks never depends on which injections are still
        inside its 500-character thought window -- run 3's root decided with
        all three [ABANDONED] injections pushed out of it."""
        rows = []
        for child in self.task_graph.direct_children(task_id):
            status = self._direct_child_status_label(child)
            excerpt = None
            if status == "completed" and child.result:
                excerpt = " ".join(str(child.result).split())
                if len(excerpt) > self.CHILD_EXCERPT_CHARS:
                    excerpt = excerpt[:self.CHILD_EXCERPT_CHARS].rstrip() + "..."
            rows.append({"task": child.description or child.task_id,
                         "status": status,
                         "label": child.label,
                         "excerpt": excerpt})
        return rows

    def handle_completion(self, event: Event):
        """PROMOTE: Routes completed results to parents, unblocks tasks, or triggers synthesizer."""
        payload = event.payload
        agent_id = event.from_agent
        task_id = payload.get("task_id")
        result = payload.get("result")

        if not task_id:
            print(f"Warning: completion_request from {agent_id} missing task_id, ignoring.")
            return

        # FIX (repetition-collapse bug): a REPORT payload can degenerate into
        # the same sentence/paragraph repeated 2+ times before trailing off
        # mid-sentence (confirmed via a real run -- greedy decoding locking
        # onto its own recent output). Previously only DIE text, ghost-context
        # goal lines, and EXECUTE critique text got this treatment; a
        # successful REPORT's result never did, so a degenerate-but-"passing"
        # result could get judged, stored, and promoted to a parent verbatim.
        # Deduped here, before the judge ever sees it, so the stored task
        # result and whatever gets handed to a parent via parent_notification
        # are both the cleaned version.
        if result:
            result = _dedupe_repeated_sentences(str(result), max_chars=2000)

        # Belt-and-braces companion to _ActionPayloadStop.REPORT_MAX_NEW_TOKENS.
        # That cap stops the model from GENERATING a 400-token REPORT; this
        # stops a long one that got through anyway from reaching the judge,
        # which reads padding as evidence the agent did not answer the
        # subtask. Trimming to the first 2-3 sentences keeps the part that
        # answers it and drops the trailing restatement. Code results are
        # exempt for the same reason they bypass the tier-2 similarity
        # check below: sentence boundaries mean nothing in source, and
        # "the first three sentences" of a function is a broken function.
        if result and not self._looks_like_code(result):
            # What feeds the trim below. The REPORT generation budget stops
            # the model mid-sentence (it has no closing token to wait for),
            # so without this the trim's last kept "sentence" can be a
            # stump. Walk back to the last complete sentence first, then
            # trim -- the trim decides how much the judge reads, this
            # decides where the text actually ended.
            completed = drop_incomplete_tail(str(result))
            if completed != str(result):
                print(f"  [report-trim] {agent_id} REPORT dropped an "
                      f"incomplete trailing sentence "
                      f"({len(str(result))} -> {len(completed)} chars).")
            # PATCH 14. The prompt header read back out, cut before anything
            # else measures this REPORT. Run 5 ended agent_6ab689f1's root
            # REPORT on "Your turn. Your role: executor Your task: run member
            # voting...", which is the agent's own prompt, not the worked-
            # REPORT exemplar -- so is_exemplar_echo never saw it and the
            # structural-reject counter read 0 while it happened.
            #
            # CUT, NOT REJECTED, and this is the whole finding: in run 5 the
            # echo was only ever in the tail, behind a complete answer that
            # the 3-sentence trim below was going to keep and the echo to
            # drop. A reject here would have killed the root REPORT of the
            # run's first honest success over text that never reached a
            # single consumer. What was actually missing was the counter.
            #
            # Cut first, so the tail trimmers and the sentence trim below
            # work on the answer rather than on the prompt: "Your task: run
            # member voting on book selections" reads as an ordinary
            # sentence to both of them and can be kept as one of the three.
            unheaded, header_reason = cut_prompt_header_echo(completed)
            if header_reason is not None:
                self.colony.record_verdict("report_prompt_header_echo_detected")
                if unheaded.strip():
                    self.colony.record_verdict("report_prompt_header_echo_cut")
                    print(f"  [report-trim] {agent_id} REPORT cut at "
                          f"{header_reason} ({len(completed)} -> "
                          f"{len(unheaded)} chars).")
                    completed = unheaded
                else:
                    # The REPORT is prompt header from its first character.
                    # Left whole for the judge to reject, the same stance the
                    # status-report tail takes two comments down: blanking it
                    # here would hand the parent an empty answer instead of a
                    # rejected one.
                    self.colony.record_verdict("report_prompt_header_echo_whole")
                    print(f"  [report-trim] {agent_id} REPORT is {header_reason} "
                          f"from its first character -- left whole for the judge.")

            # Degenerate tails come off BEFORE the trim: closer runs,
            # restating fragments and status-report/deploy-log tails
            # ("Ready to deploy. Done. Go. Final state: COMPLETED."). The trim
            # can cut such a tail partway ("A. B. Exactly five. Done. Five."
            # -> "A. B. Exactly five."), and what is left no longer looks
            # like a tail, so the promotion scrub below let it through. They
            # only ever drop a tail, so the judge and the promoted result
            # still see the same answer. A REPORT that is nothing BUT a
            # status report is left whole for the judge to reject.
            untailed, _ = trim_degenerate_tails(completed, self._grounding_for(task_id))
            if not untailed.strip():
                untailed = completed
            if untailed != completed:
                self.colony.record_verdict("report_tail_trimmed")
                print(f"  [report-trim] {agent_id} REPORT dropped a degenerate "
                      f"tail ({len(completed)} -> {len(untailed)} chars).")
                completed = untailed
            trimmed = trim_to_sentences(completed, max_sentences=3,
                                        max_words=60, marker=False)
            if trimmed != str(result):
                print(f"  [report-trim] {agent_id} REPORT trimmed "
                      f"{len(str(result))} -> {len(trimmed)} chars before judging.")
                result = trimmed

        # PATCH 16 (measure-only). Recorded for EVERY REPORT, before the
        # structural rejects below: a rejected attempt is still an attempt,
        # and the figures it asserted are what the next one has to agree
        # with. Reads the untrimmed payload, like the checks below it.
        self._check_figure_divergence(task_id, agent_id, payload.get("result"))

        # PATCH 11. decide() returning nothing is not an answer, and the
        # string it substitutes is a constant this process wrote -- so an
        # identity comparison is exact and free, and no judge call is needed
        # to establish it. In run 4 it went the whole way: tier 1 passed it
        # (not empty, no trailing colon, not a heading, 52 alphanumeric
        # characters, and judge._PLACEHOLDER_REPORTS misses it because that
        # lookup strips " \t\n.-_*#" but not the square brackets), tier 2 was
        # skipped twice over (9 words, decomposer role), and tier 3 spent a
        # full deep_critique writing ~500 characters about a placeholder, for
        # 5 energy.
        #
        # Placed here with the other two pre-judge structural rejects, and
        # deliberately ABOVE last_partial_result: recorded below, this text
        # would become the salvaged "partial result" handed to the parent if
        # the task were later abandoned. Reads the raw payload, like the
        # checks around it -- report-trim rewrites the local `result`, not
        # payload["result"].
        empty_agent = self.colony.get_agent(agent_id)
        if empty_agent is not None and payload.get("result") == EMPTY_DECISION_PLACEHOLDER:
            self.colony.record_verdict("report_empty_generation_rejected")
            print(f"REJECT (structural) on {agent_id}/{task_id}: the REPORT is the "
                  f"empty-generation placeholder, not an answer -- rejected before "
                  f"the judge.")
            empty_agent.fail_reason = (
                "Previous attempt was REJECTED: your last response produced no "
                "content at all, so nothing was submitted. Write the answer to "
                "your task as plain sentences after 'PAYLOAD:'."
            )
            self._kill_and_respawn(
                agent_id, task_id, empty_agent.role, empty_agent.parent_id,
                verdict={"verdict": "execute", "reason": empty_agent.fail_reason},
            )
            return

        # The prompt's own worked example handed back as the answer. It is
        # domain-free by design, so it answers nothing: run 3 got it back as
        # agent_8d992982's whole REPORT, which was stopped only because tier 2
        # happened to run on its 14 words -- at 12 or fewer, or from a
        # decomposer, tier 2 is skipped and tier 3 alone would have read it.
        # Rejected here whatever the length or role, before the judge and
        # before last_partial_result, so it is never salvaged for a parent
        # either. Reads the untrimmed payload, like the check below. A REPORT
        # that quotes the example ONCE and then gives its own answer is not
        # rejected: the promotion scrub cuts the quoted sentence out of it.
        # Two or more copies IS rejected (PATCH 11, is_exemplar_echo).
        # The worked REPORT only: an echoed SPAWN placeholder still goes to
        # the judge, and the promotion scrub marks it if it gets that far.
        echo_agent = self.colony.get_agent(agent_id)
        if echo_agent is not None and is_exemplar_echo(payload.get("result"),
                                                       EXEMPLAR_REPORT_TEXTS):
            self.colony.record_verdict("report_exemplar_echo_rejected")
            print(f"REJECT (structural) on {agent_id}/{task_id}: the REPORT is the "
                  f"prompt's own worked example, not an answer -- rejected before "
                  f"the judge.")
            echo_agent.fail_reason = (
                "Previous attempt was REJECTED: your REPORT repeated the example "
                "answer from the instructions instead of answering your task. "
                "That example is not about your task. REPORT your own answer to "
                "YOUR task, in your own words."
            )
            self._kill_and_respawn(
                agent_id, task_id, echo_agent.role, echo_agent.parent_id,
                verdict={"verdict": "execute", "reason": echo_agent.fail_reason},
            )
            return

        # Software framing: a REPORT that is a file list, a function or a
        # made-up checksum on a project that never asked for software is not
        # an answer to it, however well the judge's similarity score rates
        # its vocabulary. Rejected before the judge and never promoted to a
        # parent -- and checked BEFORE last_partial_result below, or a task
        # later abandoned by the attempt cap would hand this exact text to
        # its parent as the salvaged answer anyway. Reads the untrimmed
        # payload: the 3-sentence report-trim above can cut the file list or
        # checksum line off and let the rest through.
        software_reason = self._software_framing_reason(payload.get("result"))
        if software_reason is not None:
            self.colony.record_verdict("software_framing_report_detected")
        framing_agent = self.colony.get_agent(agent_id)
        if (software_reason is not None and framing_agent is not None
                and "block" not in self.framing_levers):
            print(f"  [software-framing] REPORT from {agent_id} {software_reason} "
                  f"(block lever off).")
        elif software_reason is not None and framing_agent is not None:
            self.colony.record_verdict("software_framing_report_rejected")
            print(f"REJECT (software framing) on {agent_id}/{task_id}: REPORT "
                  f"{software_reason}, but this project never asked for software.")
            framing_agent.fail_reason = (
                f"Previous attempt was REJECTED: your answer "
                f"{_without_quoted_example(software_reason)}, "
                f"but nobody asked for a program. Give the answer itself in "
                f"plain sentences -- the decision, list, plan or text the task "
                f"asks for. No code, pseudocode, file names or function names."
            )
            self._kill_and_respawn(
                agent_id, task_id, framing_agent.role, framing_agent.parent_id,
                verdict={"verdict": "execute", "reason": framing_agent.fail_reason},
            )
            return

        agent_node = self.colony.get_agent(agent_id)

        # FIX (root-acceptance bug, widened to every decomposer): a
        # decomposer's answer is its children's work rolled up. With not one
        # DIRECT subtask completed there is nothing to roll up, so the REPORT
        # is the decomposer answering the task itself -- rejected
        # structurally, before the judge and its expensive tier-3 critique,
        # and respawned rather than allowed to fall through as a "result".
        #
        # On every path, not only at the end of a run: Agent.final_actions
        # withholds REPORT from a decomposer with nothing completed once it
        # is capped or SPAWN-closed, but a decomposer woken by its last
        # child's abandonment still has REPORT on its ordinary menu, and only
        # tier 3 read what it wrote. It also covers a decomposer that never
        # spawned at all, including an executor converted by TASK TOO LARGE.
        #
        # Same predicate as the SPAWN-refusal fail branch, over
        # TaskGraph.direct_children: AgentNode.children holds grandchildren
        # adopted from retired child decomposers, which is how run 3's root
        # passed the old root-only version of this gate with 0 of its 3
        # subtasks completed.
        #
        # Checked BEFORE last_partial_result below, like the two checks
        # above: recorded first, a task later abandoned would hand its parent
        # exactly the text rejected here as its salvaged answer.
        if (agent_node is not None and agent_node.role == "decomposer"
                and not self._has_completed_direct_child(task_id)):
            is_root = task_id == self.root_task_id
            self.colony.record_verdict("decomposer_report_no_completed_child")
            print(f"REJECT (structural) on {agent_id}/{task_id}: "
                  f"{'root ' if is_root else ''}decomposer REPORTed with no "
                  f"completed subtasks -- a decomposer submits what its "
                  f"children produced, it does not answer the task itself.")
            agent_node.fail_reason = (
                "Previous attempt was REJECTED: you REPORTed a final answer "
                "directly instead of decomposing the task. As the ROOT "
                "decomposer (generation 0) your job is to SPAWN subtasks, "
                "not answer the question yourself."
                if is_root else
                "Previous attempt was REJECTED: you REPORTed an answer of your "
                "own, but none of your subtasks completed, so there were no "
                "results to combine. SPAWN the work as subtasks, or DIE if it "
                "cannot be decomposed."
            )
            self._kill_and_respawn(
                agent_id, task_id, agent_node.role, agent_node.parent_id,
                verdict={"verdict": "execute", "reason": agent_node.fail_reason},
            )
            return

        # Remembered whatever the judge decides next: if this task is later
        # abandoned by the attempt cap, this is the salvage value handed to
        # the parent. Recorded before judging on purpose -- a rejected
        # REPORT is still more than nothing.
        if result and task_id:
            self.last_partial_result[task_id] = str(result)

        verdict = {"verdict": "promote", "reason": "no judge configured"}
        if self.judge is not None and agent_node is not None:
            output_embedding = None
            is_root = task_id == self.root_task_id

            # N2b: score against this task's OWN description, not the
            # colony's overall goal -- a subtask several decomposition
            # levels down was never going to read as semantically similar
            # to the whole project goal even when it's doing exactly the
            # right thing. The root task has no separate description to
            # target (its description already IS the goal), so it alone
            # keeps using colony.goal_embedding.
            if is_root:
                target_embedding = self.colony.goal_embedding
            else:
                completing_task_node = self.task_graph.tasks.get(task_id)
                target_embedding = (
                    completing_task_node.description_embedding
                    if completing_task_node is not None else None
                )

            word_count = len(str(result).split())
            # Tier 2 is a MiniLM cosine between the output and the task
            # description. Comparing Python source against an English task
            # description puts correct and useless answers within ~0.1 of
            # each other (measured: a plausible implementation scored 0.593,
            # 270 words of content-free filler scored 0.496) -- near enough
            # to noise that it strike-limits agents for being roughly right.
            # A code output is bypassed for the same reason a very short
            # answer already is: the score would not mean anything.
            looks_like_code = self._looks_like_code(result)
            skip_tier2 = (not is_root) and (
                word_count <= self.SHORT_ANSWER_WORD_THRESHOLD or looks_like_code
            )
            print(f"  [judge-bypass-check] word_count={word_count} "
                  f"threshold={self.SHORT_ANSWER_WORD_THRESHOLD} "
                  f"looks_like_code={looks_like_code} "
                  f"will_skip_tier2={skip_tier2} "
                  f"result={str(result)!r}")
            if (self.embed_model is not None and target_embedding is not None
                    and not skip_tier2):
                try:
                    output_embedding = self.embed_model.encode(str(result), convert_to_numpy=True)
                except Exception as e:
                    print(f"Warning: failed to embed output for judging: {e}")
            verdict = self.judge.decide(
                agent_node,
                output=result,
                output_type="text",
                output_embedding=output_embedding,
                target_embedding=target_embedding,
                # A8 STEP 2, re-enabled: tier 1 now closes the empty-answer
                # hole (see judge.fast_check), so a REPORT reaching
                # deep_critique is no longer a bare heading/colon/placeholder
                # by construction. Tools stay disabled (see
                # _run_live_agents's available_tools) -- this is the clean
                # tier-3 experiment: if deep_critique itself is still
                # blocking subtask completion, the tier3_accept/tier3_reject
                # counters below will now actually populate and show it.
                needs_deep_check=True,
                # PATCH 8. Tier 3 judged against agent.task and nothing else,
                # so an ISBN-deduplication answer to an ISBN-deduplication
                # subtask was correctly scored as fully on-topic -- the
                # critique had no way to see that the subtask itself did not
                # belong to this project. The goal goes in as context for one
                # added question; the other three stay about agent.task.
                # goal_embedding is measure-only (see judge.decide): it
                # annotates a tier-2 verdict the task-level score already
                # reached, and gates nothing.
                project_goal=self._project_goal_text(),
                goal_embedding=self.colony.goal_embedding,
            )

            if verdict.get("tier") == 3:
                critique_cost = max(1, len(str(verdict.get("reason", ""))) // 100)
                self.colony.debit_energy(agent_id, critique_cost, category="tier3_critique")
                print(f"  [judge tier-3 cost] agent={agent_id} deep_critique "
                      f"debited {critique_cost} energy (previously untracked).")

                # Step 1: count the tier-3 split explicitly. decide() maps
                # deep_critique's accept -> "promote" and reject ->
                # "execute", so the tier-3 verdict has to be read back off
                # that mapping here. Without this the accept rate is only
                # inferrable by eyeballing log lines; with it, the final
                # report states it as a number.
                self.colony.record_verdict(
                    "tier3_accept" if verdict["verdict"] == "promote" else "tier3_reject"
                )

        if verdict["verdict"] == "warn":
            print(f"Judge WARN on {agent_id}/{task_id}: {verdict['reason']}")
            self.colony.record_verdict("judge_warn")

            # A WARN is a non-terminal cycle by the same definition the agent
            # already applies to THINK: nothing finished and it is going back
            # around. This was the one decision point in the system that
            # counted nothing -- a task has respawn_counts, an agent has
            # non_terminal_cycles, a WARN had neither -- so REPORT -> WARN ->
            # REPORT ran for as long as the wording kept changing, and at full
            # price: report() clears the KV cache, and think() resets
            # _total_generated whenever the cache is None, so
            # MAX_TOTAL_THINK_TOKENS restarted on every pass instead of
            # bounding the loop. Charging the cycle puts the loop under the
            # per-agent cap that already exists rather than adding another.
            live_agent = self.live_agents.get(agent_id)
            if live_agent is not None:
                was_capped = live_agent.cycles_capped
                live_agent.non_terminal_cycles += 1
                # The same tally _run_live_agents keeps when run() crosses the
                # cap, repeated here because a WARN crosses it OUTSIDE run():
                # the branch below respawns the agent in this same call, so
                # _run_live_agents never sees a capped agent and the ledger
                # reported zero agents reaching a cap that had just cut four
                # warn loops. Guarded the same way it is there, so an agent
                # that already capped on a THINK is not counted twice.
                if live_agent.cycles_capped and not was_capped:
                    self.colony.record_verdict("cycle_cap_reached")

            # A WARN sends the agent back to try again. If it comes back with
            # a byte-identical REPORT, it is not converging -- it has already
            # shown it will re-derive the same string, and a third strike
            # spent to arrive at the same place is a wasted cycle (observed:
            # one agent burning two WARNs on one identical string). Skip
            # straight to the respawn the strikes would have produced.
            previous = self.last_warned_report.get(agent_id)
            if previous is not None and previous == str(result):
                print(f"  [warn-short-circuit] {agent_id} REPORTed a "
                      f"byte-identical result after a WARN -- respawning now "
                      f"instead of spending another strike on the same string.")
                if agent_node is not None:
                    agent_node.fail_reason = (
                        "Previous attempt was REJECTED: you submitted exactly "
                        "the same REPORT twice after being asked to revise it. "
                        "Repeating the same output is not a revision -- change "
                        "the approach, not the wording."
                    )
                self.last_warned_report.pop(agent_id, None)
                role = getattr(agent_node, "role", "worker") if agent_node else "worker"
                parent_id = getattr(agent_node, "parent_id", None) if agent_node else None
                self._kill_and_respawn(agent_id, task_id, role, parent_id, verdict=verdict)
                return

            # Out of cycles and still not accepted. Sending it back again is
            # what made this a loop rather than a retry: the cap narrows
            # decide() to REPORT/DIE, but REPORT is exactly what a WARN
            # answers, so a capped agent kept re-REPORTing with its counter
            # climbing past the cap and nothing reading it. A WARN at the cap
            # is an EXECUTE in everything but name -- routed the same way,
            # which is also the only path that consults MAX_TASK_ATTEMPTS.
            if live_agent is not None and live_agent.cycles_capped:
                self.colony.record_verdict("warn_cycle_cap_respawn")
                print(f"  [warn-cycle-cap] {agent_id} spent its "
                      f"{live_agent.non_terminal_cycles} cycles without an "
                      f"accepted REPORT -- respawning instead of warning again.")
                if agent_node is not None:
                    agent_node.fail_reason = (
                        "Previous attempt was REJECTED: review sent your REPORT "
                        "back to be revised and you ran out of attempts without "
                        "an acceptable answer. Do not restate the task or pad "
                        "the result -- give the answer itself."
                    )
                self.last_warned_report.pop(agent_id, None)
                role = getattr(agent_node, "role", "worker") if agent_node else "worker"
                parent_id = getattr(agent_node, "parent_id", None) if agent_node else None
                self._kill_and_respawn(agent_id, task_id, role, parent_id, verdict=verdict)
                return

            # The task's cumulative energy, checked before the agent is let
            # back around. A WARN is the one decision point that continues an
            # agent without routing through _kill_and_respawn, so this loop --
            # REPORT -> WARN -> REPORT, each pass paying for a full cycle plus
            # the judge's own tier-3 critique -- is how a task crossed its
            # ceiling mid-agent with no respawn boundary to be caught at,
            # as task_ac3e5d85 did. Reproduced in
            # test_the_warn_loop_is_stopped_mid_agent_not_at_the_respawn_boundary:
            # without this check the loop ran to 106 against a ceiling of 81
            # and was closed by the respawn-time check, as before.
            # Placed after the short-circuit and cycle-cap branches above so
            # their _kill_and_respawn keeps its own precedence (success cache,
            # then attempt cap, then ceiling), and after the judge has ruled,
            # so the REPORT being abandoned is one already rejected -- and one
            # already recorded in last_partial_result, so it is salvaged for
            # the parent rather than discarded.
            if self._enforce_task_energy_ceiling(agent_id, task_id):
                return

            # Now that these passes are counted, they have to be worth
            # spending: the agent was previously re-ticked with no idea a WARN
            # had happened at all -- same prompt, same fail_reason, one more
            # sentence of its own thoughts -- so it re-derived the same answer
            # and burned the cycle for nothing. One clause, prefixed as a
            # verdict, is the same shape the EXECUTE branch below uses, and
            # for the same reason: judge prose at full length reads as more of
            # the agent's own reasoning rather than as a ruling on it.
            if agent_node is not None:
                warn_text = first_clause(
                    _dedupe_repeated_sentences(str(verdict.get("reason", ""))),
                    max_chars=120,
                )
                agent_node.fail_reason = (
                    f"SENT BACK BY REVIEW -- {warn_text}" if warn_text else
                    "SENT BACK BY REVIEW -- your last REPORT was not accepted."
                )

            self.last_warned_report[agent_id] = str(result)
            return

        if verdict["verdict"] == "execute":
            print(f"Judge EXECUTE on {agent_id}/{task_id}: {verdict['reason']}")

            if agent_node is not None:
                # The critique used to be pasted in at up to 400 chars of the
                # judge's own flowing prose. Sitting in the next prompt a few
                # lines above "Your Previous Thoughts", prose of that length
                # and register is indistinguishable from the agent's own
                # reasoning -- so the model did not act on it as a verdict, it
                # continued it as thought. One clause, prefixed as a verdict,
                # cannot be read that way.
                critique_text = _dedupe_repeated_sentences(str(verdict.get("reason", "")))
                critique_text = first_clause(critique_text, max_chars=120)
                agent_node.fail_reason = (
                    f"REJECTED BY REVIEW -- {critique_text}"
                    if critique_text else
                    "REJECTED BY REVIEW -- output did not satisfy the subtask."
                )

            role = getattr(agent_node, "role", "worker") if agent_node else "worker"
            parent_id = getattr(agent_node, "parent_id", None) if agent_node else None
            self._kill_and_respawn(agent_id, task_id, role, parent_id, verdict=verdict)
            return

        # Accepted. This `result` becomes the canonical one: stored, set on
        # the task node, sent to the parent, written to the success cache,
        # and read by every dependent sibling. The dedupe and report-trim
        # above only shaped what the judge read. A tail like "Exactly five.
        # Done." or a sentence that started looping survived both, got
        # accepted, and went everywhere from here. Scrubbing once, before
        # any of those consumers, follows judge.py's rule that every
        # consumer gets clean text from the origin. A success-cache hit in
        # _kill_and_respawn re-promotes a value written below, so it is
        # covered without a check of its own.
        result = self._scrub_result(result, "promotion", task_id=task_id)

        print(
            f"SUBTASK COMPLETE: {agent_id}/{task_id} -- "
            f"\"{self.task_graph.tasks.get(task_id).description if self.task_graph.tasks.get(task_id) else '?'}\" "
            f"-- verdict={verdict['verdict']} ({verdict.get('reason', 'no judge configured')})"
        )

        # The ceiling, checked after judging. A REPORT cycle is exempt from
        # the per-cycle check so a possibly-good result reaches the judge
        # instead of being thrown away unread -- which leaves this, the
        # promotion, as the one exit from that exemption with no check at
        # all (the tier-3 critique above is billed on this same path). The
        # result has been produced and accepted, so abandoning it now would
        # waste finished work; it is kept, and the overrun recorded instead.
        overrun = self._task_energy_overrun(task_id)
        if overrun is not None:
            spent, ceiling = overrun
            self.completed_over_ceiling[task_id] = (spent, ceiling)
            print(f"  [energy-ceiling] {agent_id} completed task {task_id} at "
                  f"{spent} energy against a ceiling of {ceiling} -- result "
                  f"kept, flagged as completed over ceiling.")

        self.task_graph.complete_task(task_id)
        self.colony.store_result(task_id, result)

        task_node = self.task_graph.tasks.get(task_id)
        if task_node:
            task_node.result = result
        self._thread_results_to_unblocked_dependents(task_node)

        # Root completion check
        if task_id == self.root_task_id:
            print("Root task completed! Triggering Synthesizer...")
            if self.synthesizer is not None:
                goal_text = self._synthesis_problem_text()
                try:
                    final_answer = self.synthesizer.run(
                        self.colony, self.task_graph, goal_text,
                        root_task_id=self.root_task_id,
                        abandoned_ids=set(self.abandoned_tasks),
                    )
                    # Read by terminate()'s partial banner: this answer is a
                    # synthesis, not the root agent's raw REPORT.
                    self._final_spec_synthesized = True
                except Exception:
                    print(f"Warning: Synthesizer failed on root completion -- "
                          f"falling back to the raw agent result instead of crashing.\n"
                          f"{traceback.format_exc()}")
                    final_answer = result
            else:
                final_answer = result
            self.colony.results["final_spec"] = final_answer
            self.colony.update_status(agent_id, "completed")
            return

        # FIX (smaller #1): the old fallback
        # `... if agent_node else payload.get("parent_id")` could never fire
        # -- the next line already required `agent_node` truthy regardless.
        # Simplified to be honest about that instead of carrying dead code.
        # If you actually want the payload fallback to work, loosen the
        # `if agent_node and parent_id:` gate below to `if parent_id:`.
        parent_id = getattr(agent_node, 'parent_id', None) if agent_node else None

        # This child just genuinely completed -- a real fan-out slot under
        # its decomposer parent opened up (unlike _kill_and_respawn, which
        # reuses the same task_id/slot). Drain one queued overflow subtask
        # into it, if that parent has any waiting.
        self._drain_pending_overflow(parent_id)

        if agent_node and parent_id:
            self.messenger.push_event(
                "parent_notification",
                "orchestrator",
                {"parent_id": parent_id, "child_id": agent_id, "task_id": task_id,
                 "result": result}
            )

        self.colony.update_status(agent_id, "completed")

        # "promote"/"pass" from judge.decide is an accepted REPORT. The "no
        # judge configured" default is only an acceptance when there really
        # is no judge; with a judge and no agent node, nothing ruled on it.
        unjudged = self.judge is not None and agent_node is None
        self._write_cache_outcome(task_node, task_id, result,
                                  "unjudged" if unjudged else CACHE_OUTCOME_ACCEPTED)

    @staticmethod
    def _cache_outcome(verdict: Optional[Dict[str, Any]]) -> str:
        """Failure outcome a _kill_and_respawn records in the success cache.
        verdict is None only for a DIE: nothing sets judge_verdict."""
        if verdict is None:
            return "die"
        if verdict.get("verdict") == "warn":
            return "warn_exhausted"
        tier = verdict.get("tier")
        if tier == 3:
            return "tier3_reject"
        if tier in (1, 2):
            return f"tier{tier}_execute"
        return "structural_reject"  # software framing / root structural

    def _write_cache_outcome(self, task_node, task_id, result, outcome: str):
        """Every terminal outcome goes into the success cache with its
        outcome recorded. Only CACHE_OUTCOME_ACCEPTED is ever served; a
        negative entry carries no result, so there is nothing in it to serve."""
        if self.memory_store is None or task_node is None:
            return
        accepted = outcome == CACHE_OUTCOME_ACCEPTED
        try:
            self.memory_store.write("success", task_node.description, {
                "result": result if accepted else None,
                "task_id": task_id,
                "outcome": outcome,
            })
        except Exception:
            print(f"Warning: failed writing success cache for {task_id}:\n"
                  f"{traceback.format_exc()}")
            return
        self.colony.record_verdict(
            "cache_write_positive" if accepted else f"cache_write_negative_{outcome}")

    def handle_failure(self, event: Event):
        """EXECUTE: Harvest context, kill agent, and respawn a smarter version."""
        payload = event.payload
        agent_id = event.from_agent
        task_id = payload.get("task_id")
        role = payload.get("role", "worker")
        parent_id = payload.get("parent_id")
        verdict = payload.get("judge_verdict")

        die_text = payload.get("result", "") or ""
        # A DIE's text becomes the dead agent's fail_reason (Agent.die), and
        # extract_agent_ghost hands that to the respawn as ghost context. So
        # a "TASK TOO LARGE: split into scheduler.py, scorer.py, ..." would
        # seed the replacement decomposer with the very software breakdown
        # it should not make. With the "block" lever on, the reason is
        # replaced by a plain note before the respawn reads it; the DIE
        # itself (and the TASK TOO LARGE re-plan below) still proceeds.
        software_reason = self._software_framing_reason(die_text)
        if software_reason is not None:
            self.colony.record_verdict("software_framing_die_detected")
            dead_node = self.colony.get_agent(agent_id)
            if "block" in self.framing_levers and dead_node is not None:
                self.colony.record_verdict("software_framing_die_scrubbed")
                too_large = die_text.startswith("TASK TOO LARGE:")
                dead_node.fail_reason = (
                    f"Previous attempt DIED{' (task too large)' if too_large else ''}. "
                    f"Its stated reason {_without_quoted_example(software_reason)} "
                    f"and was withheld, "
                    f"because this project never asked for software. Treat "
                    f"the task as decisions, lists, plans or writing, in "
                    f"plain sentences."
                )
                print(f"  [software-framing] DIE from {agent_id} {software_reason} "
                      f"-- withheld from the respawn's ghost context.")
            else:
                print(f"  [software-framing] DIE from {agent_id} {software_reason} "
                      f"(block lever off).")
        if verdict is None and role == "executor" and die_text.startswith("TASK TOO LARGE:"):
            if task_id and not self._admit_conversion(task_id):
                # No budget to decompose: the executor's DIE stands, and the
                # parent gets whatever it left instead of a re-plan.
                print(f"Agent {agent_id} reported its task as too large, but the "
                      f"uncommitted budget cannot fund decomposing it -- closing "
                      f"the task instead of converting.")
                self._retire_agent(agent_id, task_id, verdict)
                self._write_cache_outcome(self.task_graph.tasks.get(task_id), task_id, None, "die")
                self._abandon_task(
                    task_id, parent_id, agent_id, self.respawn_counts.get(task_id, 0),
                    reason=("it was too large for one agent and there was not "
                            "enough uncommitted energy to decompose it further"),
                    reason_label="admission",
                )
                return
            print(f"Agent {agent_id} reported its task as too large -- "
                  f"respawning as a decomposer for the same task instead of an executor.")
            role = "decomposer"

        count_attempt = True
        if payload.get("derived_subtask_rejection") and task_id:
            rejections = self.derived_rejection_counts.get(task_id, 0)
            if rejections < self.MAX_DERIVED_REJECTIONS:
                self.derived_rejection_counts[task_id] = rejections + 1
                self.colony.record_verdict("derived_subtask_respawn_uncounted")
                count_attempt = False

        self._kill_and_respawn(agent_id, task_id, role, parent_id, verdict=verdict,
                               count_attempt=count_attempt)

    def _retire_agent(self, agent_id: str, task_id: Optional[str],
                      verdict: Optional[Dict[str, Any]] = None) -> dict:
        """Take an agent off the colony: ghost record, VRAM, registry.

        Extracted from _kill_and_respawn so the two ways an agent can be
        stopped -- respawned onto the same task, or stopped outright
        because its task is finished with -- tear it down identically.
        Returns the ghost context, which only the respawn path uses.
        """
        live_agent = self.live_agents.pop(agent_id, None)

        # Overflow queued under this agent dies with it. It is keyed by
        # agent_id, so nothing else ever consumes it: a respawn on the same
        # task re-plans from k=0 under a new id, yet committed_energy kept
        # reserving every entry for the rest of the run, and a child of the
        # old agent completing would still drain one under a dead parent.
        dropped = self.pending_overflow.pop(agent_id, None)
        if dropped:
            print(f"  [admission] {agent_id} retired with {len(dropped)} queued "
                  f"overflow subtask(s) -- dropped, releasing their reservation.")
            for _ in dropped:
                self.colony.record_verdict("overflow_dropped_parent_retired")

        if self.memory_store is not None:
            try:
                ghost_source = live_agent if live_agent is not None else self.colony.get_agent(agent_id)
                if ghost_source is not None:
                    record = ghost_extractor.extract(ghost_source, verdict)
                    self.memory_store.write("ghost", record.get("task", task_id or ""), record)
            except Exception:
                print(f"Warning: failed writing ghost record for {agent_id}:\n"
                      f"{traceback.format_exc()}")

        if live_agent is not None:
            live_agent.KV_Cache = None
            live_agent.last_hidden_state = None

        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        ghost_context = self.colony.extract_agent_ghost(agent_id)
        self.colony.unregister_agent(agent_id)
        return ghost_context

    def _note_root_cache_lookup(self, task_node, task_id: str) -> None:
        """Count a root cache lookup that was not served, and say whether an
        accepted entry WOULD have completed the root had it been.

        The probe is read-only -- get_success_cache queries the index and
        builds a result, writing nothing -- and it deliberately touches none
        of the cache's own counters: those belong to the serving path above,
        where they keep counting only lookups that could actually serve.
        """
        self.colony.record_verdict("root_cache_lookup_skipped")
        try:
            lookup = self.memory_store.get_success_cache(task_node.description,
                                                         task_id=task_id)
        except Exception:
            print(f"Warning: root success-cache probe failed for {task_id}:\n"
                  f"{traceback.format_exc()}")
            return
        hit = getattr(lookup, "hit", None)
        if hit is None or hit.get("outcome") != CACHE_OUTCOME_ACCEPTED:
            return
        self.colony.record_verdict("root_cache_hit_withheld")
        print(f"  [success-cache] {task_id} is the ROOT: an accepted entry "
              f"(donor {hit.get('task_id')}, score "
              f"{getattr(lookup, 'hit_score', 0.0):.2f}) would have completed it "
              f"-- withheld. The root is never completed from the cache.")

    def _measure_ghost_lessons(self, agent_id: str, task_id: str, task_node) -> None:
        """PATCH 33, MEASURE-ONLY: would the persistent ghost memory have had
        an earlier lesson for this respawn?

        Queries the ghost index with the task's description, keeps hits at
        or above GHOST_LESSON_THRESHOLD, and drops the ghost _retire_agent
        just wrote for agent_id -- a task must not "find" its own fresh
        death. Prints what it found and counts ghost_lessons_available /
        ghost_lessons_none. Nothing it finds reaches ghost_context or any
        prompt; injection is a later patch. Never raises.
        """
        query_ghosts = getattr(self.memory_store, "query_ghosts", None)
        if not callable(query_ghosts):
            return  # a store with no ghost index (test fakes): nothing to measure
        try:
            # One extra neighbour: the agent's own fresh ghost scores ~1.0
            # against its task and would otherwise take one of the slots.
            raw = query_ghosts(task_node.description, top_k=GHOST_LESSON_TOP_K + 1) or []
            hits = []
            for h in raw:
                meta = h.get("metadata") or {}
                if meta.get("agent_id") == agent_id:
                    continue
                if float(h.get("score", 0.0)) < GHOST_LESSON_THRESHOLD:
                    continue
                hits.append((float(h.get("score", 0.0)), meta))
            hits = hits[:GHOST_LESSON_TOP_K]
        except Exception:
            print(f"Warning: ghost read-back failed for {task_id}:\n"
                  f"{traceback.format_exc()}")
            return
        if not hits:
            self.colony.record_verdict("ghost_lessons_none")
            return
        self.colony.record_verdict("ghost_lessons_available")
        print(f"  [ghost-memory] {task_id}: {len(hits)} earlier failure(s) on similar tasks")
        for score, meta in hits:
            reason = " ".join(str(meta.get("failure_reason") or "").split())
            if len(reason) > 100:
                reason = reason[:97] + "..."
            print(f"      {score:.2f}  {meta.get('failure_type') or 'unknown'}  {reason!r}")

    def _kill_and_respawn(self, agent_id: str, task_id: Optional[str], role: str,
                           parent_id: Optional[str], verdict: Optional[Dict[str, Any]] = None,
                           count_attempt: bool = True):
        """
        Shared kill/ghost/respawn path used by both a self-reported DIE
        (handle_failure) and a judge-triggered EXECUTE (handle_completion).

        count_attempt=False is a respawn after a derived-subtask batch
        rejection (see MAX_DERIVED_REJECTIONS): no agent executed the task,
        so it neither spends an attempt nor records a failed outcome for
        the task in the success cache. The energy ceiling still applies.
        """
        ghost_context = self._retire_agent(agent_id, task_id, verdict)

        if not task_id:
            print(f"Agent {agent_id} failed with no task_id -- cannot respawn.")
            return

        task_node = self.task_graph.tasks.get(task_id)
        if count_attempt:
            self._write_cache_outcome(task_node, task_id, None, self._cache_outcome(verdict))
        if (self.memory_store is not None and task_node is not None
                and task_id == self.root_task_id):
            # The root is never completed from the cache. Serving it here
            # would end the run on a donor task's answer with no synthesis at
            # all: handle_completion's root branch is the only caller of the
            # synthesizer and is not on this path, so
            # colony.results["final_spec"] stays empty and terminate() ships
            # the cached string whole. A root that fails, fails. This path
            # became reachable when the SPAWN-refusal fail branch started
            # routing a root whose subtasks all failed down the DIE path.
            self._note_root_cache_lookup(task_node, task_id)
        elif self.memory_store is not None and task_node is not None:
            lookup = self.memory_store.get_success_cache(task_node.description,
                                                         task_id=task_id)
            cached = getattr(lookup, "hit", None)
            negatives = getattr(lookup, "negative_count", 0)
            blocked = getattr(lookup, "cross_task_below_threshold", 0)
            if blocked:
                print(f"  [success-cache] {task_id}: {blocked} accepted entry(ies) "
                      f"from other tasks below the cross-task threshold -- not served.")
                self.colony.record_verdict("cache_cross_task_below_threshold")
            if negatives:
                print(f"  [success-cache] {task_id}: {negatives} failed outcome(s) on "
                      f"record for this subtask or a near match "
                      f"({getattr(lookup, 'negative_outcomes', {})}).")
            # Never complete a node off anything but an accepted entry,
            # whatever the store handed back.
            if cached is not None and cached.get("outcome") != CACHE_OUTCOME_ACCEPTED:
                cached = None
            if cached is None and negatives:
                self.colony.record_verdict("cache_miss_negative_match")
            if cached is not None:
                self.colony.record_verdict("cache_hit_served")
                if cached.get("task_id") != task_id:
                    self.colony.record_verdict("cache_hit_served_cross_task")
                print(f"Success cache hit for task {task_id} -- skipping respawn "
                      f"(donor {cached.get('task_id')}, score "
                      f"{getattr(lookup, 'hit_score', 0.0):.2f}).")
                self.task_graph.complete_task(task_id)
                cached_result = cached.get("result")
                self.colony.store_result(task_id, cached_result)
                task_node.result = cached_result
                self._thread_results_to_unblocked_dependents(task_node)

                # FIX (#3, silent-stall bug): this branch previously marked
                # the task complete and stored the result but never pushed
                # parent_notification, unlike handle_completion's normal
                # success path. If this child was the only (or last) thing
                # a parent was `awaiting`, the parent never resumed -- and
                # since awaiting agents are exempt from the deadlock
                # watchdog, nothing else caught it either. Mirrors the
                # normal-path notification exactly.
                # A completion frees a fan-out slot on this path too. Without
                # the drain, a queued overflow subtask was never started --
                # and since queued subtasks count as unfinished children in
                # _open_child_task_ids, the parent then waited forever.
                self._drain_pending_overflow(parent_id)
                if parent_id:
                    self.messenger.push_event(
                        "parent_notification",
                        "orchestrator",
                        {"parent_id": parent_id, "child_id": agent_id, "task_id": task_id,
                         "result": cached_result}
                    )
                return

        # A2 attempt cap. Checked here, after the success-cache shortcut
        # (a cache hit is a completion, not another attempt) and before the
        # respawn it would otherwise authorise.
        attempts = self.respawn_counts.get(task_id, 0)
        if count_attempt and attempts >= self.MAX_TASK_ATTEMPTS:
            self._abandon_task(task_id, parent_id, agent_id, attempts)
            return

        # Per-task energy ceiling -- independent of the attempt count above.
        # The same check also runs every cycle in _run_live_agents; this one
        # stays because a respawn is the other way a task can acquire more
        # spend, and it must not be authorised past the line either.
        overrun = self._task_energy_overrun(task_id)
        if overrun is not None:
            # Counted separately from the mid-agent stop so the ledger can
            # say WHICH check is doing the work. Both label the task
            # "energy ceiling" in the abandoned list, which is the right
            # thing for a reader of that list and useless for deciding
            # whether the per-cycle check earns its place.
            self.colony.record_verdict("energy_ceiling_at_respawn")
            self._abandon_task(
                task_id, parent_id, agent_id, attempts,
                reason=self._energy_ceiling_reason(*overrun),
                reason_label="energy ceiling",
            )
            return

        # PATCH 33: ghost read-back, measure-only. Counted once per respawn,
        # here where the respawn is certain to be attempted (cache hits and
        # abandonments above return first). ghost_context is not touched.
        if self.memory_store is not None and task_node is not None:
            self._measure_ghost_lessons(agent_id, task_id, task_node)

        if count_attempt:
            print(f"Agent {agent_id} failed. Respawning {role} with ghost context "
                  f"(attempt {attempts + 2} of {self.MAX_TASK_ATTEMPTS + 1} for this task).")
            self.respawn_counts[task_id] = attempts + 1
        else:
            print(f"Agent {agent_id}'s subtask batch was rejected. Respawning {role} "
                  f"with ghost context (rejection "
                  f"{self.derived_rejection_counts.get(task_id, 0)} of "
                  f"{self.MAX_DERIVED_REJECTIONS}; not counted as a task attempt).")
        if self.spawn_agent(role=role, task_id=task_id, parent_id=parent_id,
                            ghost_context=ghost_context) is None:
            # No energy for the replacement. The old agent is already gone,
            # so the task sat at "running" with no agent: never ticked, never
            # finished, and its parent was never told. Energy does not come
            # back, so close it now with whatever the earlier attempts left.
            self._abandon_task(
                task_id, parent_id, agent_id, attempts,
                reason="the colony had no energy left to respawn an agent for it",
                reason_label="no energy to respawn",
            )

    # ------------------------------------------------------------------
    # Energy reservation (see RESERVE_* / ADMISSION_* in __init__)

    def _reservation(self, task_id: str) -> TaskReservation:
        """The task's reservation record, created on first use. A task that
        reached the graph without going through spawn admission (the root
        before bootstrap, recovery spawns, hand-built test graphs) is
        classified from its agent's role, or its required_role."""
        res = self.task_reservations.get(task_id)
        if res is not None:
            return res
        task = self.task_graph.tasks.get(task_id)
        agent = (self.colony.get_agent(task.agent_id)
                 if task is not None and task.agent_id else None)
        role = (getattr(agent, "role", None)
                or getattr(task, "required_role", None) or "executor")
        if task_id == self.root_task_id:
            kind = "root"
        elif role == "decomposer":
            kind = "decomposer"
        else:
            kind = "executor"
        res = TaskReservation(kind=kind)
        if kind != "executor" and agent is not None and agent.children:
            res.spawned, res.k = True, len(agent.children)
        self.task_reservations[task_id] = res
        return res

    def _own_cost(self, res: TaskReservation) -> float:
        """One attempt's cost to its own task. A decomposer that has not
        SPAWNed yet is priced at k=1, its cheapest decomposition."""
        if res.kind == "executor":
            return self.RESERVE_EXECUTOR
        k = res.k if res.spawned else 1
        cost = self.RESERVE_DECOMPOSER_BASE + self.RESERVE_PER_CHILD * k
        if res.kind == "root":
            cost += self.RESERVE_ROOT_EXTRA
        return cost

    @staticmethod
    def _child_kind(res: TaskReservation) -> str:
        # Mirrors _enforce_child_role: the root's children are decomposers,
        # every other decomposer's are executors.
        return "decomposer" if res.kind == "root" else "executor"

    def _unspawned_children_reservation(self, res: TaskReservation) -> float:
        """A decomposer that has not SPAWNed yet also reserves the one child
        its k=1 pricing assumes, so its parent cannot admit work the child
        will then be unable to afford."""
        if res.kind == "executor" or res.spawned:
            return 0.0
        return self._new_task_reservation(self._child_kind(res))

    def _new_task_reservation(self, kind: str) -> float:
        """Full reservation of a task about to be created (nothing spent)."""
        res = TaskReservation(kind=kind)
        return self._own_cost(res) + self._unspawned_children_reservation(res)

    @staticmethod
    def _reservation_kind(role: Optional[str]) -> str:
        return "decomposer" if role == "decomposer" else "executor"

    def _remaining_reservation(self, task_id: str) -> float:
        res = self._reservation(task_id)
        spent = self.colony.task_energy_spent.get(task_id, 0)
        return (max(0.0, res.sunk + self._own_cost(res) - spent)
                + self._unspawned_children_reservation(res))

    def committed_energy(self) -> float:
        """Spent so far plus everything still reserved: every unfinished
        task's remaining reservation, and every queued overflow subtask
        (reserved when its batch was admitted, not yet a task)."""
        total = float(self.colony.starting_budget - self.colony.budget_remaining)
        for task_id, task in self.task_graph.tasks.items():
            if task.status in (0, 1):
                total += self._remaining_reservation(task_id)
        for queue in self.pending_overflow.values():
            for spawn_kwargs in queue:
                total += self._new_task_reservation(
                    self._reservation_kind(spawn_kwargs.get("role")))
        return total

    def _admits(self, extra: float = 0.0) -> bool:
        """The admission rule, evaluated with any hypothetical change already
        applied to task_reservations and `extra` for work not yet created."""
        if not self.ADMISSION_CONTROL:
            return True
        limit = self.colony.starting_budget - self.ADMISSION_FLOOR
        return self.committed_energy() + extra <= limit

    def _admit_spawn_batch(self, parent_task_id: str, child_roles: list) -> int:
        """How many of this batch's children (in order) may be created: the
        largest k whose reservation fits. 0 means not even one. On success
        the parent's reservation is re-priced at that k."""
        res = self._reservation(parent_task_id)
        saved = (res.k, res.spawned)
        extras = [self._new_task_reservation(self._reservation_kind(r)) for r in child_roles]
        for k in range(len(child_roles), 0, -1):
            res.k, res.spawned = k, True
            if self._admits(sum(extras[:k])):
                return k
        res.k, res.spawned = saved
        return 0

    def _task_energy_records(self) -> list:
        """One record per task: what the RESERVE_* figures are re-fitted
        from (sims/reservation_formula/refit_from_runs.py). A task that
        completed on its first attempt with no conversion is one clean sample
        of what an attempt of its kind -- at its k -- costs."""
        records = []
        for task_id, task in self.task_graph.tasks.items():
            res = self._reservation(task_id)
            records.append({
                "task_id": task_id,
                "kind": res.kind,
                "k": res.k if res.spawned else None,
                "attempts": res.attempts,
                "converted": res.conv_sunk > 0,
                "conv_sunk": res.conv_sunk,
                "spent": self.colony.task_energy_spent.get(task_id, 0),
                "ceiling": self.task_energy_ceiling(task_id),
                "status": task.status,
                "abandon_reason": self.abandon_reasons.get(task_id),
                "description": (task.description or "")[:80],
            })
        return records

    def _write_energy_trace(self, outcome: str) -> None:
        """Append this run's energy picture as one JSON line to
        energy_trace_path. Off (None) unless set -- tests call terminate()."""
        if not self.energy_trace_path:
            return
        import json
        line = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "outcome": outcome,
            "budget": self.colony.starting_budget,
            "spent": self.colony.starting_budget - self.colony.budget_remaining,
            "ticks": self.tick_count,
            "budget_source": "override" if self.budget_override is not None else "phaser",
            "params": {
                "e": self.RESERVE_EXECUTOR, "alpha": self.RESERVE_DECOMPOSER_BASE,
                "beta": self.RESERVE_PER_CHILD, "sigma": self.RESERVE_ROOT_EXTRA,
                "A": self.TASK_CEILING_ATTEMPTS, "floor": self.ADMISSION_FLOOR,
                "admission": self.ADMISSION_CONTROL,
            },
            "verdicts": dict(self.colony.verdict_counts),
            "ledger": dict(self.colony.energy_ledger),
            "tasks": self.run_trace.get("task_energy", []),
        }
        try:
            with open(self.energy_trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")
            print(f"[energy-trace] appended this run to {self.energy_trace_path}")
        except Exception as e:
            print(f"Warning: failed writing energy trace to {self.energy_trace_path}: {e}")

    def _admit_conversion(self, task_id: str) -> bool:
        """Re-price a TASK TOO LARGE task as a decomposer and admit it or not.
        The executor's spend so far is carried as sunk (and conv_sunk, which
        the ceiling adds on top) instead of eating the decomposer's share."""
        res = self._reservation(task_id)
        saved = (res.kind, res.k, res.spawned, res.sunk, res.conv_sunk)
        spent = self.colony.task_energy_spent.get(task_id, 0)
        res.kind, res.k, res.spawned = "decomposer", 0, False
        res.sunk = res.conv_sunk = spent
        if self._admits():
            self._admitted_conversions.add(task_id)
            return True
        res.kind, res.k, res.spawned, res.sunk, res.conv_sunk = saved
        self.colony.record_verdict("admission_conversion_refused")
        return False

    def _refuse_spawn_for_budget(self, parent_id: str, spawner) -> None:
        """Not even one child fits. Handled exactly like a batch whose
        children could not be started for lack of energy: a decomposer with
        finished children is released to REPORT them, one without goes down
        the DIE path (where respawn admission decides whether anything more
        is spent on the task)."""
        print(f"  [admission] {parent_id}'s SPAWN refused: not one subtask fits "
              f"the uncommitted budget (committed "
              f"{self.committed_energy():.0f} against a limit of "
              f"{self.colony.starting_budget - self.ADMISSION_FLOOR}: budget "
              f"{self.colony.starting_budget} minus the {self.ADMISSION_FLOOR} floor).")
        self.colony.record_verdict("admission_spawn_refused")
        if self.live_agents.get(parent_id) is not None:
            self._release_spawner_if_nothing_started(parent_id, out_of_energy=True)
            return
        spawner.fail_reason = ("SPAWN refused: the colony has no uncommitted "
                               "energy left to start any subtask.")
        self.messenger.push_event(
            "failure_request",
            parent_id,
            {
                "task_id": spawner.task_id,
                "role": spawner.role,
                "parent_id": spawner.parent_id,
                "result": "No energy budget left to decompose this task.",
            },
        )

    def _tell_parent_about_dropped_subtasks(self, parent_id: str, dropped: list) -> None:
        """Truncation is only honest if the decomposer knows: its roll-up
        must say what was not covered instead of presenting a partial plan as
        the whole one. Delivered into its context like a child result, and
        billed the same way."""
        for _ in dropped:
            self.colony.record_verdict("admission_subtasks_dropped")
        names = "; ".join(f'"{d[:80]}"' for d in dropped)
        print(f"  [admission] {parent_id}: dropped {len(dropped)} subtask(s) the "
              f"budget cannot cover: {names}")
        live_parent = self.live_agents.get(parent_id)
        if live_parent is None:
            return
        note = (f"\n[BUDGET] These subtasks were NOT started because the colony "
                f"cannot afford them: {names}. Your final REPORT must say they "
                f"were not covered.\n")
        live_parent.thought_process += note
        self.colony.debit_energy(parent_id, max(1, len(note) // 100), category="injection")

    def _energy_ceiling_reason(self, spent: int, ceiling: int) -> str:
        """The abandonment reason line, shared by both ceiling checks so a
        task stopped mid-agent and one stopped at a respawn decision read
        identically in the log and in the partial handed to the parent."""
        return (f"its agents used {spent} energy, at or over the per-task "
                f"ceiling of {ceiling}")

    def _task_energy_overrun(self, task_id: Optional[str]):
        """(spent, ceiling) if this task is at or over its energy ceiling,
        else None."""
        if not task_id:
            return None
        spent = self.colony.task_energy_spent.get(task_id, 0)
        ceiling = self.task_energy_ceiling(task_id)
        return (spent, ceiling) if spent >= ceiling else None

    def _enforce_task_energy_ceiling(self, agent_id: str,
                                     task_id: Optional[str]) -> bool:
        """Stop a task that crossed its energy ceiling mid-agent.

        The ceiling used to be consulted only where MAX_TASK_ATTEMPTS is,
        on the respawn decision -- so a task whose agent kept going without
        reaching one had no boundary to be caught at, and sailed past the
        line uncaught (task_ac3e5d85).

        Called from both places an agent is let continue after a cycle:
          * _run_live_agents, for a cycle that reached no terminal action
            at all -- a THINK, a refused TOOL/SPAWN, or a crash under
            MAX_CONSECUTIVE_CRASHES;
          * handle_completion's WARN branch, for a REPORT the judge sent
            back, which is the loop that actually produced the overshoot;
          * handle_parent_notification, right after a child's result is
            billed to its waiting parent -- the one debit that lands on an
            agent _run_live_agents is not ticking.
        Between them the overshoot is bounded by one cycle rather than by
        one whole agent's remaining budget.

        Ends in the same abandon-and-release-dependents path the attempt
        cap uses: the task is marked failed, its best partial goes to the
        parent with a marker, and its dependents are released.

        Returns True if the task was abandoned (the caller must stop
        touching this agent).
        """
        if not task_id or task_id in self.abandoned_tasks:
            return False
        task_node = self.task_graph.tasks.get(task_id)
        if task_node is None or task_node.status != 1:
            # Already completed, failed or not running: nothing to stop,
            # and abandoning it would overwrite a real disposition.
            return False
        overrun = self._task_energy_overrun(task_id)
        if overrun is None:
            return False
        spent, ceiling = overrun

        # Read off the colony node before _retire_agent unregisters it.
        agent_node = self.colony.get_agent(agent_id)
        parent_id = getattr(agent_node, "parent_id", None) if agent_node else None
        attempts = self.respawn_counts.get(task_id, 0)

        print(f"  [energy-ceiling] {agent_id} took task {task_id} to {spent} "
              f"energy against a ceiling of {ceiling} without reaching a "
              f"respawn decision -- stopping it here.")
        self.colony.record_verdict("energy_ceiling_midagent")
        self._retire_agent(
            agent_id, task_id,
            verdict={"verdict": "execute",
                     "reason": self._energy_ceiling_reason(spent, ceiling)},
        )
        self._abandon_task(
            task_id, parent_id, agent_id, attempts,
            reason=self._energy_ceiling_reason(spent, ceiling),
            reason_label="energy ceiling",
        )
        return True

    def task_energy_ceiling(self, task_id: Optional[str] = None) -> int:
        """Most energy this task's agents may spend, all attempts together:
        conv_sunk + TASK_CEILING_ATTEMPTS * own_cost. See __init__.

        Priced at the task's OWN cost -- role and batch size -- because a
        decomposer's attempt grows with k and the root's carries sigma; one
        flat figure fitted only executors. With no task_id, the executor
        ceiling (the reference figure the ledger prints).

        A decomposer is priced at _ceiling_k, never below its fan-out cap,
        so its ceiling is always above an executor's. See __init__.
        """
        res = (self._reservation(task_id) if task_id
               else TaskReservation(kind="executor"))
        if res.kind != "executor":
            res = replace(res, k=self._ceiling_k(res), spawned=True)
        return int(res.conv_sunk + self.TASK_CEILING_ATTEMPTS * self._own_cost(res))

    def _ceiling_k(self, res: TaskReservation) -> int:
        """The batch size a decomposer's ceiling is priced at: its fan-out
        cap, or the largest batch it has had admitted if that is bigger
        (queued overflow counts). Never the k=1 admission prices an
        unspawned decomposer at, and never lower after a respawn resets k."""
        cap = (self.MAX_SUBTASKS_ROOT if res.kind == "root"
               else self.MAX_SUBTASKS_NON_ROOT)
        return max(cap, res.k_peak, res.k if res.spawned else 0)

    def _abandon_task(self, task_id: str, parent_id: Optional[str],
                       agent_id: Optional[str], attempts: int,
                       reason: Optional[str] = None,
                       reason_label: str = "attempt cap"):
        """
        Terminal disposition for a task that has burned through
        MAX_TASK_ATTEMPTS respawns without ever satisfying the judge.

        Deliberately NOT task_graph.fail_task: that cascades status=3 down
        every dependent, which throws away work that could still complete
        from a partial input. This marks only this task failed, hands the
        parent the best partial result it produced with an explicit
        abandonment marker so the parent can reason about the gap instead of
        stalling on `awaiting`, and releases the dependents the same way a
        completion would.
        """
        self.abandoned_tasks.add(task_id)
        self.abandon_reasons[task_id] = reason_label
        task_node = self.task_graph.tasks.get(task_id)
        description = task_node.description if task_node is not None else "<unknown task>"

        partial = self.last_partial_result.get(task_id)
        if partial is None and task_node is not None:
            partial = task_node.result
        if partial is None:
            partial = self.colony.results.get(task_id)
        partial_text = str(partial).strip() if partial else ""

        marker = (
            f"[ABANDONED after {attempts + 1} attempt(s) -- {reason}.]"
            if reason else
            f"[ABANDONED after {attempts + 1} attempt(s) -- no agent assigned to "
            f"this subtask produced an acceptable result, so it was stopped "
            f"rather than recycled further.]"
        )
        if partial_text:
            abandoned_result = (
                f"{marker}\nBest partial output produced before abandonment "
                f"(unverified -- it was rejected by review at least once):\n"
                f"{partial_text}"
            )
        else:
            abandoned_result = f"{marker}\nNo usable output was produced for this subtask."

        print("=" * 62)
        print(f"TASK ABANDONED: {task_id} -- \"{description}\"")
        if reason:
            print(f"  After {attempts + 1} attempt(s): {reason}. Marking failed "
                  f"and releasing dependents.")
        else:
            print(f"  {attempts + 1} attempt(s) exhausted (cap MAX_TASK_ATTEMPTS="
                  f"{self.MAX_TASK_ATTEMPTS}). Marking failed and releasing dependents.")
        print(f"  Salvaged partial: {'yes' if partial_text else 'none'}")
        print("=" * 62)

        if task_node is not None:
            task_node.status = 3
            task_node.result = abandoned_result
        self.colony.store_result(task_id, abandoned_result)

        self._release_dependents(task_node)
        self._thread_results_to_unblocked_dependents(task_node)

        # A freed fan-out slot is a freed slot whether the child succeeded or
        # was abandoned -- otherwise a decomposer's queued overflow never
        # spawns at all once one of its children dies permanently.
        self._drain_pending_overflow(parent_id)

        if parent_id:
            self.messenger.push_event(
                "parent_notification",
                "orchestrator",
                {"parent_id": parent_id, "child_id": agent_id, "task_id": task_id,
                 "result": abandoned_result}
            )

        if task_id == self.root_task_id:
            # Nothing further can be attempted on the root. tick() sees
            # status==3 and stops the loop, which lands in terminate()'s
            # partial-synthesis path instead of spinning to energy death.
            print("  The ABANDONED task is the ROOT task -- ending the run and "
                  "synthesizing whatever subtasks did complete.")

    @staticmethod
    def _summarize_tool_result(result) -> str:
        """
        The single string an agent actually gets to read about its tool call.

        The previous `result.get("data") or result.get("message") or str(result)`
        had two holes that both ended with the agent unable to act:
          - a successful run whose script printed nothing has data == "",
            which is falsy, so the agent was handed the raw dict repr;
          - an error carried its "reason" ("script_crash", "timeout",
            "output_overflow") only inside that repr, so a condensed
            TracebackSummarizer line arrived with no marker saying it WAS a
            failure -- easy for the model to read as ordinary output.
        Status is now stated explicitly and the reason is kept alongside the
        message.
        """
        if not isinstance(result, dict):
            return str(result)

        status = result.get("status")
        if status == "success":
            data = result.get("data")
            if data is None:
                data = result.get("message")
            if data is None or str(data).strip() == "":
                return ("SUCCESS: the tool ran without error but produced no "
                        "output. If you expected output, your code did not "
                        "print anything -- add a print() of the value you "
                        "need.")
            return f"SUCCESS: {data}"

        reason = result.get("reason")
        message = result.get("message") or result.get("data") or str(result)
        reason_str = f" ({reason})" if reason else ""
        return f"ERROR{reason_str}: {message}"

    def _tool_request_allowed(self, event: Event) -> bool:
        """
        Returns True if this tool request may run, False if it was refused.

        Decomposers are not offered TOOL in their prompt (see
        agent_node.role_may_use_tools); prompt-only enforcement of a
        structural rule does not hold (see handle_spawn), so it is refused
        here too. The refusal is NOT routed through receive_tool_result: the
        tool never ran, so it must not count toward the agent's
        consecutive-tool-failure circuit breaker.
        """
        agent_id = event.from_agent
        live_agent = self.live_agents.get(agent_id)
        colony = getattr(self, "colony", None)
        record = colony.get_agent(agent_id) if colony is not None and agent_id else None
        role = live_agent.role if live_agent is not None else getattr(record, "role", None)
        # Orchestrator-originated requests have no agent record to check.
        if role is None or role_may_use_tools(role):
            return True

        tool_name = event.payload.get("tool_name")
        print(f"REJECT (structural) on {agent_id}: a '{role}' agent requested "
              f"TOOL '{tool_name}', which its role may not call -- not executed.")
        if live_agent is not None:
            live_agent.fail_reason = (
                f"Previous attempt was REJECTED: you issued TOOL '{tool_name}', "
                f"but a {role} never calls tools -- it SPAWNs the work to "
                f"executors. SPAWN a subtask for it, or REPORT/DIE."
            )
        return False

    def handle_tool_request(self, event: Event):
        """Handles external tool execution requests from agents."""
        payload = event.payload
        agent_id = event.from_agent
        tool_name = payload.get("tool_name")
        args = payload.get("args", {}) or {}
        domain = self.spec.get("domain", "General Discourse") if self.spec else "General Discourse"

        if not self._tool_request_allowed(event):
            return

        print(f"Executing tool '{tool_name}' for agent {agent_id}...")

        result = ToolRegistry.execute(tool_name, args, domain=domain, agent_id=agent_id)

        output_type = "text"
        if tool_name == "run_code":
            output_type = "code_result"
        elif tool_name == "verify_math":
            output_type = "math_result"

        fast_result = {"pass": True, "error": None}
        if self.judge is not None:
            fast_result = self.judge.fast_check(result, output_type)

        summary = self._summarize_tool_result(result)

        # Whether the call succeeded is decided by the tool's own status
        # first, and only then narrowed by the judge. Reading it off
        # fast_result alone had two failure modes that both told the agent an
        # error was a success -- clearing its fail_reason and (since the
        # consecutive-failure circuit breaker counts these) letting a call
        # that fails every time run to the 15-attempt ceiling:
        #   - with no judge configured, fast_result was hardcoded to pass;
        #   - for any tool other than run_code/verify_math, output_type is
        #     "text", and fast_check's text branch only asks "is this
        #     non-empty?" -- an error dict is non-empty, so a failed
        #     write_file/safe_read_file/query_dataframe passed.
        tool_ok = not (isinstance(result, dict) and result.get("status") == "error")
        success = tool_ok and bool(fast_result["pass"])

        live_agent = self.live_agents.get(agent_id)
        if live_agent is not None:
            live_agent.receive_tool_result(tool_name, summary, success)

        self.messenger.push_event(
            "tool_response",
            "orchestrator",
            {"agent_id": agent_id, "result": result, "tool_name": tool_name}
        )

    def merge_agents(self):
        """MERGE: Consolidation mechanism to survive low-energy bottlenecks."""
        print("System strain detected! Consolidating idle / overlapping agents...")
        culled_ids = self.colony.consolidate_idle_agents()

        for agent_id in culled_ids:
            live_agent = self.live_agents.pop(agent_id, None)
            if live_agent is not None:
                live_agent.KV_Cache = None
                live_agent.last_hidden_state = None

        print(f"Consolidated {len(culled_ids)} idle agents and reclaimed energy.")

    def handle_deadlock(self):
        """
        Runs systemic watchdog checks to detect and resolve system bottlenecks or silent agents.
        """
        ready_tasks = self.task_graph.get_ready_tasks()
        root_task = self.task_graph.tasks.get(self.root_task_id)
        root_incomplete = root_task is not None and root_task.status != 2

        if self.messenger.peek() == 0 and len(ready_tasks) == 0 and root_incomplete:
            now = time.time()
            running_tasks = [t for t in self.task_graph.tasks.values() if t.status == 1]
            
            active_agents = []
            stuck_agents = []

            tickable_agent_count = sum(
                1 for t in running_tasks
                if t.agent_id and self.colony.get_agent(t.agent_id)
                and self.colony.get_agent(t.agent_id).awaiting is None
            )

            for task in running_tasks:
                if task.agent_id:
                    agent = self.colony.get_agent(task.agent_id)
                    if agent:
                        if agent.awaiting is not None:
                            continue

                        effective_timeout = self.timeout_threshold * max(1, tickable_agent_count)
                        silence_duration = now - agent.last_active
                        if silence_duration > effective_timeout:
                            stuck_agents.append((agent, task))
                        else:
                            active_agents.append(agent)

            if not stuck_agents and len(active_agents) > 0:
                return

            print("System deadlock confirmed! Investigating state nodes...")

            if stuck_agents:
                for agent, task in stuck_agents:
                    print(f"Deadlock Watchdog: Agent {agent.agent_id} on Task {task.task_id} "
                          f"is non-responsive for {now - agent.last_active:.2f}s. Forcing respawn.")
                    # FIX (#2, wrong-agent-killed bug): this previously
                    # pushed with from_agent="orchestrator" and put the real
                    # target in payload["agent_id"] -- but handle_failure
                    # reads its target from event.from_agent, not from the
                    # payload. That meant the watchdog was always calling
                    # _kill_and_respawn("orchestrator", ...): a no-op pop/
                    # unregister (nothing registered under that id), so the
                    # actually-stuck agent was never removed from
                    # live_agents/the colony, and a duplicate agent got
                    # spawned onto the same task alongside it. Setting
                    # from_agent to the real agent id matches the
                    # self-reported-DIE convention handle_failure already
                    # expects.
                    self.messenger.push_event(
                        "failure_request",
                        agent.agent_id,
                        {
                            "task_id": task.task_id,
                            "role": agent.role,
                            "parent_id": agent.parent_id
                        }
                    )
            else:
                pending_unassigned_tasks = [
                    t for t in self.task_graph.tasks.values() 
                    if t.status == 0 and t.agent_id is None
                ]
                if pending_unassigned_tasks:
                    print(f"Deadlock Watchdog: Recovered {len(pending_unassigned_tasks)} unassigned pending tasks. Spawning workers...")
                    for task in pending_unassigned_tasks:
                        self.spawn_agent(role=task.required_role, task_id=task.task_id)
                else:
                    print("Deadlock Watchdog: System is stable. No actionable stalls detected.")

    def tick(self) -> bool:
        """Processes a single heartbeat of the orchestrator loop."""
        self.tick_count += 1
        if self.tick_count % self.LEDGER_EVERY_TICKS == 0:
            self._print_energy_report(header=f"MID-RUN tick {self.tick_count}")
        rem_energy = self._check_energy()
        status = self._get_energy_status(rem_energy)
        self.peak_live_agents = max(self.peak_live_agents, len(self.live_agents))

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                print(f"[VRAM:tick] allocated={allocated:.1f}MB reserved={reserved:.1f}MB "
                      f"live_agents={len(self.live_agents)}")
        except Exception:
            pass
        
        if status == "death":
            print(f"Critical failure. Colony energy depleted ({rem_energy} remaining).")
            return False
        elif status == "stressed":
            self.merge_agents()

        self.handle_deadlock()

        events = self.messenger.drain()
        self._route_events(events)
        self._process_unblocked_tasks()

        self._run_live_agents()
        
        root_task = self.task_graph.tasks.get(self.root_task_id)
        # status 3 as well as 2: the root can now be ABANDONED by the
        # attempt cap, and there is nothing left to tick once it is.
        if root_task and root_task.status in (2, 3):
            return False
    
        return True

    def _print_energy_report(self, header: str = "final"):
        """Prints the energy ledger, credits, reconciliation, and respawn counts.

        A1 deliverable: answers "where did the budget go, and which tasks
        cycled?" at the end of every run. Read-only -- it must never be able
        to change the outcome of a run, hence the broad guard at the bottom.
        """
        try:
            ledger = dict(getattr(self.colony, "energy_ledger", {}) or {})
            credits = dict(getattr(self.colony, "energy_credits", {}) or {})
            start = getattr(self.colony, "starting_budget", None)
            remaining = getattr(self.colony, "budget_remaining", 0)

            total_debits = sum(ledger.values())
            total_credits = sum(credits.values())

            print(chr(10) + "=" * 62)
            print(f"ENERGY LEDGER [{header}] (where the budget went)")
            print("=" * 62)

            if not ledger:
                print("  (no energy was debited this run)")
            else:
                print(f"  {'category':<20}{'total':>10}{'% of start':>14}")
                for category, total in sorted(ledger.items(), key=lambda kv: -kv[1]):
                    pct = f"{(100.0 * total / start):.1f}%" if start else "n/a"
                    print(f"  {category:<20}{total:>10}{pct:>14}")
                print(f"  {'-' * 44}")
                print(f"  {'TOTAL DEBITED':<20}{total_debits:>10}")

            print("\n  CREDITS (refunds)")
            if not credits:
                print("    (none)")
            else:
                for category, total in sorted(credits.items(), key=lambda kv: -kv[1]):
                    print(f"    {category:<18}{total:>10}")
                print(f"    {'TOTAL CREDITED':<18}{total_credits:>10}")

            # These three must agree. If they don't, there is an energy path
            # writing budget_remaining directly instead of going through
            # debit_energy/credit_energy -- find it before tuning anything.
            expected = (start - total_debits + total_credits) if start is not None else None
            print("\n  RECONCILIATION")
            print(f"    starting budget            : {start}")
            print(f"    start - debits + credits   : {expected}")
            print(f"    colony.budget_remaining    : {remaining}")
            if expected is None:
                print("    [WARN] no starting_budget recorded -- cannot reconcile.")
            elif expected != remaining:
                print(f"    [MISMATCH] delta = {remaining - expected} -- there is an "
                      f"energy path bypassing debit_energy/credit_energy.")
            else:
                print("    [OK] ledger reconciles with the remaining budget.")

            print(chr(10) + "  TIER-3 VERDICTS (deep_critique accept/reject)")
            verdicts = dict(getattr(self.colony, "verdict_counts", {}) or {})
            accepts = verdicts.get("tier3_accept", 0)
            rejects = verdicts.get("tier3_reject", 0)
            tier3_total = accepts + rejects
            if tier3_total == 0:
                print("    (deep_critique never ran this run)")
            else:
                rate = 100.0 * accepts / tier3_total
                print(f"    accept : {accepts}")
                print(f"    reject : {rejects}")
                print(f"    accept rate : {accepts}/{tier3_total} ({rate:.1f}%)")

            print(chr(10) + "  SUCCESS CACHE (outcome-gated)")
            neg = {k[len("cache_write_negative_"):]: v for k, v in verdicts.items()
                   if k.startswith("cache_write_negative_")}
            print(f"    positive writes             : {verdicts.get('cache_write_positive', 0)}")
            print(f"    negative writes             : {sum(neg.values())} {neg if neg else ''}")
            print(f"    servable hits               : {verdicts.get('cache_hit_served', 0)}")
            print(f"      of which cross-task donor : {verdicts.get('cache_hit_served_cross_task', 0)}")
            print(f"    misses on a negative match  : {verdicts.get('cache_miss_negative_match', 0)}")
            print(f"    cross-task below threshold  : {verdicts.get('cache_cross_task_below_threshold', 0)}")
            # The root is never served from the cache (see
            # _note_root_cache_lookup). The second line is what a hit would
            # have completed the run with, had it been allowed to.
            print(f"    root lookups not served     : {verdicts.get('root_cache_lookup_skipped', 0)}")
            print(f"      would have hit            : {verdicts.get('root_cache_hit_withheld', 0)}")
            # PATCH 33, measure-only: nothing found here reaches a prompt.
            ghost_avail = verdicts.get('ghost_lessons_available', 0)
            ghost_total = ghost_avail + verdicts.get('ghost_lessons_none', 0)
            print(f"    ghost memory: respawns with earlier lessons available : "
                  f"{ghost_avail} of {ghost_total}")

            print(chr(10) + "  CYCLE CAP (Agent.MAX_NON_TERMINAL_CYCLES)")
            # Read these three together. "decisions at the cap" is the number
            # that proves enforcement ran: each one is a decide() whose menu
            # had THINK/SPAWN stripped. "forced" is the subset where the model
            # ignored that menu and the backstop overrode it -- so 0 forced
            # with decisions > 0 is the strip WORKING, not failing. Reached
            # with 0 decisions means every capped agent was respawned before
            # it made one (the WARN-at-cap path), so the menu never came up.
            # Decisions by a decomposer whose SPAWN was closed for lack of
            # energy (spawn_closed) run the same reduced menu and count here.
            decisions = verdicts.get('cycle_cap_decisions', 0)
            coerced = verdicts.get('cycle_cap_coerced', 0)
            print(f"    agents that reached it   : {verdicts.get('cycle_cap_reached', 0)}")
            print(f"    decisions at the cap     : {decisions}")
            print(f"      model kept to the menu : {max(0, decisions - coerced)}")
            print(f"      actions forced by the cap: {coerced}")
            # A WARN is a non-terminal cycle too (see handle_completion). These
            # two lines are what says whether REPORT -> WARN -> REPORT is still
            # running long: warns far above respawns means agents are revising
            # and converging, warns clustered just under the cap per agent
            # means they are looping and being cut off.
            warns = verdicts.get('judge_warn', 0)
            print(f"    judge WARNs issued      : {warns}")
            print(f"    warn loops cut at the cap: "
                  f"{verdicts.get('warn_cycle_cap_respawn', 0)}")

            print(chr(10) + "  SOFTWARE FRAMING (code-shaped work on a non-software request)")
            if not self._software_framing_guard_active:
                print("    (guard off -- the request asked for software, or there was none)")
            else:
                print(f"    levers on: {sorted(self.framing_levers) or 'none (baseline)'}")
                print("    detected (counted whatever the levers):")
                print(f"      subtasks in code-coded wording : {verdicts.get('software_framing_subtask_codeword', 0)}")
                print(f"      code-shaped subtasks           : {verdicts.get('software_framing_spawn_detected', 0)}")
                print(f"      code-shaped THINK cycles       : {verdicts.get('software_framing_think_detected', 0)}")
                print(f"      code-shaped REPORTs            : {verdicts.get('software_framing_report_detected', 0)}")
                print(f"      code-shaped DIEs               : {verdicts.get('software_framing_die_detected', 0)}")
                print("    acted on:")
                print(f"      subtasks reworded      : {verdicts.get('software_framing_subtask_reworded', 0)}")
                print(f"      SPAWN batches rejected : {verdicts.get('software_framing_spawn_rejected', 0)}")
                print(f"      REPORTs rejected       : {verdicts.get('software_framing_report_rejected', 0)}")
                print(f"      DIE reasons withheld   : {verdicts.get('software_framing_die_scrubbed', 0)}")

                print("    software-shaped TASKS (PATCH 10, measure-only):")
                print(f"      subtasks that ASK for software : "
                      f"{verdicts.get('software_shaped_task_detected', 0)}")
                print("      (counted, never blocked this round -- these are "
                      "subtasks whose own text uses the vocabulary that would "
                      "have switched the guard on had the USER written it)")

            # PATCH 7. The distribution, so a threshold can be chosen from
            # data rather than guessed. Nothing in this run was gated on any
            # of these numbers.
            print(chr(10) + "  ARTIFACT MANUFACTURING (PATCH 15, measure-only)")
            if not self._artifact_guard_active:
                print("    (guard off -- the request named a document or deliverable, or there was none)")
            else:
                artifact_hits = verdicts.get("artifact_shaped_task_detected", 0)
                both = verdicts.get("artifact_shaped_task_also_drift_dropped", 0)
                print(f"      subtasks that ASK for a document : {artifact_hits}")
                print(f"      of those, also dropped by the drift gate : {both}")
                print(f"      caught ONLY by this check : {artifact_hits - both}"
                      f"  (the cases PATCH 12 cannot see)")
                print("      (counted, never blocked this round -- PATCH 12's gate "
                      "went live in the same run, and two new gates firing at "
                      "once leaves no way to attribute what changed)")

            print(chr(10) + "  GOAL DRIFT AT SPAWN (PATCH 7 measures; PATCH 12's gate "
                  + ("ACTING" if GOAL_DRIFT_GATE_ACTS else "MEASURE-ONLY") + ")")
            samples = getattr(self, "goal_drift_samples", []) or []
            if not samples:
                print("    (no subtasks spawned this run)")
            else:
                goal_vals = [s["goal_drift"] for s in samples if s["goal_drift"] is not None]
                parent_vals = [s["parent_drift"] for s in samples if s["parent_drift"] is not None]
                gated = [s for s in samples if s.get("gated")]
                verb = ("dropped by the gate" if GOAL_DRIFT_GATE_ACTS
                        else "below threshold, started anyway")
                spawned = len(samples) - (len(gated) if GOAL_DRIFT_GATE_ACTS else 0)
                print(f"    subtasks measured : {len(goal_vals)} of {len(samples)} "
                      f"scored ({spawned} spawned, {len(gated)} {verb})")
                print(f"    gate : below {GOAL_DRIFT_GATE_FRACTION:g} x the run's "
                      f"observed max goal-cosine"
                      + (f" (= {GOAL_DRIFT_GATE_FRACTION * max(goal_vals):.3f} "
                         f"against a max of {max(goal_vals):.3f})" if goal_vals else "")
                      + f", once {GOAL_DRIFT_GATE_MIN_SAMPLES} scores exist")
                print(f"      subtasks {'dropped        ' if GOAL_DRIFT_GATE_ACTS else 'that WOULD drop'} : "
                      f"{verdicts.get('goal_drift_gate_dropped', 0)}")
                print(f"      whole batches spared    : "
                      f"{verdicts.get('goal_drift_gate_whole_batch_spared', 0)}"
                      f"  (every subtask under threshold -- counted, not dropped)")
                unmeasurable = verdicts.get("spawn_drift_goal_unmeasurable", 0)
                if unmeasurable:
                    # All-or-nothing in practice: a zero-norm goal vector is
                    # the phaser's np.zeros fallback, which affects the whole
                    # run, not one task.
                    print(f"    goal-cosine unmeasurable : {unmeasurable} "
                          f"(missing or zero-norm vector -- see "
                          f"problem_phaser.py:601/:636)")
                clause_vals = [s["clause_drift"] for s in samples
                               if s.get("clause_drift") is not None]
                for name, vals in (("cosine to GOAL   (cumulative)", goal_vals),
                                   ("cosine to CLAUSE (max, P17)  ", clause_vals),
                                   ("cosine to PARENT (per hop)   ", parent_vals)):
                    if not vals:
                        print(f"    {name} : (none measurable)")
                        continue
                    print(f"    {name} : min={min(vals):.3f}  "
                          f"median={statistics.median(vals):.3f}  "
                          f"max={max(vals):.3f}  n={len(vals)}")
                if goal_vals:
                    print("    5 lowest by goal-cosine (furthest from the project):")
                    ranked = sorted(
                        (s for s in samples if s["goal_drift"] is not None),
                        key=lambda s: s["goal_drift"],
                    )[:5]
                    for s in ranked:
                        description = " ".join(str(s["description"] or "").split())
                        if len(description) > 58:
                            description = description[:55] + "..."
                        parent = ("n/a" if s["parent_drift"] is None
                                  else f"{s['parent_drift']:.3f}")
                        # Both numbers on one row on purpose: a low goal
                        # cosine next to a HIGH parent cosine is the
                        # gradual-drift signature (every hop looked fine),
                        # and is a different problem from a low pair.
                        mark = "  [GATED]" if s.get("gated") else ""
                        print(f"      goal={s['goal_drift']:.3f}  parent={parent}  "
                              f"{s['task_id']}  {description}{mark}")
                self._print_clause_reference_comparison(samples)

            print(chr(10) + "  FIGURE DIVERGENCE ACROSS ATTEMPTS (PATCH 16, measure-only)")
            print(f"    same task, consecutive attempts, figures that did "
                  f"not carry over : {verdicts.get('figure_divergence_detected', 0)}")
            print("      (both attempts must assert at least "
                  f"{FIGURE_DIVERGENCE_MIN} figures and share under "
                  f"{FIGURE_DIVERGENCE_MAX_OVERLAP:.0%} of them; a figure-free "
                  "attempt is vague, not contradictory, and is not compared)")

            self._print_fidelity_report()
            self._print_data_free_report()

            print(chr(10) + "  STRUCTURAL REJECTS (caught before the judge)")
            print(f"    decomposer REPORT, no completed subtask : "
                  f"{verdicts.get('decomposer_report_no_completed_child', 0)}")
            print(f"    REPORT was the prompt's worked example  : "
                  f"{verdicts.get('report_exemplar_echo_rejected', 0)}")
            print(f"    REPORT echoed the prompt header (PATCH 14) : "
                  f"{verdicts.get('report_prompt_header_echo_detected', 0)}")
            print(f"      cut from the REPORT, answer kept      : "
                  f"{verdicts.get('report_prompt_header_echo_cut', 0)}")
            print(f"      REPORT was header from char 0, judged : "
                  f"{verdicts.get('report_prompt_header_echo_whole', 0)}")
            print(f"    REPORT was the empty-generation placeholder : "
                  f"{verdicts.get('report_empty_generation_rejected', 0)}")
            print(f"    zero-requirement tasks given the goal referent : "
                  f"{verdicts.get('goal_referent_attached', 0)}")

            print("\n  RESPAWNS (top 10 by count)")
            respawns = getattr(self, "respawn_counts", {}) or {}
            if not respawns:
                print("    (no respawns this run)")
            else:
                for task_id, count in sorted(respawns.items(), key=lambda kv: -kv[1])[:10]:
                    task = self.task_graph.tasks.get(task_id)
                    status = task.status if task is not None else "?"
                    description = (task.description or "") if task is not None else "<unknown task>"
                    if len(description) > 60:
                        description = description[:57] + "..."
                    print(f"    {count:>3}x  status={status}  {task_id}  {description}")

            print(f"\n  TASK ENERGY (top 10 by energy; per-task ceiling = "
                  f"conversion carry-over + {self.TASK_CEILING_ATTEMPTS} x the "
                  f"task's own attempt cost, executor "
                  f"{self.task_energy_ceiling()}; OVER next to a reason = "
                  f"stopped as intended)")
            task_energy = dict(getattr(self.colony, "task_energy_spent", {}) or {})
            if not task_energy:
                print("    (no energy attributed to a task this run)")
            else:
                reasons = getattr(self, "abandon_reasons", {}) or {}
                # A REPORT/DIE is judged a tick after the cycle that produced
                # it, and that cycle is exempt from the per-cycle check -- so a
                # mid-run ledger can catch a task over its line with its
                # verdict still queued. That is the one-cycle window, not a
                # leak: it is stopped (or kept, if promoted) when routed.
                queued_verdicts = {
                    (e.payload or {}).get("task_id")
                    for e in self.messenger.pending()
                    if e.type in ("completion_request", "failure_request")
                }
                for task_id, spent in sorted(task_energy.items(), key=lambda kv: -kv[1])[:10]:
                    task = self.task_graph.tasks.get(task_id)
                    status = task.status if task is not None else "?"
                    agents = 1 + respawns.get(task_id, 0)
                    # OVER is expected on a task the ceiling stopped: the
                    # check runs after a cycle's energy is debited, so a task
                    # always crosses the line before it can be caught -- by
                    # one cycle's spend now that the check is per-cycle,
                    # rather than by a whole agent's.
                    # OVER with nothing that stopped it is the real signature
                    # -- a task past its ceiling that is still being funded.
                    # A promoted REPORT that landed over the line is kept,
                    # not stopped -- labelled as such so it neither hides
                    # under a plain status=2 nor reads as a leak.
                    flag = ""
                    ceiling = self.task_energy_ceiling(task_id)
                    if spent >= ceiling:
                        if task_id in reasons:
                            flag = "  OVER"
                        elif task_id in self.completed_over_ceiling:
                            flag = "  OVER -- COMPLETED OVER CEILING"
                        elif task_id in queued_verdicts:
                            flag = "  OVER -- VERDICT PENDING"
                        else:
                            flag = "  OVER -- NOT STOPPED"
                    label = f"  [{reasons[task_id]}]" if task_id in reasons else ""
                    print(f"    {spent:>6}/{ceiling:<4} agents={agents}  status={status}  "
                          f"{task_id}{flag}{label}")

            # Which of the two ceiling checks actually stopped things. The
            # abandoned list labels both "energy ceiling", so without this
            # split there is no way to tell from a run whether the per-cycle
            # check is carrying the load, duplicating the respawn-time one,
            # or never firing at all -- the exact ambiguity that left the
            # cycle cap suspected of not being implemented.
            midagent = verdicts.get("energy_ceiling_midagent", 0)
            at_respawn = verdicts.get("energy_ceiling_at_respawn", 0)
            print(f"    stopped mid-agent (per cycle)  : {midagent}")
            print(f"    stopped at a respawn decision  : {at_respawn}")

            # What admission turned away. All zero on a run whose budget
            # covered everything it planned.
            print(f"\n  ADMISSION (committed at end: {self.committed_energy():.0f} "
                  f"against a limit of {self.colony.starting_budget - self.ADMISSION_FLOOR}: "
                  f"budget {self.colony.starting_budget}, floor {self.ADMISSION_FLOOR})")
            for label, key in (("subtasks dropped from a batch", "admission_subtasks_dropped"),
                               ("SPAWN batches refused outright", "admission_spawn_refused"),
                               ("TASK TOO LARGE conversions refused", "admission_conversion_refused"),
                               ("respawns refused", "admission_respawn_refused"),
                               ("root refused at bootstrap", "admission_root_refused"),
                               ("SPAWN closed (no energy)", "spawn_closed_no_energy"),
                               ("SPAWN refused, no subtask completed",
                                "spawn_refused_no_completed_child"),
                               ("overflow dropped, parent retired", "overflow_dropped_parent_retired")):
                print(f"    {label:<35}: {verdicts.get(key, 0)}")

            live_count = getattr(self, "_live_agents_at_terminate", len(self.live_agents))
            print(f"\n  live_agents at terminate : {live_count}")
            print(f"  peak live_agents        : {getattr(self, 'peak_live_agents', 'n/a')}")
            print("=" * 62 + "\n")
        except Exception:
            print(f"Warning: failed printing energy report:\n{traceback.format_exc()}")

    def _final_answer_grounding(self) -> str:
        """What the final answer was allowed to draw on: the problem and
        every subtask's stored result. The root's own result and the final
        answer itself are left out, or the text under check would ground
        its own made-up figures."""
        parts = []
        if self.spec:
            parts += [str(self.spec.get("raw_text") or ""), str(self.spec.get("goal") or "")]
        for task_id, task in self.task_graph.tasks.items():
            if task_id != self.root_task_id and task.result is not None:
                parts.append(str(task.result))
        return "\n".join(parts)

    # trim_artifact_tail's reason -> the verdict counter it bumps.
    _ARTIFACT_VERDICTS = {
        "code fence": "final_answer_artifact_fence",
        "JSON tail": "final_answer_artifact_json",
        "separator run": "final_answer_artifact_separator",
    }

    def _guard_final_answer(self, best_result):
        """The last check between a collapsed decode and the user.

        Synthesizer.format_output cuts its own output, but it is not the
        only way a final answer leaves this class. A synthesizer that
        raised, a colony built without one, and the root-task fallback all
        return an agent's REPORT verbatim -- and a REPORT is deduped and
        trimmed to three sentences on its way through handle_completion and
        nothing more, so an "ACTION REQUESTED:" line or an echoed prompt
        exemplar inside one reaches the user untouched. That is the same
        failure the synthesizer guard was added for, through a door it does
        not cover.

        Placed on the one return instead of on those three assignments, so
        it is true of every exit including ones added later. Running it over
        an already-cut synthesis is a no-op by construction -- the cut is
        idempotent -- which is what makes covering the exit cheaper than
        covering each source.

        A non-string result (the success branch can return colony.results,
        a dict) has no tail to cut and is returned untouched.
        """
        if not isinstance(best_result, str) or not best_result.strip():
            return best_result

        # PATCH 24. Internal task IDs never reach the user, whichever exit
        # this answer took -- a root REPORT verbatim carries them as easily
        # as a synthesis does.
        best_result, removed_ids = strip_task_ids(best_result)
        if removed_ids:
            self.colony.record_verdict("final_answer_task_ids_stripped")

        # The same three cuts format_output makes, in the same order: a
        # block repeated back to back, degeneracy_cut, then the status-report
        # and closer/restating tails -- so a raw REPORT leaving by the
        # fallback exits no dirtier than a synthesis would have.
        kept, reason = cut_adjacent_repeat(best_result)
        reasons = [reason] if reason else []
        kept, reason = degeneracy_cut(kept, exemplars=PROMPT_EXEMPLARS)
        if reason:
            reasons.append(reason)
        if kept.strip():
            kept, tail_reasons = trim_degenerate_tails(kept, self._final_answer_grounding())
            reasons += tail_reasons
        # Scaffold artifacts: a code fence, a JSON object tail, a separator
        # run. None of the passes above keys on them -- run 3's answer ended
        # "--- --- --- ----", an unterminated ```json fence and a JSON copy
        # of its own first sentence, with every counter here at 0. Fences
        # and JSON stay when the user asked for software.
        if kept.strip():
            raw_text = (getattr(self, "spec", None) or {}).get("raw_text")
            kept, artifact_reasons = trim_artifact_tail(
                kept, allow_code=bool(raw_text) and asks_for_software(raw_text))
            for artifact in artifact_reasons:
                self.colony.record_verdict(self._ARTIFACT_VERDICTS[artifact])
            reasons += artifact_reasons
        # The walk-back again, now that something was cut. It used to run
        # only on the raw decode, before any cut, and a cut can end the text
        # on an unfinished sentence: a line-boundary cut, a mid-line
        # scaffolding cut, or an artifact welded onto a sentence stump.
        if reasons and kept.strip():
            walked = drop_incomplete_tail(kept)
            if walked != kept:
                self.colony.record_verdict("final_answer_incomplete_tail_after_cut")
                reasons.append("an unfinished sentence the cut exposed")
                kept = walked
        if not reasons:
            return best_result

        self.colony.record_verdict("final_answer_cut")
        print(f"  [final-answer] cut at {', '.join(reasons)} "
              f"({len(best_result)} -> {len(kept)} chars) on the way out.")
        if kept.strip():
            return kept

        # Nothing before the collapse. Returned as an explicit failure
        # rather than as "" or as the degenerate text itself: a caller that
        # cannot tell a failed run from a real answer is the hole this
        # exists to close.
        self.colony.record_verdict("final_answer_empty_after_cut")
        print("  [final-answer] nothing survived the cut -- the run's last "
              "text was degenerate from its first sentence.")
        return Synthesizer.DEGENERATE_ANSWER_MESSAGE

    def _synthesis_problem_text(self):
        """PATCH 32 (cont.). What the synthesizer is told the PROBLEM is: the
        user's request (spec["raw_text"]), not the phaser's goal. Run 10's
        goal said "12 gardens" where the request said "12 plots", and the
        answer followed the goal. Used by BOTH synthesis paths (root
        completion and terminate()'s partial synthesis). raw_text may end
        in the phaser's "... [TRUNCATED]" suffix; it is passed as is. Falls
        back to the goal only when there is no raw_text, and counts that."""
        raw = (self.spec or {}).get("raw_text") or ""
        if raw:
            return raw
        if self.synthesizer is not None:
            self.synthesizer._record_trim("problem_fell_back_to_goal")
        return (self.spec or {}).get("goal", "") or ""

    def _print_final_answer_report(self):
        """What the guards did to the text the user actually reads, and to
        the child results that text is built from.

        Printed here and not in _print_energy_report because most of it
        happens after that runs: the partial-result synthesis and the exit
        guard both come later in terminate(), so the counters would read
        zero there no matter what happened.
        """
        verdicts = getattr(self.colony, "verdict_counts", {}) or {}

        # Read together: "at promotion" is the choke point doing its job.
        # Anything under injection is a result that reached storage without
        # going through promotion (a salvaged partial, or a new path). A
        # count there is worth tracking down.
        scrub_rows = (
            ("cut at promotion                  ", verdicts.get("result_scrubbed_at_promotion", 0)),
            ("  empty after the cut             ",
             verdicts.get("result_empty_after_scrub_at_promotion", 0)),
            ("cut at parent injection           ",
             verdicts.get("result_scrubbed_at_parent_injection", 0)),
            ("cut at dependency injection       ",
             verdicts.get("result_scrubbed_at_dependency_injection", 0)),
        )
        if any(count for _, count in scrub_rows):
            print("\n  CHILD RESULT SCRUBS (degeneracy reaching other agents)")
            for label, count in scrub_rows:
                print(f"    {label} : {count}")

        trims = dict(getattr(self.synthesizer, "trim_counts", {}) or {})
        rows = (
            ("synthesis cut at a degenerate tail", trims.get("cut", 0)),
            ("synthesis cut at a repeated block ", trims.get("repeat_cut", 0)),
            ("synthesis cut at a status report  ", trims.get("status_tail", 0)),
            ("synthesis empty after its cut     ", trims.get("empty", 0)),
            ("synthesis shipped still degenerate", trims.get("shipped_degenerate", 0)),
            ("synthesis shipped made-up metrics ", trims.get("shipped_telemetry", 0)),
            ("synthesis stump walked back       ", trims.get("incomplete_tail_after_cut", 0)),
            ("not-completed sentence appended   ", trims.get("not_completed_appended", 0)),
            ("model's own not-completed list cut", trims.get("model_not_completed_cut", 0)),
            ("task IDs stripped from synthesis  ", trims.get("task_ids_stripped", 0)),
            ("synthesis roll-ups omitted (children present)", trims.get("rollups_omitted", 0)),
            ("synthesis problem fell back to goal         ",
             trims.get("problem_fell_back_to_goal", 0)),
            ("task IDs stripped on the way out  ", verdicts.get("final_answer_task_ids_stripped", 0)),
            ("returned answer cut on the way out", verdicts.get("final_answer_cut", 0)),
            ("returned answer empty after cut   ",
             verdicts.get("final_answer_empty_after_cut", 0)),
            ("  code fence cut                  ", verdicts.get("final_answer_artifact_fence", 0)),
            ("  JSON tail cut                   ", verdicts.get("final_answer_artifact_json", 0)),
            ("  separator run cut               ",
             verdicts.get("final_answer_artifact_separator", 0)),
            ("  stump walked back after a cut   ",
             verdicts.get("final_answer_incomplete_tail_after_cut", 0)),
            ("partial banner attached           ", verdicts.get("final_answer_partial_banner", 0)),
            ("  on a SUCCESS run                ", verdicts.get("partial_banner_on_success", 0)),
        )
        if not any(count for _, count in rows):
            return
        print("\n  FINAL ANSWER GUARDS (degeneracy reaching the user)")
        for label, count in rows:
            print(f"    {label.ljust(34)} : {count}")

    def terminate(self) -> Any:
        """
        1. Prints system resolution (Success or Energy Death).
        2. Captures system execution trace using a snapshot.
        3. Clears execution loop flags.
        4. Isolates and returns the highest quality result spec.
        """
        self.running = False
        self.run_trace = self.colony.get_snapshot()

        # Captured before the teardown below empties the dict -- the energy
        # report is printed after it and would otherwise always read 0.
        self._live_agents_at_terminate = len(self.live_agents)

        for live_agent in self.live_agents.values():
            live_agent.KV_Cache = None
            live_agent.last_hidden_state = None
        self.live_agents.clear()

        if self.memory_store is not None:
            try:
                self.memory_store.save_ghosts()
            except Exception as e:
                print(f"Warning: failed to save ghost index on terminate: {e}")

        root_task = self.task_graph.tasks.get(self.root_task_id)
        is_successful = root_task is not None and root_task.status == 2

        if is_successful:
            print("System Status: TERMINATED [SUCCESS]")
            print("The root task successfully completed and synthesized.")
            if self.abandoned_tasks:
                # Both at once: the root was accepted, and some of the work
                # under it was not done. Said here and in the answer itself.
                print(f"PARTIAL: {len(self.abandoned_tasks)} subtask(s) were "
                      f"abandoned -- the returned answer carries the "
                      f"[PARTIAL RESULT] banner.")
        elif self.hit_tick_ceiling:
            print("System Status: TERMINATED [TICK CEILING]")
            print(f"The colony ran {self.tick_count} ticks without converging and "
                  f"hit the {self.MAX_TICKS}-tick ceiling. Energy was NOT the "
                  f"binding constraint -- remaining budget: {self._check_energy()}.")
        elif self.root_task_id in self.abandoned_tasks:
            print("System Status: TERMINATED [ABANDONED]")
            root_reason = self.abandon_reasons.get(self.root_task_id, "attempt cap")
            if root_reason == "attempt cap":
                print(f"The root task exhausted its {self.MAX_TASK_ATTEMPTS} respawn "
                      f"attempts without an acceptable result. Synthesizing whatever "
                      f"subtasks did complete.")
            else:
                print(f"The root task was abandoned ({root_reason}) without an "
                      f"acceptable result. Synthesizing whatever subtasks did complete.")
        else:
            rem_energy = self._check_energy()
            print("System Status: TERMINATED [ENERGY DEATH]")
            print(f"The colony depleted its energy allocation. Remaining budget: {rem_energy}")

        if self.abandoned_tasks:
            print(f"\n  ABANDONED TASKS ({len(self.abandoned_tasks)}):")
            for abandoned_id in self.abandoned_tasks:
                abandoned = self.task_graph.tasks.get(abandoned_id)
                description = abandoned.description if abandoned is not None else "<unknown>"
                label = self.abandon_reasons.get(abandoned_id, "attempt cap")
                print(f"    {abandoned_id}  [{label}]  {description[:60]}")

        if self.unstarted_tasks:
            print(f"\n  NOT STARTED ({len(self.unstarted_tasks)}) -- no energy left "
                  f"to spawn an agent:")
            for unstarted_id in self.unstarted_tasks:
                unstarted = self.task_graph.tasks.get(unstarted_id)
                description = unstarted.description if unstarted is not None else "<unknown>"
                print(f"    {unstarted_id}  {description[:70]}")

        # Printed on BOTH exit paths on purpose: a successful run's ledger is
        # the baseline the failing runs get compared against.
        self._print_energy_report()
        if is_successful:
            outcome = "success"
        elif self.hit_tick_ceiling:
            outcome = "tick_ceiling"
        elif self.root_task_id in self.abandoned_tasks:
            outcome = "root_abandoned"
        else:
            outcome = "energy_death"
        self.run_trace["task_energy"] = self._task_energy_records()
        self._write_energy_trace(outcome)

        best_result = self.colony.results.get("final_spec")
        # The success path's final_spec is a synthesis unless the synthesizer
        # was missing or raised (handle_completion then stores the root's own
        # REPORT); the partial path below always is one.
        synthesized = is_successful and getattr(self, "_final_spec_synthesized", False)
        partial_synthesis = False

        if not best_result and not is_successful and self.synthesizer is not None:
            try:
                partial_results = self.synthesizer.collect_results(
                    self.colony, self.task_graph, root_task_id=self.root_task_id
                )
            except Exception:
                print(f"Warning: failed collecting partial results for synthesis:\n"
                      f"{traceback.format_exc()}")
                partial_results = []

            if partial_results:
                goal_text = self._synthesis_problem_text()
                try:
                    not_completed = self.synthesizer.collect_not_completed(
                        self.task_graph, set(self.abandoned_tasks), partial_results)
                    best_result = self.synthesizer.format_output(
                        partial_results, goal_text, not_completed=not_completed)
                    synthesized = partial_synthesis = True
                except Exception as e:
                    print(f"Warning: failed synthesizing partial result: {e}")

        if not best_result and root_task:
            best_result = getattr(root_task, 'result', None)

        if not best_result:
            if not is_successful:
                rem_energy = self._check_energy()
                best_result = (
                    f"The colony was unable to produce any completed subtask "
                    f"before exhausting its energy budget (remaining: "
                    f"{rem_energy}). No partial result is available to "
                    f"synthesize. Consider increasing the energy budget or "
                    f"breaking the problem into a more explicit, narrower "
                    f"prompt."
                )
            else:
                best_result = self.colony.results

        best_result = self._guard_final_answer(best_result)
        # After the guard, so its cuts never reach into the banner and an
        # answer cut to nothing is judged on the answer, not on the banner.
        # Any run with abandoned subtasks gets it, whatever its status: run 3
        # was SUCCESS with 8 abandoned, all three of the root's subtasks among
        # them, and its answer carried no sign of it.
        if partial_synthesis or self.abandoned_tasks:
            best_result = self._with_partial_banner(best_result, is_successful, synthesized)
        self._print_final_answer_figures(best_result)
        self._print_final_answer_report()
        return best_result

    def _with_partial_banner(self, answer, is_successful: bool, synthesized: bool):
        """`answer` under the [PARTIAL RESULT] banner. Left alone when it is
        not a string, or when no subtask completed -- the answer is then
        already an explicit no-result message, and a banner promising
        completed subtasks below it would be false."""
        if not isinstance(answer, str) or not answer.strip():
            return answer
        completed = sum(
            1 for task_id, task in self.task_graph.tasks.items()
            if task_id != self.root_task_id and task.status == 2 and task.result is not None
        )
        if not completed:
            return answer
        cause = (
            "one or more subtasks were abandoned ("
            + ", ".join(sorted(set(self.abandon_reasons.values())))
            + ")"
            if self.abandoned_tasks
            else "the colony ran out of its energy budget"
        )
        below = ("are synthesized below" if synthesized
                 else "the answer below is the colony's own final report")
        self.colony.record_verdict("final_answer_partial_banner")
        if is_successful:
            self.colony.record_verdict("partial_banner_on_success")
        return (
            f"[PARTIAL RESULT -- {cause} before every subtask finished. "
            f"{completed} subtask(s) completed and {below}; anything not "
            f"mentioned was not reached.]\n\n{answer}"
        )

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
            if self.tick_count >= self.MAX_TICKS:
                self.hit_tick_ceiling = True
                print(f"TERMINATED [TICK CEILING] -- {self.tick_count} ticks elapsed "
                      f"(MAX_TICKS={self.MAX_TICKS}) with the root task still "
                      f"unresolved. Stopping.")
                break

        return self.terminate()