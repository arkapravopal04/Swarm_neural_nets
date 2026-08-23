"""
ghost_extractor.py

Autopsies a dead agent into a ghost record. Pure function, zero dependencies --
takes an agent, returns a dict. Does NOT write to memory_store itself (see
project discussion): the orchestrator is the only thing that decides when a
ghost gets committed, so it can reconcile judge/agent disagreement or inspect
the record before writing. Same boundary discipline as judge.py not owning an
embedding model, and agents not calling tools directly -- everything routes
through or gets handed to the orchestrator, nobody reaches across boundaries.

Two ghost mechanisms exist in the system and this is the second one:
    1. ColonyState.extract_agent_ghost() -- cheap, in-run only, feeds the
       respawned agent's ghost_context within the SAME run.
    2. ghost_extractor.extract() (this file) -- richer record, meant to be
       written into memory_store's persistent ghost index so the lesson
       survives across sessions, queried via memory_store.query_ghosts()
       before a new agent on a similar task even starts thinking.
"""

import re
import time


TIER_1_CRASH = "TIER_1_CRASH"
SEMANTIC_DRIFT = "SEMANTIC_DRIFT"
SELF_REPORTED = "SELF_REPORTED"


def _dedupe_and_cap(text, max_chars: int = 500):
    """
    FIX (confirmed via a real run): verdict["reason"] -- Judge's
    deep_critique output -- was being written into the persistent ghost
    record completely raw. Confirmed via two separate deep_critique calls
    in one run: both degenerated into runaway repetition after their real
    critique content ("The response is entirely fabricated. The
    simulation claim is entirely absent..." repeated 6x verbatim; a chain
    of ~25 unrelated "The X must be Y." bureaucratic filler sentences with
    no connection to the actual critique). agent_node.py's
    _dedupe_repeated_sentences already exists for exactly this failure
    shape and orchestrator.py already applies it before this same
    verdict["reason"] value becomes the NEXT respawn's fail_reason -- but
    ghost_extractor.extract() had no equivalent protection before writing
    it into memory_store's PERSISTENT, cross-session ghost index (see this
    module's own docstring: the whole point of a ghost record is to
    survive across sessions and get surfaced to a FUTURE agent via
    query_ghosts() before it even starts thinking). A degenerate,
    repetition-amplified critique baked permanently into that index is a
    worse, longer-lived version of the exact problem
    _dedupe_repeated_sentences was built to prevent for the in-session
    case.

    This module's docstring states "Pure function, zero dependencies" --
    reusing agent_node.py's helper across files (even via the shared
    notebook-namespace convention other files in this project rely on)
    would quietly violate that stated contract, so this is a small,
    self-contained local equivalent rather than a cross-file reach.
    max_chars defaults higher (500 vs agent_node's 300) since a ghost
    record is meant to be a richer, more informative artifact for a
    FUTURE session (per this module's own docstring) than an immediate
    in-run fail_reason snippet -- it should hold more real signal before
    the cap kicks in.
    """
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    deduped = []
    for s in sentences:
        if not deduped or s.strip() != deduped[-1].strip():
            deduped.append(s)
    result = " ".join(deduped)
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0] + "..."
    return result


def _derive_failure_type(dead_agent, verdict: dict = None) -> str:
    if verdict is None:
        return SELF_REPORTED
    if dead_agent.warning_count >= 3:
        return SEMANTIC_DRIFT
    return TIER_1_CRASH


def extract(dead_agent, verdict: dict = None) -> dict:
    """
    Build a ghost record from a dead agent. Returns a JSON-serializable
    dict -- caller (orchestrator) writes it via memory_store.write("ghost", ...).
    """
    failure_type = _derive_failure_type(dead_agent, verdict)

    if verdict is not None:
        failure_reason = verdict.get("reason")
    else:
        failure_reason = dead_agent.fail_reason

    # FIX: see _dedupe_and_cap's docstring above -- this is the actual fix
    # site. Applied to BOTH branches (verdict-derived and self-reported
    # DIE) since a self-reported DIE payload can degenerate exactly the
    # same way (confirmed elsewhere in this project's history -- DIE
    # payloads repeating the same conclusion many times under greedy
    # decoding).
    failure_reason = _dedupe_and_cap(failure_reason)

    # FIX: confirmed via code audit -- this previously read
    # dead_agent.node.energy_spent unconditionally. That only works when
    # dead_agent is a live Agent instance (agent_node.py's Agent always has
    # a .node pointing to its AgentNode). But orchestrator.py's
    # _kill_and_respawn() has a fallback path -- ghost_source = live_agent
    # if live_agent is not None else self.colony.get_agent(agent_id) --
    # that can hand this function a bare AgentNode instead, whenever the
    # agent_id wasn't (or was no longer) in orchestrator.live_agents. A
    # bare AgentNode has no .node attribute at all (it IS the node, not a
    # wrapper around one), so that line threw AttributeError every time
    # this fallback fired -- silently swallowed by orchestrator's
    # try/except around the ghost-write call, meaning ghost persistence
    # quietly did nothing in that scenario with no visible sign anything
    # went wrong. Trying energy_spent directly off dead_agent first covers
    # the bare-AgentNode case (it has the field natively); falling back to
    # .node.energy_spent covers the live-Agent case (which doesn't proxy
    # this field itself, only .node does) -- works for either shape without
    # needing to know in advance which one was passed in.
    energy_spent = getattr(dead_agent, "energy_spent", None)
    if energy_spent is None:
        energy_spent = getattr(getattr(dead_agent, "node", None), "energy_spent", 0)

    return {
        "agent_id": dead_agent.agent_id,
        "role": dead_agent.role,
        "task": dead_agent.task,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "warning_count": dead_agent.warning_count,
        "thought_process": getattr(dead_agent, "thought_process", None),
        "energy_spent": energy_spent,
        # FIX: confirmed via a real run -- this was documented as "tensor,
        # None in Phase 1" but was never actually None. Agent.think() sets
        # self.last_hidden_state on every call (outputs.hidden_states[-1][:, -1, :]),
        # and _kill_and_respawn() extracts the ghost from the LIVE Agent
        # object before its KV_Cache/hidden-state get nulled out later in
        # that same function -- so a real GPU tensor was leaking into every
        # single ghost record. MemoryStore.write() then crashed trying to
        # json.dump() it on every respawn, silently swallowed by the
        # orchestrator's try/except, meaning ghost persistence had never
        # once actually succeeded. Explicitly discard it here so the record
        # matches what it was always supposed to be: JSON-serializable, and
        # actually written.
        "last_hidden_state": None,  # tensor, None in Phase 1 (see FIX above)
        "tool_calls": [],  # populate from thought_process parsing later
        "timestamp": time.time(),
    }