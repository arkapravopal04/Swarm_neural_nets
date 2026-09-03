"""
ghost_extractor.py

Autopsies a dead agent into a ghost record. Pure function -- takes an
agent, returns a dict; its only import is text_utils, which is itself
dependency-free (see that module's docstring for why the dedupe/cap
logic now lives there instead of as a local copy). Does NOT write to
memory_store itself (see
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

import time

from text_utils import dedupe_and_cap as _dedupe_and_cap


TIER_1_CRASH = "TIER_1_CRASH"
SEMANTIC_DRIFT = "SEMANTIC_DRIFT"
SELF_REPORTED = "SELF_REPORTED"


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