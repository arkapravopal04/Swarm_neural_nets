"""kv.py -- the plumbing for sharing a KV cache between agents.

See README.md for the design and the API contract. Standalone: imports nothing
from the rest of the repo. Batch size 1 everywhere.
"""

import math
import re
from dataclasses import dataclass

import torch

try:
    from transformers import DynamicCache
except ImportError:  # pragma: no cover - ancient transformers
    DynamicCache = None

SINKS = 4


# --- cache version shim (copied from probes/_cache_compat.py) -----------------
# 5.x: .layers[i].keys/.values; 4.36+: .key_cache/.value_cache; older: tuples.
# Key layout is stable: [batch, kv_heads, seq, head_dim].

def cache_layers(cache):
    if cache is None:
        return []
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(k, v) for k, v in cache]


def build_cache(pairs):
    pairs = [(k, v) for k, v in pairs]
    if DynamicCache is None:
        return tuple(pairs)
    try:
        return DynamicCache(ddp_cache_data=pairs)
    except TypeError:
        pass
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(pairs))
    cache = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        cache.update(k, v, i)
    return cache


def cache_len(cache):
    layers = cache_layers(cache)
    return int(layers[0][0].shape[2]) if layers else 0


# --- Memory -------------------------------------------------------------------

@dataclass
class Memory:
    cache: object                # DynamicCache; keys already RoPE-rotated
    pos: torch.LongTensor        # [slots] position number each slot was written at
    next_pos: int                # where the next token continues

    @property
    def slots(self):
        return cache_len(self.cache)

    def clone(self):
        return Memory(build_cache([(k.clone(), v.clone()) for k, v in cache_layers(self.cache)]),
                      self.pos.clone(), self.next_pos)


# --- model introspection (works on a bare model and a PEFT wrapper) -----------

def _device(model):
    return next(model.parameters()).device


def _rotary(model):
    for name, mod in model.named_modules():
        if name.endswith("rotary_emb") and hasattr(mod, "inv_freq"):
            return mod.inv_freq.float(), float(getattr(mod, "attention_scaling", 1.0))
    raise RuntimeError("no *rotary_emb module with inv_freq found")


def _q_norms(model):
    found = []
    for name, mod in model.named_modules():
        if name.endswith("self_attn.q_norm"):
            m = re.search(r"layers\.(\d+)\.", name)
            found.append((int(m.group(1)), mod))
    if not found:
        raise RuntimeError("no *self_attn.q_norm modules found")
    return [mod for _, mod in sorted(found, key=lambda t: t[0])]


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(x, angle_pos, inv_freq, scale=1.0):
    """Rotates x [..., L, D] (fp32) by angle_pos [L] * inv_freq, rotate_half style."""
    freqs = angle_pos.float()[:, None] * inv_freq.to(x.device)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return x * (emb.cos() * scale) + _rotate_half(x) * (emb.sin() * scale)


# --- run ----------------------------------------------------------------------

@torch.no_grad()
def run(model, ids, mem=None, *, inplace=False, logits="last", capture_q=False, chunk=1024):
    """Prefills ids [1, L] on top of mem. Returns (Memory, logits, q); see README."""
    dev = _device(model)
    ids = ids.to(dev)
    L = ids.shape[1]
    if mem is not None and not inplace:
        mem = mem.clone()
    cache = mem.cache if mem is not None else None
    start = mem.next_pos if mem is not None else 0
    pos_all = torch.arange(start, start + L, device=dev)

    grabbed, hooks = [], []
    if capture_q:
        norms = _q_norms(model)
        grabbed = [[] for _ in norms]
        for i, mod in enumerate(norms):
            # q_norm output is the pre-RoPE query, [1, L, H_q, D]
            hooks.append(mod.register_forward_hook(
                lambda _m, _inp, out, i=i: grabbed[i].append(out.detach().float())))

    outs = []
    try:
        for s in range(0, L, chunk):
            piece = ids[:, s:s + chunk]
            last_chunk = s + chunk >= L
            keep_n = 0 if logits == "all" else 1
            # position_ids explicit, cache_position never passed: HF then derives
            # cache_position from slot count, so the causal mask follows slot order,
            # which stays right after keep() drops slots or restamp() renumbers them.
            out = model(input_ids=piece, past_key_values=cache, use_cache=True,
                        position_ids=pos_all[s:s + chunk][None], logits_to_keep=keep_n)
            cache = out.past_key_values
            if logits == "all":
                outs.append(out.logits.float())
            elif logits == "last" and last_chunk:
                outs.append(out.logits[:, -1].float())
    finally:
        for h in hooks:
            h.remove()

    old_pos = mem.pos.to(dev) if mem is not None else torch.zeros(0, dtype=torch.long, device=dev)
    new = Memory(cache, torch.cat([old_pos, pos_all]), start + L)
    if mem is not None and inplace:
        mem.cache, mem.pos, mem.next_pos = new.cache, new.pos, new.next_pos
        new = mem

    lg = None
    if logits == "all":
        lg = torch.cat(outs, dim=1)
    elif logits == "last":
        lg = outs[-1]

    q = None
    if capture_q:
        inv_freq, scale = _rotary(model)
        q = []
        for parts in grabbed:
            x = torch.cat(parts, dim=1).transpose(1, 2)          # [1, H_q, L, D]
            q.append(_rope(x, pos_all.to(x.device), inv_freq, scale))
    return new, lg, q


# --- cutting and renumbering --------------------------------------------------

def keep(mem, idx):
    """Keeps slots idx (plus the sinks). Positions and next_pos are unchanged."""
    idx = idx.long().cpu()
    sinks = torch.arange(min(SINKS, mem.slots))
    idx = torch.unique(torch.cat([sinks, idx]))                  # sorted, deduped
    pairs = []
    for k, v in cache_layers(mem.cache):
        i = idx.to(k.device)
        pairs.append((k.index_select(2, i).clone(), v.index_select(2, i).clone()))
    return Memory(build_cache(pairs), mem.pos[idx.to(mem.pos.device)].clone(), mem.next_pos)


@torch.no_grad()
def restamp(model, mem, start=0):
    """Renumbers slots start..start+slots-1 by re-rotating each key by (new - old)."""
    inv_freq, _ = _rotary(model)
    n = mem.slots
    new_pos = torch.arange(start, start + n, device=mem.pos.device)
    delta = new_pos - mem.pos
    pairs = []
    for k, v in cache_layers(mem.cache):
        # pure rotation (no attention_scaling: the cached keys already carry it)
        k2 = _rope(k.float(), delta.to(k.device), inv_freq).to(k.dtype)
        pairs.append((k2, v.clone()))
    return Memory(build_cache(pairs), new_pos, start + n)


# --- scoring ------------------------------------------------------------------

@torch.no_grad()
def scores(mem, q, upto=None):
    """Attention of captured queries onto the first `upto` slots, -> [upto].

    q is assumed to come from the last L slots of mem (as run() leaves it), so
    query l sits at slot mem.slots - L + l and is masked from later slots; that
    only matters when those slots fall inside `upto` (build's writer window).
    """
    layers = cache_layers(mem.cache)
    upto = mem.slots if upto is None else upto
    total = None
    for (k, _), ql in zip(layers, q):
        K = k[0, :, :upto].float()                               # [H_kv, S, D]
        H_kv, S, D = K.shape
        H_q, L = ql.shape[1], ql.shape[2]
        g = H_q // H_kv
        Q = ql[0].to(K.device).reshape(H_kv, g * L, D)           # heads h*g..h*g+g-1 share kv head h
        att = Q @ K.transpose(1, 2) / math.sqrt(D)               # [H_kv, g*L, S]
        q_slot = torch.arange(mem.slots - L, mem.slots, device=K.device).repeat(g)
        att = att.masked_fill(torch.arange(S, device=K.device)[None, :] > q_slot[:, None], float("-inf"))
        s = att.softmax(-1).sum(dim=(0, 1))
        s = (s / s.sum()).cpu()
        total = s if total is None else total + s
    return total / len(layers)


def pick_last(mem, n, lo=0):
    return torch.arange(max(lo, mem.slots - n), mem.slots)


def pick_top(s, n, lo=0):
    sub = s[lo:]
    top = torch.topk(sub, min(n, sub.numel())).indices + lo
    return torch.sort(top.cpu()).values


def build(model, ids, window=32):
    """Prefills ids; the last `window` tokens run with capture_q to score A's slots."""
    L = ids.shape[1]
    w = min(window, L)
    mem = None
    if L > w:
        mem, _, _ = run(model, ids[:, :L - w], logits=None)
    mem, _, q = run(model, ids[:, L - w:], mem, inplace=True, logits=None, capture_q=True)
    return mem, scores(mem, q)


def reader_scores(model, mem, b_ids):
    """B's ids over a clone of mem; returns B's attention onto mem's slots."""
    ext, _, q = run(model, b_ids, mem, logits=None, capture_q=True)
    return scores(ext, q, upto=mem.slots)


# --- decoding -----------------------------------------------------------------

@torch.no_grad()
def decode(model, mem, ids, max_new, stop_ids, temperature=0.0, top_p=1.0, seed=0):
    """Prefills ids on a clone of mem (or fresh if mem is None), then 1 token per step.

    Returned ids exclude the stop token. The returned Memory holds ids plus every
    returned token (the stop token is not fed in).
    """
    stop = set(int(s) for s in stop_ids)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mem, lg, _ = run(model, ids, mem)
    out = []
    for _ in range(max_new):
        if temperature == 0:
            tok = int(lg[0].argmax())
        else:
            probs = (lg[0].float().cpu() / temperature).softmax(-1)
            sp, si = probs.sort(descending=True)
            cut = sp.cumsum(-1) - sp >= top_p                    # keep the smallest set reaching top_p
            sp = sp.masked_fill(cut, 0.0)
            tok = int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen)])
        if tok in stop:
            break
        out.append(tok)
        mem, lg, _ = run(model, torch.tensor([[tok]]), mem, inplace=True)
    return out, mem


def kl(logits_p, logits_q):
    """Per-row KL(softmax p || softmax q), float32."""
    lp = torch.log_softmax(logits_p.float(), dim=-1)
    lq = torch.log_softmax(logits_q.float().to(lp.device), dim=-1)
    return (lp.exp() * (lp - lq)).sum(-1)
