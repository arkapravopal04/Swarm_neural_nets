"""
_cache_compat.py -- one thin shim over the transformers KV-cache API.

The probes need to read, slice, concatenate and rebuild KV caches. The API
for that has moved three times:

    transformers <4.36 : past_key_values is a tuple of (key, value) tuples
    transformers 4.36+ : DynamicCache with .key_cache[i] / .value_cache[i]
    transformers 5.x   : DynamicCache with .layers[i].keys / .layers[i].values

Kaggle's pinned version is not necessarily the one you develop against, and
a probe that dies on an AttributeError tells you nothing about the model.
Every probe goes through these five functions so the version question is
answered in exactly one place.

Key tensor layout is stable across all three: (batch, kv_heads, seq, head_dim).
Sequence position is dim=2 -- that is the axis every splice concatenates on.
"""

import torch

try:
    from transformers import DynamicCache
except ImportError:  # pragma: no cover - ancient transformers
    DynamicCache = None


def cache_layers(cache):
    """Returns [(keys, values), ...] per layer, oldest-to-newest layer order."""
    if cache is None:
        return []
    # 5.x
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    # 4.36+
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    # legacy tuple-of-tuples
    return [(k, v) for k, v in cache]


def build_cache(layer_pairs):
    """Rebuilds a cache object from [(keys, values), ...]."""
    if DynamicCache is None:
        return tuple(tuple(p) for p in layer_pairs)

    # 5.x accepts layer data straight through the constructor.
    try:
        return DynamicCache(ddp_cache_data=layer_pairs)
    except TypeError:
        pass

    # 4.36+ path: from_legacy_cache takes the tuple form.
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(tuple(p) for p in layer_pairs))

    cache = DynamicCache()
    for idx, (k, v) in enumerate(layer_pairs):
        cache.update(k, v, idx)
    return cache


def cache_len(cache):
    """Number of tokens currently held in the cache."""
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    layers = cache_layers(cache)
    return int(layers[0][0].shape[2]) if layers else 0


def clone_cache(cache):
    """Deep copy. Generating on a cache MUTATES it -- any A/B comparison that
    reuses the same cache twice without cloning is measuring the first arm's
    leftovers, not the second arm."""
    return build_cache([(k.clone(), v.clone()) for k, v in cache_layers(cache)])


def concat_caches(first, second):
    """Splices `second` onto the end of `first` along the sequence axis.

    This is the whole LatentMAS primitive in one line per layer: agent A's
    attention state becomes a prefix of agent B's. What it does NOT do is
    fix up positions -- the keys in `first` were rotated by RoPE at the
    positions they were built at, and they keep those. Probe 3 exists
    specifically to measure whether that matters."""
    a, b = cache_layers(first), cache_layers(second)
    if len(a) != len(b):
        raise ValueError(f"layer count mismatch: {len(a)} vs {len(b)}")
    return build_cache([
        (torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2))
        for (ka, va), (kb, vb) in zip(a, b)
    ])


def crop_cache(cache, length):
    """Truncates the cache back to `length` tokens, in place where supported.

    This is what makes decide()'s 3-attempt retry loop possible on a shared
    cache: snapshot the length, generate, crop back, try again."""
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    return build_cache([(k[:, :, :length, :], v[:, :, :length, :])
                        for k, v in cache_layers(cache)])


def describe_cache(cache):
    layers = cache_layers(cache)
    if not layers:
        return "empty cache"
    k = layers[0][0]
    total = sum(kk.numel() * kk.element_size() + vv.numel() * vv.element_size()
                for kk, vv in layers)
    return (f"{len(layers)} layers, seq={k.shape[2]}, kv_heads={k.shape[1]}, "
            f"head_dim={k.shape[3]}, dtype={k.dtype}, {total / 1024**2:.1f} MB")
