"""
probe1_vram_census.py -- how many agents actually fit on this GPU?

Answers one question: with the model resident, how much VRAM does one
agent's KV cache cost per token, and therefore how many concurrent agents
can the colony hold before it OOMs?

This gates every later decision. agent_node.Agent.MAX_TOTAL_THINK_TOKENS is
4096 and orchestrator.terminate() is currently the only place a cache is
ever freed, so today's answer is "however many agents happen to spawn". If
this probe says four, then the fan-out caps in orchestrator.py
(MAX_SUBTASKS_ROOT / MAX_SUBTASKS_NON_ROOT) and cache eviction are
load-bearing architecture rather than tuning constants, and that needs to
be known before any of the latent work starts.

Measures rather than derives -- the config-based formula is printed too, as
a cross-check, but real allocation includes fragmentation and allocator
slack that arithmetic misses.

Run:  python probes/probe1_vram_census.py --adapter /path/to/adapter
"""

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _cache_compat import cache_layers, cache_len, describe_cache

MB = 1024 ** 2
GB = 1024 ** 3


def _allocated():
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


def _free_total():
    if not torch.cuda.is_available():
        return 0, 0
    free, total = torch.cuda.mem_get_info()
    return free, total


def formula_bytes_per_token(config, dtype_bytes):
    """2 (K and V) * layers * kv_heads * head_dim * bytes.

    kv_heads is num_key_value_heads, NOT num_attention_heads -- Qwen3 uses
    grouped-query attention, so getting this wrong overestimates the cache
    by the GQA ratio (4x on a 32/8 split)."""
    layers = getattr(config, "num_hidden_layers", None)
    kv_heads = getattr(config, "num_key_value_heads", None) or getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden = getattr(config, "hidden_size", 0)
        heads = getattr(config, "num_attention_heads", 1)
        head_dim = hidden // heads if heads else 0
    if not all([layers, kv_heads, head_dim]):
        return None, {}
    per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    return per_token, {"layers": layers, "kv_heads": kv_heads, "head_dim": head_dim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (optional)")
    ap.add_argument("--lengths", default="256,512,1024,2048,4096",
                    help="cache lengths to measure, in tokens")
    ap.add_argument("--think-ceiling", type=int, default=4096,
                    help="agent_node.Agent.MAX_TOTAL_THINK_TOKENS")
    ap.add_argument("--test-offload", action="store_true",
                    help="time a GPU->CPU->GPU cache round trip (eviction cost)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device. This probe measures GPU memory and has nothing "
              "to say on CPU -- run it where the colony actually runs.")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = _allocated()

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter {args.adapter} ...")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    weights = _allocated() - baseline
    free, total = _free_total()
    print(f"\n{'='*66}\nMODEL RESIDENT")
    print(f"  weights            {weights/GB:6.2f} GB")
    print(f"  device total       {total/GB:6.2f} GB")
    print(f"  free after load    {free/GB:6.2f} GB")

    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    dtype_bytes = 2
    predicted, dims = formula_bytes_per_token(cfg, dtype_bytes)
    if predicted:
        print(f"\nCONFIG-DERIVED (cross-check)")
        print(f"  layers={dims['layers']} kv_heads={dims['kv_heads']} "
              f"head_dim={dims['head_dim']} dtype=fp16")
        print(f"  predicted          {predicted/1024:6.1f} KB/token")

    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"\n{'='*66}\nMEASURED CACHE COST")
    print(f"{'tokens':>8} {'cache MB':>10} {'KB/token':>10} {'peak MB':>10}")

    measured = {}
    for n in lengths:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = _allocated()

        ids = torch.randint(0, tok.vocab_size, (1, n), device=model.device)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values

        cache_bytes = _allocated() - before
        peak = torch.cuda.max_memory_allocated() - before
        measured[n] = cache_bytes
        print(f"{n:>8} {cache_bytes/MB:>10.1f} {cache_bytes/n/1024:>10.2f} {peak/MB:>10.1f}")

        if n == max(lengths):
            print(f"\n  {describe_cache(cache)}")
            if args.test_offload:
                layers = cache_layers(cache)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                cpu = [(k.to("cpu", non_blocking=False), v.to("cpu", non_blocking=False))
                       for k, v in layers]
                torch.cuda.synchronize(); t1 = time.perf_counter()
                _ = [(k.to(model.device), v.to(model.device)) for k, v in cpu]
                torch.cuda.synchronize(); t2 = time.perf_counter()
                print(f"  offload  GPU->CPU {1000*(t1-t0):7.1f} ms"
                      f"   CPU->GPU {1000*(t2-t1):7.1f} ms"
                      f"   (eviction round trip for one {n}-token agent)")
                del cpu

        del out, cache, ids
        gc.collect()
        torch.cuda.empty_cache()

    # Capacity: the number that actually decides the architecture.
    ceiling = args.think_ceiling
    per_token = measured.get(ceiling)
    if per_token:
        per_agent = per_token
    else:
        biggest = max(measured)
        per_agent = measured[biggest] / biggest * ceiling

    gc.collect(); torch.cuda.empty_cache()
    free, total = _free_total()
    headroom = free * 0.85  # leave room for activations during generation

    print(f"\n{'='*66}\nCOLONY CAPACITY")
    print(f"  per agent @ {ceiling} tokens   {per_agent/MB:7.1f} MB")
    print(f"  usable headroom (85% free)  {headroom/GB:7.2f} GB")
    print(f"  concurrent agent caches     {int(headroom // per_agent)}")
    print(f"\n  orchestrator.py currently caps fan-out at "
          f"MAX_SUBTASKS_ROOT=3, MAX_SUBTASKS_NON_ROOT=2, and frees caches "
          f"only in terminate().")
    print(f"  If the number above is smaller than the agent count a real run "
          f"reaches, cache eviction is a prerequisite for LatentMAS, not a "
          f"later optimization.")


if __name__ == "__main__":
    main()
