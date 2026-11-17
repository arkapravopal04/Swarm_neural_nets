"""
fine_tune_v2/generate_eval.py -- generate decide() outputs for an evaluation
set on a GPU (TRAINING_SPEC.md §9-§10). Scoring is done separately, on a CPU,
by score_eval.py.

One process loads the base model once and serves every run it is given:
the base arm with the adapter switched off, each adapter by name. Serving is
the v2 contract (serving.render: chat template, one user turn, thinking off;
no prompt penalties; stops on <|im_end|> / <|endoftext|>), sampled as decide()
samples (T 0.7, top_p 0.9, top_k from the base generation_config,
min_new_tokens 10, max_new_tokens 400) or greedy.

    python fine_tune_v2/generate_eval.py --rows fine_tune_v2/data/dev.jsonl --out-dir eval_out \
        --adapter s0=adapters/v2-lora-full-lr0.0002-s0/best --adapter s1=.../s1/best \
        --run base:sampled:0 --run base:greedy:0 --run s0:sampled:0 --run s1:sampled:1

Each run writes <out-dir>/<arm>_<mode>_<seed>.jsonl, one line per row:
{"id", "text", "eos", "n_new"}. A run that is interrupted resumes where it
stopped (rows already written are skipped).
"""
import argparse
import json
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import serving  # noqa: E402

MAX_NEW = 400


def parse_run(spec):
    arm, mode, seed = spec.split(":")
    if mode not in ("sampled", "greedy"):
        raise SystemExit(f"mode must be sampled or greedy: {spec}")
    return arm, mode, int(seed)


def batches(items, budget):
    """Rows sorted by prompt length, packed so (longest prompt + MAX_NEW) x
    batch size stays under `budget` tokens (the KV cache dominates memory)."""
    items = sorted(items, key=lambda x: len(x[1]))
    out, cur = [], []
    for it in items:
        longest = max([len(c[1]) for c in cur] + [len(it[1])])
        if cur and (longest + MAX_NEW) * (len(cur) + 1) > budget:
            out.append(cur)
            cur = []
        cur.append(it)
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--run", action="append", required=True, help="arm:mode:seed; arm 'base' = adapter off")
    ap.add_argument("--batch-tokens", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import fine_tuning
    fine_tuning.disable_peft_torchao_probe()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapters = dict(a.split("=", 1) for a in args.adapter)
    runs = [parse_run(r) for r in args.run]
    for arm, _, _ in runs:
        if arm != "base" and arm not in adapters:
            raise SystemExit(f"run arm {arm!r} has no --adapter {arm}=path")

    kw = {"revision": args.revision} if args.revision else {}
    tok = AutoTokenizer.from_pretrained(args.base, **kw)
    tok.padding_side = "left"
    cuda = torch.cuda.is_available()
    dev = 0 if cuda else "cpu"
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16) if cuda         else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map={"": dev}, **kw)
    if adapters:
        names = list(adapters)
        model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0])
        for n in names[1:]:
            model.load_adapter(adapters[n], adapter_name=n)
    model.eval()
    print(f"serving: chat template sha {serving.chat_template_sha256(tok)[:16]}, dtype {dtype}, "
          f"adapters {sorted(adapters)}", flush=True)

    with open(args.rows, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][:args.limit]
    encoded = [(r["id"], serving.render(tok, r["prompt"])) for r in rows]
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    os.makedirs(args.out_dir, exist_ok=True)

    for arm, mode, seed in runs:
        path = os.path.join(args.out_dir, f"{arm}_{mode}_{seed}.jsonl")
        done, kept = set(), []
        if os.path.exists(path):  # resume; a line cut off by a dead session is dropped
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["id"])
                        kept.append(line.rstrip("\n") + "\n")
                    except (ValueError, KeyError):
                        pass
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.writelines(kept)
        todo = [e for e in encoded if e[0] not in done]
        print(f"== {arm} {mode} seed {seed}: {len(todo)} rows to go ({len(done)} done) -> {path}", flush=True)
        if not todo:
            continue
        if arm != "base":
            model.set_adapter(arm)
        gen_kw = dict(max_new_tokens=MAX_NEW, min_new_tokens=10, eos_token_id=stop_ids,
                      pad_token_id=tok.pad_token_id, use_cache=True)
        gen_kw.update(dict(do_sample=True, temperature=0.7, top_p=0.9) if mode == "sampled"
                      else dict(do_sample=False))
        t0, n_tok = time.time(), 0
        bl = batches(todo, args.batch_tokens)
        with open(path, "a", encoding="utf-8", newline="\n") as out:
            for bi, batch in enumerate(bl):
                enc = tok.pad({"input_ids": [ids for _, ids in batch]}, return_tensors="pt").to(dev)
                # Seeded by the batch's rows, not its position in the run.
                torch.manual_seed(seed * 1_000_003 + zlib.crc32("|".join(i for i, _ in batch).encode()))
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=cuda):
                    if arm == "base" and adapters:
                        with model.disable_adapter():
                            seqs = model.generate(**enc, **gen_kw)
                    else:
                        seqs = model.generate(**enc, **gen_kw)
                width = enc["input_ids"].shape[1]
                for (rid, _), seq in zip(batch, seqs):
                    new = seq[width:].tolist()
                    cut = next((i for i, t in enumerate(new) if t in stop_ids), None)
                    kept = new if cut is None else new[:cut]
                    n_tok += len(kept) + (0 if cut is None else 1)
                    out.write(json.dumps({"id": rid, "text": tok.decode(kept, skip_special_tokens=True),
                                          "eos": cut is not None,
                                          "n_new": len(kept) + (0 if cut is None else 1)},
                                         ensure_ascii=False) + "\n")
                out.flush()
                if bi % 10 == 0 or bi == len(bl) - 1:
                    el = time.time() - t0
                    print(f"   batch {bi + 1}/{len(bl)} ({len(batch)} rows)  {n_tok / max(el, 1e-9):.0f} tok/s  "
                          f"{el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
