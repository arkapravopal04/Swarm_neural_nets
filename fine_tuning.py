"""
fine_tuning.py -- train Hive adapter v2 (TRAINING_SPEC.md §11).

    python fine_tuning.py --adapter dora --data fine_tune_v2/data --out adapters/v2-dora-s0 --seed 0

Rows are fine_tune_v2/data/{train,dev}.jsonl (prompt = the runtime-rendered user
turn, response = the gold decide() output). Each row is tokenised by
serving.encode_training_pair -- the same chat helper v2 serving uses -- so the
training bytes are the serving bytes: one user turn, no system turn, thinking
off, loss on the response and its <|im_end|> only.

Memory: logits are computed only at answer positions (`logits_to_keep`), never
over the 2,000-3,000-token prompt; gradient checkpointing; microbatch 1 with
gradient accumulation. fp16 base + fp32 adapter on T4/P100 (no bf16 there),
bf16 where the GPU supports it.

Resumable: a checkpoint holds the adapter, optimiser, scheduler, scaler, RNG
state and the position in the data order, so a dead Kaggle session continues
exactly (--resume).
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import serving  # noqa: E402

DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"  # D6


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["dora", "lora"], required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "fine_tune_v2", "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--revision", default=None, help="pin the base revision (S5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-4 DoRA, 2e-4 LoRA")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--subset", type=float, default=1.0, help="fraction of train rows (lr sweep: 0.25)")
    ap.add_argument("--checkpoint-every", type=int, default=25, help="optimiser updates")
    ap.add_argument("--dev-gen", type=int, default=60, help="dev rows generated at each checkpoint")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-updates", type=int, default=None, help="stop early (smoke runs)")
    ap.add_argument("--include-longest", action="store_true",
                    help="always train on the longest row first (smoke: measures peak memory)")
    ap.add_argument("--dev-loss-rows", type=int, default=150, help="dev rows scored for loss per eval")
    return ap.parse_args(argv)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ model

def pick_dtype():
    """bf16 only on hardware that has it (compute capability >= 8).
    torch.cuda.is_bf16_supported() counts emulation by default and says True
    on a T4, which would skip fp16 loss scaling and run emulated bf16."""
    if not torch.cuda.is_available():
        return torch.float32  # CPU smoke runs only
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def load(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"revision": args.revision} if args.revision else {}
    tok = AutoTokenizer.from_pretrained(args.base, **kw)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = pick_dtype()
    # One GPU: a naive layer split over T4 x2 gives no speed-up and needs every
    # index tensor on the right device; the model (~8 GB fp16) plus the adapter,
    # its optimiser state and checkpointed activations fit one 15-16 GB card.
    device_map = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map=device_map, **kw)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=args.targets.split(","), use_dora=(args.adapter == "dora"),
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    # Adapter weights and optimiser state in fp32 (PEFT keeps the base frozen).
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return tok, model, dtype


def encode_rows(tok, rows, max_length):
    out, rejected = [], 0
    for r in rows:
        try:
            e = serving.encode_training_pair(tok, r["prompt"], r["response"], max_length)
        except ValueError as err:
            rejected += 1
            print(f"  [encode] {r['id']}: {err}")
            continue
        e["id"] = r["id"]
        out.append(e)
    return out, rejected


def row_loss(model, ex, device, dtype):
    """Cross-entropy on answer tokens only, logits computed only there."""
    ids = torch.tensor([ex["input_ids"]], device=device)
    labels = torch.tensor(ex["labels"], device=device)
    pos = (labels[1:] != -100).nonzero(as_tuple=True)[0]  # logit at t predicts token t+1
    with torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        # CPU indices are valid for a tensor on any device.
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=pos.cpu(),
                    use_cache=False)
    logits = out.logits[0].float()
    target = labels[pos + 1].to(logits.device)
    return torch.nn.functional.cross_entropy(logits, target), int(target.numel())


# -------------------------------------------------------------- evaluation

_ACTION_RE = re.compile(r"^ACTION:\s*([A-Z]+)\s*\nPAYLOAD:\s*(.*)$", re.S)


def score(gen, gold):
    """Cheap dev metrics for checkpoint choice; TRAINING_SPEC §9's full
    metrics (runtime parsers, figure checks) run in the evaluation harness."""
    g = _ACTION_RE.match(gold.strip())
    m = _ACTION_RE.match(gen.strip())
    res = {"action_match": 0, "payload_valid": 0, "exact": int(gen.strip() == gold.strip())}
    if not m or not g:
        return res
    res["action_match"] = int(m.group(1) == g.group(1))
    if m.group(1) in ("SPAWN", "TOOL"):
        try:
            json.loads(m.group(2))
            res["payload_valid"] = 1
        except ValueError:
            pass
    else:
        res["payload_valid"] = int(bool(m.group(2).strip()))
    return res


@torch.no_grad()
def evaluate(model, tok, dev_enc, dev_rows, device, dtype, n_gen, n_loss):
    model.eval()
    model.config.use_cache = True
    try:
        return _evaluate(model, tok, dev_enc[:n_loss], dev_rows, device, dtype, n_gen)
    finally:
        model.config.use_cache = False
        model.train()


def _evaluate(model, tok, dev_enc, dev_rows, device, dtype, n_gen):
    losses = []
    for ex in dev_enc:
        loss, _ = row_loss(model, ex, device, dtype)
        losses.append(loss.item())
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    agg = {"action_match": 0, "payload_valid": 0, "exact": 0, "stopped": 0}
    rows = dev_rows[:n_gen]
    for r in rows:
        ids = torch.tensor([serving.render(tok, r["prompt"])], device=device)
        with torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=400,
                                 do_sample=False, eos_token_id=stop_ids, pad_token_id=tok.pad_token_id)
        new = out[0, ids.shape[1]:].tolist()
        agg["stopped"] += int(any(t in stop_ids for t in new))
        s = score(tok.decode(new, skip_special_tokens=True), r["response"])
        for k in s:
            agg[k] += s[k]
    n = max(1, len(rows))
    return {"dev_loss": sum(losses) / max(1, len(losses)), **{k: v / n for k, v in agg.items()},
            "n_gen": len(rows)}


# ---------------------------------------------------------------- training

def save_adapter_atomic(model, path):
    """Write to path.tmp; _swap moves it into place, so a session killed
    mid-save never leaves a half-written adapter where a resume points."""
    import shutil
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    return tmp


def _swap(tmp, path):
    """Old -> path.old, tmp -> path, then delete path.old. A kill at any point
    leaves a complete directory under path, path.old or path.tmp, which
    find_checkpoint() picks up on --resume."""
    import shutil
    old = path + ".old"
    if os.path.exists(old):
        shutil.rmtree(old)
    if os.path.exists(path):
        os.replace(path, old)
    os.replace(tmp, path)
    if os.path.exists(old):
        shutil.rmtree(old)


def find_checkpoint(ckpt):
    """The newest complete checkpoint: last/, else last.tmp/ (finished but not
    swapped in), else last.old/. trainer_state.pt is written last, so its
    presence marks a complete directory."""
    for cand in (ckpt, ckpt + ".tmp", ckpt + ".old"):
        if os.path.exists(os.path.join(cand, "trainer_state.pt")):
            return cand
    return None


def save_checkpoint(model, opt, sched, scaler, state, path):
    path_final = path
    path = save_adapter_atomic(model, path_final)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "rng": {"py": random.getstate(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                "state": state}, os.path.join(path, "trainer_state.pt"))
    _swap(path, path_final)


def versions():
    out = {"torch": torch.__version__}
    for mod in ("transformers", "peft", "accelerate"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def code_commit():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    args = parse_args(argv)
    if args.lr is None:
        args.lr = 1e-4 if args.adapter == "dora" else 2e-4
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    manifest = json.load(open(os.path.join(args.data, "MANIFEST.json"), encoding="utf-8"))
    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    dev_rows = read_jsonl(os.path.join(args.data, "dev.jsonl"))
    if args.subset < 1.0:
        random.Random(args.seed).shuffle(train_rows)
        train_rows = train_rows[:max(1, int(len(train_rows) * args.subset))]

    tok, model, dtype = load(args)
    sha = serving.chat_template_sha256(tok)
    if sha != manifest.get("chat_template_sha256") and args.revision:
        raise SystemExit(f"chat template sha {sha[:16]} differs from the data's "
                         f"{str(manifest.get('chat_template_sha256'))[:16]} with a pinned --revision (S5)")
    if sha != manifest.get("chat_template_sha256"):
        print(f"WARNING: chat template sha {sha[:16]} differs from the one the data was measured with "
              f"({str(manifest.get('chat_template_sha256'))[:16]}). Training and serving still match "
              f"each other (both use this tokenizer); token counts in the manifest may be off.")
    # Inputs go where the embedding lives (device_map="auto" may split layers
    # over two GPUs on Kaggle T4 x2).
    device = model.get_input_embeddings().weight.device
    t_enc = time.time()
    train_enc, rej_t = encode_rows(tok, train_rows, args.max_length)
    if args.include_longest and train_enc:
        # The longest row of the FULL train set goes first, so peak memory is measured.
        full_enc, _ = encode_rows(tok, read_jsonl(os.path.join(args.data, "train.jsonl")), args.max_length)
        longest = max(full_enc, key=lambda e: len(e["input_ids"]))
        train_enc = [longest] + [e for e in train_enc if e["id"] != longest["id"]]
        print(f"longest train row: {longest['id']} ({len(longest['input_ids'])} tokens) is trained first")
    dev_enc, rej_d = encode_rows(tok, dev_rows, args.max_length)
    print(f"encoded {len(train_enc)} train / {len(dev_enc)} dev rows in {time.time() - t_enc:.0f} s "
          f"(rejected {rej_t} / {rej_d} over max_length {args.max_length})")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_updates = math.ceil(len(train_enc) * args.epochs / args.accum)
    if args.max_updates:
        total_updates = min(total_updates, args.max_updates)
    warm = max(1, int(total_updates * args.warmup))

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_updates - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    # init_scale 2**12: the default 65536 overflows the first several fp16 steps.
    scaler = (torch.amp.GradScaler("cuda", init_scale=2 ** 12)
              if dtype == torch.float16 and torch.cuda.is_available() else None)

    order_rng = random.Random(args.seed)
    order = [0] if args.include_longest else []
    # Room for skipped fp16 steps, which consume rows but no update (capped at 40 above).
    while len(order) < (total_updates + 50) * args.accum:
        # With --include-longest the longest row (index 0) opens the run once; later epochs include it as usual.
        epoch = list(range(1 if (args.include_longest and not order[1:]) else 0, len(train_enc)))
        order_rng.shuffle(epoch)
        order += epoch
    state = {"update": 0, "cursor": 0, "history": [], "best": None, "skipped": 0,
             "train_seconds": 0.0, "tokens_seen": 0, "peak_gib": 0.0}
    ckpt = os.path.join(args.out, "last")
    found = find_checkpoint(ckpt) if args.resume else None
    if args.resume and found is None:
        print("WARNING: --resume found no complete checkpoint under last/ -- starting from scratch")
    if found is not None:
        if found != ckpt:
            print(f"resuming from {found} (last/ was missing or incomplete)")
        ckpt_in = found
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        blob = torch.load(os.path.join(ckpt_in, "trainer_state.pt"), weights_only=False)
        res = set_peft_model_state_dict(model, load_file(os.path.join(ckpt_in, "adapter_model.safetensors")))
        unexpected = list(getattr(res, "unexpected_keys", []) or [])
        missing = [k for k in (getattr(res, "missing_keys", []) or []) if "lora_" in k]
        if unexpected or missing:
            raise SystemExit(f"resume weights do not match the adapter: missing {missing[:5]}, "
                             f"unexpected {unexpected[:5]}")
        opt.load_state_dict(blob["opt"])
        sched.load_state_dict(blob["sched"])
        if scaler is not None and blob["scaler"]:
            scaler.load_state_dict(blob["scaler"])
        random.setstate(blob["rng"]["py"])
        torch.set_rng_state(blob["rng"]["torch"])
        if blob["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(blob["rng"]["cuda"])
        state = blob["state"]
        print(f"resumed at update {state['update']}")

    model.train()
    while state["update"] < total_updates:
        t_step = time.time()
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(args.accum):
            ex = train_enc[order[state["cursor"]]]
            state["cursor"] += 1
            loss, n = row_loss(model, ex, device, dtype)
            state["tokens_seen"] += len(ex["input_ids"])
            loss = loss / args.accum
            (scaler.scale(loss) if scaler else loss).backward()
            acc_loss += loss.item()
        skipped = False
        if scaler:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            prev = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < prev  # inf/NaN grads: the optimiser step was skipped
        else:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        state["train_seconds"] += time.time() - t_step
        if torch.cuda.is_available():
            state["peak_gib"] = max(state["peak_gib"], torch.cuda.max_memory_allocated() / 2 ** 30)
        if skipped:
            # The schedule and the update count move only on real steps.
            state["skipped"] += 1
            state["skip_run"] = state.get("skip_run", 0) + 1
            if state["skip_run"] >= 10 or state["skipped"] > 40:
                raise SystemExit(f"fp16 overflow on {state['skip_run']} consecutive steps "
                                 f"({state['skipped']} in total): stopping. Check the data and the loss "
                                 f"scale; the last good checkpoint is in last/.")
            print(f"  [fp16] overflow at scale {prev:g}: step skipped ({state['skipped']} so far)", flush=True)
            continue
        state["skip_run"] = 0
        sched.step()
        state["update"] += 1
        print(f"update {state['update']}/{total_updates} loss {acc_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"tok/s {state['tokens_seen'] / max(1e-9, state['train_seconds']):.0f} "
              f"mem {state['peak_gib']:.1f} GiB", flush=True)
        if state["update"] % args.checkpoint_every == 0 or state["update"] == total_updates:
            metrics = evaluate(model, tok, dev_enc, dev_rows, device, dtype, args.dev_gen, args.dev_loss_rows)
            metrics["update"] = state["update"]
            state["history"].append(metrics)
            print("  dev:", json.dumps(metrics), flush=True)
            better = state["best"] is None or (metrics["action_match"], -metrics["dev_loss"]) > (
                state["best"]["action_match"], -state["best"]["dev_loss"])
            if better:
                state["best"] = metrics
            # last/ first, then best/: a kill between the two leaves best/ older
            # than state["best"] says, never newer.
            save_checkpoint(model, opt, sched, scaler, state, ckpt)
            if better:
                best_dir = os.path.join(args.out, "best")
                _swap(save_adapter_atomic(model, best_dir), best_dir)

    record = {
        "adapter": args.adapter, "args": vars(args), "base": args.base, "revision": args.revision,
        "chat_template_sha256": sha, "data_manifest": manifest, "train_rows": len(train_enc),
        "dev_rows": len(dev_enc), "rejected_over_max_length": [rej_t, rej_d],
        "updates": state["update"], "skipped_fp16_steps": state["skipped"],
        "tokens_per_second": state["tokens_seen"] / max(1e-9, state["train_seconds"]),
        "train_seconds": state["train_seconds"], "peak_memory_gib": state["peak_gib"],
        "dtype": str(dtype), "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "versions": versions(), "code_commit": code_commit(),
        "prompt_builder_hash": manifest.get("prompt_builder_hash"),
        "history": state["history"], "chosen_checkpoint": state["best"],
    }
    with open(os.path.join(args.out, "TRAINING_RECORD.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    with open(os.path.join(args.out, "best", "serving.json"), "w", encoding="utf-8") as f:
        json.dump({"HIVE_SERVING": "v2", "adapter_dir": os.path.basename(args.out.rstrip("/")) + "/best",
                   "prompt_builder_hash": manifest.get("prompt_builder_hash"), "base": args.base,
                   "base_revision": args.revision, "chat_template_sha256": sha,
                   "sampling": {"temperature": 0.7, "top_p": 0.9, "min_new_tokens": 10,
                                "top_k": "from the base generation_config",
                                "repetition_penalty": None}}, f, indent=2)
    print(f"done: best checkpoint at update {state['best'] and state['best']['update']} -> {args.out}/best")


if __name__ == "__main__":
    main()
