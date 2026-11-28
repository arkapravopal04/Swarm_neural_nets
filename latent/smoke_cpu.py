"""smoke_cpu.py -- kv.py + checks.py on a tiny random Qwen3, CPU only, no download.

Exits non-zero on any failure. Also checks that captured, rotated queries
reproduce the model's own attention (vs output_attentions on an eager copy).
"""

import copy
import os
import sys

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv  # noqa: E402
from checks import run_checks  # noqa: E402

torch.manual_seed(0)
cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  max_position_embeddings=4096, initializer_range=0.1, tie_word_embeddings=False)
model = Qwen3ForCausalLM._from_config(copy.deepcopy(cfg), attn_implementation="sdpa").to("cpu", torch.float32).eval()
eager = Qwen3ForCausalLM._from_config(copy.deepcopy(cfg), attn_implementation="eager").to("cpu", torch.float32).eval()
eager.load_state_dict(model.state_dict())

a = torch.randint(0, 512, (1, 60))
b = torch.randint(0, 512, (1, 10))
fails = []

res = run_checks(model, a, b)
if not res["passed"]:
    fails.append("gate")


def ref_scores(ids, rows, cols):
    """Reference: eager attention of `rows` query tokens onto the first `cols` keys."""
    with torch.no_grad():
        att = eager(input_ids=ids, output_attentions=True).attentions
    # scores() softmaxes over the memory slots only, so renormalise each row over :cols
    per = [x[0, :, rows, :cols] for x in att]
    per = [(p / p.sum(-1, keepdim=True)).sum(dim=(0, 1)) for p in per]
    return torch.stack([p / p.sum() for p in per]).mean(0)


mem_a, writer = kv.build(model, a, window=8)
d_w = float((writer - ref_scores(a, slice(-8, None), 60)).abs().max())
reader = kv.reader_scores(model, mem_a, b)
d_r = float((reader - ref_scores(torch.cat([a, b], 1), slice(60, None), 60)).abs().max())
print(f"scores vs eager attention: writer {d_w:.2e}, reader {d_r:.2e}")
if d_w > 1e-4 or d_r > 1e-4 or writer.shape != (60,) or reader.shape != (60,):
    fails.append("scores")

# chunked prefill must equal one-shot prefill (logits and captured queries)
_, l1, q1 = kv.run(model, a, logits="all", capture_q=True)
_, l2, q2 = kv.run(model, a, logits="all", capture_q=True, chunk=16)
d_c = max(float((l1 - l2).abs().max()), max(float((x - y).abs().max()) for x, y in zip(q1, q2)))
print(f"chunked vs one-shot prefill: max diff {d_c:.2e}")
if d_c > 1e-4 or l1.shape != (1, 60, 512) or q1[0].shape != (1, 4, 60, 16):
    fails.append("chunk")

# negative control: renumbering without re-rotating keys must change B's logits
_, lb, _ = kv.run(model, b, mem_a, logits="all")
bad = kv.Memory(mem_a.clone().cache, torch.arange(1000, 1060), 1060)
_, lbad, _ = kv.run(model, b, bad, logits="all")
print(f"unrotated +1000 shift (should be large): max KL {float(kv.kl(lbad[0], lb[0]).max()):.2e}")

# exercise the rest of the API once
n0 = mem_a.slots
for idx in (kv.pick_last(mem_a, 16), kv.pick_top(writer, 16, lo=kv.SINKS), kv.pick_top(reader, 16)):
    cut = kv.keep(mem_a, idx)
    rs = kv.restamp(model, cut, start=0)
    toks, _ = kv.decode(model, rs, b, max_new=5, stop_ids=[])
    if not (cut.slots <= 20 and cut.next_pos == 60 and rs.next_pos == cut.slots and len(toks) == 5):
        fails.append("cut")
# join: b re-stamped after a must equal one pass where b's rows cannot see a
mem_b, _, _ = kv.run(model, b, logits=None)
j = kv.join(model, mem_a, mem_b)
_, lj, _ = kv.run(model, b, j, logits="all")
ids = torch.cat([a, b, b], 1)
allow = torch.tril(torch.ones(80, 80, dtype=torch.bool))
allow[60:70, :60] = False
with torch.no_grad():
    ref = model(input_ids=ids, position_ids=torch.arange(80)[None],
                attention_mask=torch.zeros(1, 1, 80, 80).masked_fill(~allow, float("-inf"))).logits[:, 70:]
d_j = float((lj - ref).abs().max())
print(f"join vs b-blind-to-a reference: max diff {d_j:.2e}")
if d_j > 1e-4 or j.pos.tolist() != list(range(70)) or j.next_pos != 70:
    fails.append("join")
assert mem_a.slots == n0 and mem_a.next_pos == 60, "a Memory was mutated"
g1, _ = kv.decode(model, None, a, max_new=6, stop_ids=[])
g2, m2 = kv.decode(model, mem_a, b, max_new=6, stop_ids=[g1[0]] if g1 else [])
s1, _ = kv.decode(model, mem_a, b, max_new=6, stop_ids=[], temperature=0.8, top_p=0.9, seed=1)
s2, _ = kv.decode(model, mem_a, b, max_new=6, stop_ids=[], temperature=0.8, top_p=0.9, seed=1)
print(f"decode greedy {g1} | sampled {s1}")
if s1 != s2 or len(g1) != 6 or m2.next_pos != 70 + len(g2):
    fails.append("decode")

print("SMOKE:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
