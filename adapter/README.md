# Hive decision adapter

LoRA adapter for [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) that teaches the model Hive's agent decision format: given an agent's role, task and recent thoughts, answer with exactly one `ACTION:` (`THINK`, `SPAWN`, `TOOL`, `REPORT`, `DIE`) and its `PAYLOAD:`.

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B`, loaded in fp16 |
| Method | LoRA, supervised fine-tuning with TRL |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Training data | 836 instruction/response pairs in [`../fine_tune/`](../fine_tune) |
| Size | ~47 MB |

`main.py` loads it from here by default. Set `HIVE_ADAPTER_PATH` to use a different one.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="float16", device_map="auto")
model = PeftModel.from_pretrained(base, "adapter")
```

The training script is not in the repo yet; the data and resulting weights are.
