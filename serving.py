"""
serving.py -- the v2 serving contract (TRAINING_SPEC.md §4).

One function turns a runtime prompt (the user turn) into model input ids. The
v2 training script (fine_tuning.py) renders training rows with this same
function, so training bytes and serving bytes cannot drift apart.

The runtime uses it only when HIVE_SERVING=v2 (W1). Unset, every call site
runs exactly as before (legacy: raw text, prompt penalties, adapter on).
Under v2: decide() is chat-templated with no penalty and the adapter on;
think(), the phaser, the judge and the synthesizer run with the adapter off
(it was trained only for decide()); the phaser, judge and synthesizer are
chat-templated with no penalty; think() keeps its raw-text cache seed.
"""
import contextlib
import hashlib
import os


def is_v2() -> bool:
    """True when the runtime serves the v2 contract (HIVE_SERVING=v2)."""
    return os.environ.get("HIVE_SERVING", "legacy").strip().lower() == "v2"


def stop_ids(tokenizer):
    """The two ids a v2 turn ends on (S3), passed to generate() explicitly."""
    return [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]


@contextlib.contextmanager
def adapter_off(model):
    """Under v2, run the block with the LoRA adapter disabled. Legacy: no
    change. A model that cannot disable an adapter raises -- a merged adapter
    would otherwise run here with the adapter on -- unless it is the base arm
    (main.load_model marks it `hive_no_adapter`)."""
    if not is_v2():
        yield
    elif hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    elif getattr(model, "hive_no_adapter", False):
        yield
    else:
        raise TypeError("HIVE_SERVING=v2 needs an unmerged PEFT model (or the base arm, "
                        "adapter path 'none'): this model cannot switch its adapter off")


def prompt_builder_hash(root=None) -> str:
    """S6: a hash over the runtime sources that shape prompts (whole files).
    The data build records it in every row; an adapter's serving.json carries
    the one it was trained on."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for f in ("agent_node.py", "orchestrator.py", "text_utils.py"):
        with open(os.path.join(root, f), "rb") as fh:
            # LF everywhere: a Windows checkout (CRLF) and Kaggle's (LF) must agree.
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def encode(tokenizer, user_text: str, device, prefill: str = ""):
    """generate() inputs for a v2 call: the rendered chat prompt as a batch
    of one, with its attention mask."""
    import torch
    ids = torch.tensor([render(tokenizer, user_text, prefill=prefill)], device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def call_fn(model, tokenizer, greedy=False, prefill=""):
    """The v2 prompt -> text call for the judge (greedy, prefill "VERDICT:")
    and the synthesizer (sampled): chat template, adapter off, no prompt
    penalty, explicit stops. A prompt ending in `prefill` has it moved out of
    the user turn into the assistant turn (S9), so the model's first words are
    the ruling; the judge prepends it to the completion as before."""
    import torch

    def call(prompt: str) -> str:
        pre = ""
        if prefill and prompt.endswith(prefill):
            prompt, pre = prompt[:-len(prefill)], prefill
        inputs = encode(tokenizer, prompt, model.device, prefill=pre)
        sampling = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.7, top_p=0.9)
        with torch.no_grad(), adapter_off(model):
            out = model.generate(**inputs, max_new_tokens=300, min_new_tokens=10, **sampling,
                                 eos_token_id=stop_ids(tokenizer), pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return call

# Qwen3's generation prompt with thinking off: an empty think block, then the
# answer. Training and serving must both end the prompt exactly here.
EMPTY_THINK_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def chat_template_sha256(tokenizer) -> str:
    """sha256 of the tokenizer's chat template text (S5 pin)."""
    template = getattr(tokenizer, "chat_template", None) or ""
    if isinstance(template, dict):  # some tokenizers keep named templates
        template = template.get("default") or next(iter(template.values()), "")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def render_text(tokenizer, user_text: str, thinking: bool = False, prefill: str = "") -> str:
    """The exact prompt string the model reads: one user turn, no system
    turn, generation prompt appended, then an optional assistant prefill."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    if not thinking and not text.endswith(EMPTY_THINK_TAIL):
        raise ValueError("chat template did not end a thinking-off prompt with the "
                         "empty think block; check the pinned tokenizer")
    return text + prefill


def render(tokenizer, user_text: str, thinking: bool = False, prefill: str = ""):
    """Input ids (list of int) for render_text."""
    text = render_text(tokenizer, user_text, thinking=thinking, prefill=prefill)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode_training_pair(tokenizer, user_text: str, response: str, max_length: int):
    """{"input_ids", "labels"} for one supervised row (TRAINING_SPEC §11).

    The prompt is rendered by render(); the full conversation by the chat
    template with the assistant turn. The prompt must be an exact prefix of
    the conversation, loss falls on the response and its <|im_end|>, and the
    newline the template writes after <|im_end|> is masked. Rows longer than
    max_length are rejected, never truncated."""
    # The template lstrips a leading newline and splits on </think>: either
    # would make the trained target differ from the row. Loud, not silent.
    if response != response.lstrip("\n") or "</think>" in response or "<think>" in response:
        raise ValueError("response starts with a newline or contains a think tag")
    prompt_ids = render(tokenizer, user_text)
    full_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text},
         {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt is not a prefix of prompt+response under this template")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Supervise up to and including the LAST <|im_end|>; mask what follows it.
    try:
        last_end = len(full_ids) - 1 - full_ids[::-1].index(im_end)
    except ValueError:
        raise ValueError("no <|im_end|> after the response")
    if last_end < len(prompt_ids):
        raise ValueError("empty supervised response")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):last_end + 1]
    labels += [-100] * (len(full_ids) - len(labels))
    if len(full_ids) > max_length:
        raise ValueError(f"{len(full_ids)} tokens exceeds max_length {max_length}; "
                         "truncation is forbidden")
    return {"input_ids": full_ids, "labels": labels}
