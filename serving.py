"""
serving.py -- the v2 serving contract's chat helper (TRAINING_SPEC.md §4 S1).

One function turns a runtime prompt (the user turn) into model input ids. The
v2 training script (fine_tuning.py) renders training rows with this same
function, so training bytes and serving bytes cannot drift apart. The runtime
call sites adopt it behind HIVE_SERVING=v2 (work item W1); until then only
training and the probes use it.
"""
import hashlib

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
