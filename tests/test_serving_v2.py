"""W1: HIVE_SERVING=v2 serves the model the way the v2 adapter was trained
(TRAINING_SPEC §3-§4), and the default (legacy) path is unchanged."""
import contextlib
import importlib
import os
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The phaser module imports sentence_transformers at the top; stub it where it
# is not installed (as tests/test_phaser_text_cleanup.py does).
if importlib.util.find_spec("sentence_transformers") is None:
    sys.modules.setdefault("sentence_transformers", SimpleNamespace(SentenceTransformer=object))

import serving  # noqa: E402
from agent_node import Agent  # noqa: E402
from colony_state import AgentNode  # noqa: E402
from event_queue import Messenger  # noqa: E402

TOK_DIR = os.path.join(ROOT, "fine_tuned_params_v1")
needs_tok = pytest.mark.skipif(not os.path.isdir(TOK_DIR), reason="needs the local Qwen3 tokenizer")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOK_DIR)


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setenv("HIVE_SERVING", "v2")


@pytest.fixture
def legacy(monkeypatch):
    monkeypatch.delenv("HIVE_SERVING", raising=False)


class FakeModel:
    """Records what generate() / forward() were called with, and whether the
    adapter was off at the time."""
    device = torch.device("cpu")

    def __init__(self, tok, reply=""):
        self.tok, self.reply, self.off, self.calls, self.forward_off = tok, reply, False, [], []

    @contextlib.contextmanager
    def disable_adapter(self):
        self.off = True
        try:
            yield
        finally:
            self.off = False

    def generate(self, input_ids=None, attention_mask=None, **kw):
        self.calls.append({"ids": input_ids[0].tolist(), "off": self.off, **kw})
        reply = self.tok(self.reply, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return torch.cat([input_ids, reply], dim=1)

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        self.forward_off.append(self.off)
        return SimpleNamespace(logits=torch.zeros(1, input_ids.shape[1], len(self.tok)), past_key_values="kv")


def _agent(tok, model, prompt="Your Task: find the current.\nYour next action:"):
    node = AgentNode(agent_id="a-1", role="executor", status="running", parent_id="p",
                     task="find the current", task_id="t-1", generation=1)
    agent = Agent(tokeniser=tok, model=model, message=Messenger(), node=node)
    agent._build_prompt = lambda *a, **k: prompt
    return agent, prompt


def test_legacy_is_the_default(legacy):
    assert not serving.is_v2()


@needs_tok
def test_decide_v2_is_the_trained_chat_turn(tok, v2):
    model = FakeModel(tok, "ACTION: REPORT\nPAYLOAD: The current is 1 A.<|im_end|>")
    agent, prompt = _agent(tok, model)
    action, payload = agent.decide()
    call = model.calls[0]
    assert call["ids"] == serving.render(tok, prompt)          # byte-identical to training
    assert "repetition_penalty" not in call
    assert call["eos_token_id"] == serving.stop_ids(tok)
    assert call["off"] is False                                 # decide() keeps the adapter on
    assert (action, payload) == ("REPORT", "The current is 1 A.")


@needs_tok
def test_decide_legacy_is_unchanged(tok, legacy):
    model = FakeModel(tok, "ACTION: REPORT\nPAYLOAD: The current is 1 A.")
    agent, prompt = _agent(tok, model)
    agent.decide()
    call = model.calls[0]
    assert call["ids"] == tok(prompt)["input_ids"]
    assert call["repetition_penalty"] == 1.15 and "eos_token_id" not in call


@needs_tok
@pytest.mark.parametrize("mode", ["v2", "legacy"])
def test_think_runs_with_the_adapter_off_only_under_v2(tok, monkeypatch, mode):
    if mode == "v2":
        monkeypatch.setenv("HIVE_SERVING", "v2")
    else:
        monkeypatch.delenv("HIVE_SERVING", raising=False)
    model = FakeModel(tok)
    agent, _ = _agent(tok, model)
    agent.think()
    assert model.forward_off and all(off is (mode == "v2") for off in model.forward_off)


@needs_tok
def test_judge_call_moves_its_prefill_into_the_assistant_turn(tok, v2):
    from judge import Judge
    model = FakeModel(tok, " accept. The figures check out.")
    call = serving.call_fn(model, tok, greedy=True, prefill=Judge.VERDICT_PREFILL)
    out = call("Critique the output.\nRespond with one verdict line.\n" + Judge.VERDICT_PREFILL)
    rec = model.calls[0]
    assert rec["ids"] == serving.render(tok, "Critique the output.\nRespond with one verdict line.\n",
                                        prefill=Judge.VERDICT_PREFILL)
    assert rec["do_sample"] is False and rec["off"] is True and "repetition_penalty" not in rec
    assert out == "accept. The figures check out."


@needs_tok
def test_synthesizer_call_is_sampled_with_the_adapter_off(tok, v2):
    model = FakeModel(tok, "Final answer text.")
    serving.call_fn(model, tok)("Combine these results.")
    rec = model.calls[0]
    assert rec["do_sample"] is True and rec["off"] is True and "repetition_penalty" not in rec


@needs_tok
@pytest.mark.parametrize("mode", ["v2", "legacy"])
def test_phaser_call(tok, monkeypatch, mode):
    from problem_phaser import Problem_Phaser
    if mode == "v2":
        monkeypatch.setenv("HIVE_SERVING", "v2")
    else:
        monkeypatch.delenv("HIVE_SERVING", raising=False)
    phaser = Problem_Phaser.__new__(Problem_Phaser)
    phaser.tokeniser, phaser.device = tok, torch.device("cpu")
    phaser.embed_dim = 4  # the embedding after the call is not under test
    phaser.embed_model = SimpleNamespace(encode=lambda *a, **k: torch.zeros(4).numpy())
    phaser.llm = FakeModel(tok, "Engineering > Electrical (Focus: circuits)")
    phaser._get_domain("A 230 V heater draws 10 A. Find its power.")
    rec = phaser.llm.calls[0]
    if mode == "v2":
        assert rec["off"] is True and "repetition_penalty" not in rec and "no_repeat_ngram_size" not in rec
        assert rec["ids"][-len(tok(serving.EMPTY_THINK_TAIL, add_special_tokens=False)["input_ids"]):] \
            == tok(serving.EMPTY_THINK_TAIL, add_special_tokens=False)["input_ids"]
        assert rec["stop_strings"] == Problem_Phaser.STOP_STRINGS  # its stop strings stay
    else:
        assert rec["off"] is False and rec["repetition_penalty"] == Problem_Phaser.REPETITION_PENALTY
        assert rec["no_repeat_ngram_size"] == 4


def test_adapter_off_raises_for_a_model_that_cannot_switch_it_off(v2):
    # A merged adapter would otherwise run think()/judge/phaser with it on.
    with pytest.raises(TypeError):
        with serving.adapter_off(object()):
            pass


def test_adapter_off_lets_the_base_arm_through(v2):
    base = SimpleNamespace(hive_no_adapter=True)  # main.load_model("none") marks it
    with serving.adapter_off(base):
        pass


def test_adapter_off_does_nothing_under_legacy(legacy):
    with serving.adapter_off(object()):
        pass
