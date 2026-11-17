"""Run third_party/deer/vllm_deer_logged.py end to end with a fake vLLM.

No GPU needed. The fake model "thinks" in scripted chunks, and its trial
answers have scripted confidences, so we can check the exit decisions and
the per-checkpoint logs. Needs the real DeepSeek tokenizer (downloaded once
from the HF Hub, a few MB); skipped if unavailable.
"""
import json
import math
import runpy
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "third_party" / "deer" / "vllm_deer_logged.py"
TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Per question: thinking chunks (the last one ends with </think>), and the
# (trial answer, confidence) returned at each checkpoint.
SCRIPT_Q = {
    "Q0": {"chunks": ["16-3-4=9, 9*2=18.\n\n", " check: 18.\n\n", " done.\n\n"],
           "trials": [("18", 0.97), ("18", 0.99)]},
    "Q1": {"chunks": ["maybe 7.\n\n", " no, it is 8.\n\n", " yes 8.\n\n"],
           "trials": [("7", 0.60), ("8", 0.90)]},
}


class Logprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeLLM:
    def __init__(self, *a, **k):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(TOKENIZER)
        self.think_id = self.tok.encode("</think>", add_special_tokens=False)[0]
        self.wait_id = self.tok.encode("Wait", add_special_tokens=False)[0]

    def _out(self, text, ids, logprobs=None):
        comp = types.SimpleNamespace(text=text, token_ids=ids, logprobs=logprobs, finish_reason="stop")
        return types.SimpleNamespace(outputs=[comp], request_id="r")

    def generate(self, prompts, params, use_tqdm=False):
        if not isinstance(params, list):
            params = [params] * len(prompts)
        outs = []
        for prompt, sp in zip(prompts, params):
            q = "Q0" if "Q0" in prompt else ("Q1" if "Q1" in prompt else None)
            if q is None:  # "Hello, world!" warm-up
                outs.append(self._out(" hi", [1]))
                continue
            gen = prompt.split("<think>\n", 1)[1]
            spec = SCRIPT_Q[q]
            if gen.endswith("\\boxed"):  # trial-answer probe
                n_done = gen.count("Wait")  # checkpoints already passed
                ans, conf = spec["trials"][min(n_done, len(spec["trials"]) - 1)]
                ids = self.tok.encode("{" + ans, add_special_tokens=False)
                lps = [{i: Logprob(math.log(conf))} for i in ids]
                outs.append(self._out("{" + ans, ids, lps))
            elif gen.endswith("\\boxed{"):  # --final_answer boxed: answer prompted inside \boxed{
                ans = spec["trials"][-1][0]
                text = ans + "}."
                outs.append(self._out(text, self.tok.encode(text, add_special_tokens=False) + [self.tok.eos_token_id]))
            elif "</think>" in gen:  # final answer
                ans = spec["trials"][-1][0]
                text = f"The answer is \\boxed{{{ans}}}."
                outs.append(self._out(text, self.tok.encode(text, add_special_tokens=False) + [self.tok.eos_token_id]))
            else:  # thinking chunk
                n_done = gen.count("Wait")
                chunk = spec["chunks"][n_done]
                last = n_done == len(spec["chunks"]) - 1
                ids = self.tok.encode(chunk, add_special_tokens=False) + [self.think_id if last else self.wait_id]
                outs.append(self._out(chunk, ids))
        return outs


def fake_modules():
    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = lambda **k: types.SimpleNamespace(**k)
    vllm_outputs = types.ModuleType("vllm.outputs")
    vllm_outputs.CompletionOutput = object

    torch = types.ModuleType("torch")
    torch.tensor = lambda x: x
    torch.exp = lambda x: types.SimpleNamespace(item=lambda: math.exp(x))
    torch.manual_seed = lambda s: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.Generator = lambda: types.SimpleNamespace(manual_seed=lambda s: None)
    nn = types.ModuleType("torch.nn")
    fn = types.ModuleType("torch.nn.functional")
    torch.nn = nn
    nn.functional = fn
    return {"vllm": vllm, "vllm.outputs": vllm_outputs, "torch": torch,
            "torch.nn": nn, "torch.nn.functional": fn}


def run_script(tmp_path, monkeypatch, threshold, extra=()):
    pytest.importorskip("transformers")
    try:
        FakeLLM()
    except Exception as e:  # offline, no cached tokenizer
        pytest.skip(f"tokenizer unavailable: {e}")
    data = tmp_path / "data" / "toy"
    data.mkdir(parents=True)
    with open(data / "test.jsonl", "w") as f:
        f.write(json.dumps({"problem": "Q0 eggs?", "answer": "18"}) + "\n")
        f.write(json.dumps({"problem": "Q1 count?", "answer": "8"}) + "\n")
    out = tmp_path / "run.jsonl"
    for name, mod in fake_modules().items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--model_name_or_path", TOKENIZER, "--dataset_dir", str(tmp_path / "data"),
        "--dataset", "toy", "--threshold", str(threshold), "--output_file", str(out),
        "--max_generated_tokens", "2000", *extra])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return [json.loads(line) for line in open(out)], json.load(open(tmp_path / "run.meta.json"))


def test_deer_exits_when_confident(tmp_path, monkeypatch):
    recs, meta = run_script(tmp_path, monkeypatch, threshold=0.95)
    assert [r["idx"] for r in recs] == [0, 1]
    q0, q1 = recs
    # Q0: confident (0.97) at the first checkpoint -> exits after 1 chunk
    assert q0["exit_reason"] == "confident"
    assert len(q0["checkpoints"]) == 1
    assert q0["checkpoints"][0]["trial_answer"] == "18"
    assert q0["checkpoints"][0]["confidence"] == pytest.approx(0.97, abs=1e-6)
    assert "check: 18" not in q0["generated_responses"][0]
    # Q1: never above 0.95 -> thinks to the natural end
    assert q1["exit_reason"] == "natural"
    assert [c["trial_answer"] for c in q1["checkpoints"]] == ["7", "8"]
    assert q1["checkpoints"][0]["thinking_tokens"] < q1["checkpoints"][1]["thinking_tokens"]
    # final thinking length covers everything up to </think>
    assert q1["num_thinking_tokens"] > q1["checkpoints"][-1]["thinking_tokens"]
    assert q0["num_thinking_tokens"] >= q0["checkpoints"][0]["thinking_tokens"]
    assert q0["num_answer_tokens"] > 0
    assert meta["n_questions"] == 2 and meta["wall_time_s"] >= 0


def test_never_exit_logs_every_checkpoint(tmp_path, monkeypatch):
    recs, _ = run_script(tmp_path, monkeypatch, threshold=1.1)
    assert all(r["exit_reason"] == "natural" for r in recs)
    assert [len(r["checkpoints"]) for r in recs] == [2, 2]

    # The output scores cleanly with our pipeline.
    from deer_eval.metrics import annotate, summarize
    s = summarize(annotate(recs))
    assert s["accuracy"] == 1.0
    assert s["avg_compute_tokens"] > s["avg_output_tokens"]


def test_offset_keeps_global_index(tmp_path, monkeypatch):
    recs, meta = run_script(tmp_path, monkeypatch, threshold=0.95, extra=["--offset", "1"])
    assert [r["idx"] for r in recs] == [1]
    assert recs[0]["question"].startswith("Q1")
    assert meta["n_questions"] == 1


def test_final_answer_trial_keeps_trial_answer(tmp_path, monkeypatch):
    recs, _ = run_script(tmp_path, monkeypatch, threshold=0.95, extra=["--final_answer", "trial"])
    q0, q1 = recs
    # Q0 exits early: the answer is the boxed trial answer, nothing regenerated
    assert q0["exit_reason"] == "confident"
    assert q0["generated_responses"][0].endswith("</think>\n\n**Final Answer**\n\\boxed{18}")
    assert "The answer is" not in q0["generated_responses"][0]
    # Q1 never exits early, so its answer is generated as usual
    assert q1["exit_reason"] == "natural"
    assert "The answer is \\boxed{8}" in q1["generated_responses"][0]
    from deer_eval.metrics import annotate
    assert all(r["correct"] for r in annotate(recs))


def test_final_answer_boxed_prompts_box(tmp_path, monkeypatch):
    recs, _ = run_script(tmp_path, monkeypatch, threshold=0.95, extra=["--final_answer", "boxed"])
    for r in recs:
        assert "</think>\n\n**Final Answer**\n\\boxed{" in r["generated_responses"][0]
    from deer_eval.metrics import annotate
    assert all(r["correct"] for r in annotate(recs))
