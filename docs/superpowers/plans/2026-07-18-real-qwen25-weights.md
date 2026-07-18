# Real Qwen2.5-0.5B Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the server-local Qwen2.5-0.5B-Instruct tokenizer and BF16 safetensors into the project's own Qwen implementation, run greedy cached chat inference, and optionally prove numerical parity with Transformers.

**Architecture:** Add optional adapters for the real BPE tokenizer and safetensors checkpoint while leaving whiteboard imports dependency-free. Add a generic greedy generator and a real-model CLI; isolate Transformers to one verification script.

**Tech Stack:** Python 3.11, PyTorch 2.10.0+cu128, safetensors 0.4.5, tokenizers 0.19.1, unittest; optional Transformers 4.43.1.

## Global Constraints

- Production modules must not import `transformers`.
- Keep all existing whiteboard tests and behavior passing.
- Required model path is `/ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct`.
- Only `lm_head.weight` may be absent from the checkpoint because embeddings are tied.
- Fail before partial loading on any other missing, unexpected, or shape-mismatched tensor.
- Develop and test locally, rsync to `/ai/projects/Infer-DaseSS`, then run GPU verification.
- Use `/root/.pyenv/shims/python3.11` on the server.

---

### Task 1: Real Qwen tokenizer adapter

**Files:**
- Create: `toy_qwen/qwen_tokenizer.py`
- Create: `tests/test_qwen_tokenizer.py`

**Interfaces:**
- Produces: `render_qwen_chat(messages, add_generation_prompt=True) -> str` and `QwenTokenizerAdapter.from_model_dir(path)` with `encode_chat`, `decode`, `token`, `eos_token_id`.

- [ ] **Step 1: Write failing tests**

Test the exact no-tools template and validation without importing tokenizers:

```python
def test_default_system_chat_template(self):
    rendered = render_qwen_chat([{"role": "user", "content": "中国的首都是哪里？"}])
    self.assertEqual(rendered,
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n中国的首都是哪里？<|im_end|>\n"
        "<|im_start|>assistant\n")

def test_explicit_system_and_assistant_history(self):
    messages = [{"role":"system","content":"简洁回答。"},
                {"role":"user","content":"你好"},
                {"role":"assistant","content":"你好！"},
                {"role":"user","content":"继续"}]
    text = render_qwen_chat(messages)
    self.assertTrue(text.startswith("<|im_start|>system\n简洁回答。<|im_end|>\n"))
    self.assertTrue(text.endswith("<|im_start|>assistant\n"))

def test_rejects_empty_or_unsupported_roles(self):
    with self.assertRaises(ValueError): render_qwen_chat([])
    with self.assertRaisesRegex(ValueError, "tool"):
        render_qwen_chat([{"role":"tool","content":"x"}])
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_qwen_tokenizer -v`; expect missing module failure.

- [ ] **Step 3: Implement template and lazy dependency**

Build each message as `<|im_start|>{role}\n{content}<|im_end|>\n`, inject the documented default system message when absent, and append `<|im_start|>assistant\n` when requested. Import `tokenizers.Tokenizer` only inside `from_model_dir`; if absent, raise `RuntimeError("install tokenizers==0.19.1")`. Read `tokenizer_config.json`, resolve EOS token through `token_to_id`, and call `Tokenizer.from_file(tokenizer.json)`.

- [ ] **Step 4: Run tests and commit**

Run the focused test and full discovery; expect all passing.

```bash
git add toy_qwen/qwen_tokenizer.py tests/test_qwen_tokenizer.py
git commit -m "feat: add real qwen tokenizer adapter"
```

---

### Task 2: Strict safetensors checkpoint loading

**Files:**
- Create: `toy_qwen/pretrained.py`
- Create: `tests/test_pretrained.py`
- Modify: `toy_qwen/__init__.py`

**Interfaces:**
- Produces: `CheckpointReport`, `validate_checkpoint(model, state_dict)`, `load_pretrained_qwen(model_path, device, dtype)`.

- [ ] **Step 1: Write failing validation tests**

Use a whiteboard model's cloned state dict as a small fixture. Remove only `lm_head.weight` and assert validation succeeds when tied; remove `model.norm.weight` and assert failure; add `bad.weight` and assert failure; replace one tensor with a wrong shape and assert the error names that key. Assert untied models cannot omit LM Head.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_pretrained -v`; expect missing module failure.

- [ ] **Step 3: Implement strict validation**

```python
@dataclass(frozen=True)
class CheckpointReport:
    tensor_count: int
    expected_tied_missing: tuple[str, ...]

def validate_checkpoint(model, state_dict):
    expected = model.state_dict()
    missing = set(expected) - set(state_dict)
    unexpected = set(state_dict) - set(expected)
    allowed = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    mismatched = {key: (tuple(state_dict[key].shape), tuple(expected[key].shape))
                  for key in set(expected) & set(state_dict)
                  if state_dict[key].shape != expected[key].shape}
    if missing - allowed or unexpected or mismatched:
        raise ValueError(...)
    return CheckpointReport(len(state_dict), tuple(sorted(missing & allowed)))
```

- [ ] **Step 4: Implement lazy real loader**

Check `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors`. Map dtype strings `float32` and `bfloat16`; reject CUDA when unavailable. Import `safetensors.torch.load_file` lazily with an installation hint. Instantiate from official config, move to requested dtype before copying, validate, call `load_state_dict(strict=False)`, reassert tied parameter identity, move to device, eval, and return `(model, report)`.

- [ ] **Step 5: Run tests and commit**

Run focused and full tests; expect all passing.

```bash
git add toy_qwen/pretrained.py toy_qwen/__init__.py tests/test_pretrained.py
git commit -m "feat: load strict qwen safetensors checkpoints"
```

---

### Task 3: Generic cached greedy generation

**Files:**
- Create: `toy_qwen/generation.py`
- Create: `tests/test_generation.py`

**Interfaces:**
- Produces: `GenerationStep`, `GenerationResult`, `greedy_generate(model, input_ids, eos_token_id, max_new_tokens, top_k=5)`.

- [ ] **Step 1: Write failing generation tests**

Use the whiteboard model to assert prefill cache exists, one generated token is ID 5, step top-k is sorted, `max_new_tokens=0` fails, and setting EOS to 5 stops after one token. Compare cached second-step logits with an uncached concatenated forward.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_generation -v`; expect missing module failure.

- [ ] **Step 3: Implement minimal generator**

Run the full prompt once with `use_cache=True`, take last logits, record top-k, append argmax, stop on EOS, then feed only the new token with returned cache. Validate rank-2 batch size 1, positive max tokens, and model position limit. Return generated IDs, steps, prefill logits shape, and first/last cache shapes; detach logged values to CPU.

- [ ] **Step 4: Run tests and commit**

Run focused and full tests; expect all passing.

```bash
git add toy_qwen/generation.py tests/test_generation.py
git commit -m "feat: add cached greedy generation"
```

---

### Task 4: Real-model CLI and dependency manifests

**Files:**
- Create: `real_qwen_inference.py`
- Create: `requirements-real.txt`
- Create: `requirements-verify.txt`
- Modify: `README.md`
- Create: `tests/test_real_cli.py`

**Interfaces:**
- Consumes all production adapters.
- Produces CLI arguments `--model-path`, `--prompt`, `--system-prompt`, `--device`, `--dtype`, `--max-new-tokens`, `--trace-shapes`.

- [ ] **Step 1: Write parser/help tests**

Import `build_parser()` without optional dependencies and assert defaults: prompt `中国的首都是哪里？`, device `cuda`, dtype `bfloat16`, max new tokens 32. Assert `--help` succeeds in subprocess without importing safetensors/tokenizers.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_real_cli -v`; expect missing script failure.

- [ ] **Step 3: Implement CLI**

Keep optional imports inside `main`. Print config summary, checkpoint report, rendered chat, input IDs, prefill/cache shapes, each step's top-5 decoded token and logit, generated IDs, and decoded generated text. Catch CUDA OOM only to add actionable context, then re-raise.

- [ ] **Step 4: Add manifests and README commands**

`requirements-real.txt` contains exact safetensors/tokenizers pins. `requirements-verify.txt` includes `-r requirements-real.txt` plus `transformers==4.43.1`. Document Python 3.11 venv, local rsync, real CLI, and whiteboard CLI separately.

- [ ] **Step 5: Run tests and commit**

Run full local suite and `python3 real_qwen_inference.py --help`.

```bash
git add real_qwen_inference.py requirements-real.txt requirements-verify.txt README.md tests/test_real_cli.py
git commit -m "feat: add real qwen inference command"
```

---

### Task 5: Server smoke test and optional Transformers oracle

**Files:**
- Create: `verification/compare_transformers.py`
- Create: `verification/README.md`

**Interfaces:**
- Produces an independent parity report; never imported by `toy_qwen`.

- [ ] **Step 1: Rsync code and create server environment**

```bash
rsync -avz --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' \
  /mnt/d/vDesktop/Infer-DaseSS/ dase314-server:/ai/projects/Infer-DaseSS/
ssh dase314-server "cd /ai/projects/Infer-DaseSS && /root/.pyenv/shims/python3.11 -m venv --system-site-packages .venv-real && .venv-real/bin/pip install -r requirements-real.txt"
```

- [ ] **Step 2: Run existing and real-model smoke tests**

Run server unittest discovery, then the real CLI with the fixed model path, BF16 CUDA, prompt `中国的首都是哪里？`, and 16 new tokens. Expected: checkpoint count 290, only expected tied alias missing, non-empty generated text, no exception.

- [ ] **Step 3: Implement oracle script**

Load the same tokenizer-rendered IDs. Load custom and Transformers models in float32 with eager attention, run no-grad single forward, print max absolute/relative error, top-10 IDs for both, and greedy token IDs. Exit nonzero unless top-10 IDs and first greedy token match and max absolute error is below a reviewed tolerance initially set to `1e-3`.

- [ ] **Step 4: Install verify dependency and run parity**

Install `requirements-verify.txt`, execute the oracle, and if tolerance fails use systematic debugging rather than relaxing it without evidence.

- [ ] **Step 5: Final verification and commit**

Run local full tests, server full tests, BF16 real CLI, oracle comparison, `rg -n '^.*import transformers' toy_qwen real_qwen_inference.py` (no matches), and `git diff --check`.

```bash
git add verification
git commit -m "test: verify custom qwen against transformers"
```
