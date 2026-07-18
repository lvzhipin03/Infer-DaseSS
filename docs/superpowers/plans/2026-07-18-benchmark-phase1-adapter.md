# Benchmark Phase 1 Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the inference-optimization benchmark call the repository's existing handwritten `toy_qwen` implementation through a correctness-first `StudentEngine` adapter.

**Architecture:** `student_release/student_engine.py` is a thin adapter that adds the repository root to its import path, loads the real tokenizer/model through `toy_qwen`, and processes prompt lists sequentially with the existing cached greedy generator. Phase 1 deliberately leaves eager attention and batch-one computation unchanged while satisfying the benchmark interface and fixed-step output contract.

**Tech Stack:** Python 3.11, PyTorch, safetensors, tokenizers, unittest, benchmark static validator; server-local Qwen2.5-0.5B-Instruct.

## Global Constraints

- `student_release` directly depends on repository-root `toy_qwen`; do not copy the model implementation.
- Production inference must not call Hugging Face model forward/generate or any full inference framework.
- `StudentEngine.generate` must preserve prompt order and return continuation-only `list[str]`.
- Do not read or branch on `suite_name`, benchmark files, case IDs, keywords, or expected answers.
- Use one inference policy for every suite: Qwen chat template, greedy decoding, fixed token budget, skipped special tokens.
- Phase 1 accepts `attn_implementation` but continues to execute existing eager attention.
- Keep every existing whiteboard and real-model test passing.
- Develop locally, then sync to `/ai/projects/Infer-DaseSS` and verify with `/root/.pyenv/shims/python3.11` through `.venv-real`.

---

### Task 1: Add strict float16 checkpoint support

**Files:**
- Modify: `toy_qwen/pretrained.py`
- Modify: `tests/test_pretrained.py`

**Interfaces:**
- Consumes: `_resolve_dtype(dtype: str | torch.dtype) -> torch.dtype`.
- Produces: support for `"float16"` and `torch.float16` in `load_pretrained_qwen` without changing strict checkpoint validation.

- [ ] **Step 1: Write failing float16 tests**

Add focused tests:

```python
from toy_qwen.pretrained import _resolve_dtype

def test_float16_dtype_is_supported(self):
    self.assertIs(_resolve_dtype("float16"), torch.float16)
    self.assertIs(_resolve_dtype(torch.float16), torch.float16)

def test_unsupported_dtype_message_lists_all_supported_values(self):
    with self.assertRaisesRegex(ValueError, "float16.*bfloat16.*float32"):
        _resolve_dtype("float64")
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest tests.test_pretrained.CheckpointValidationTest.test_float16_dtype_is_supported \
  tests.test_pretrained.CheckpointValidationTest.test_unsupported_dtype_message_lists_all_supported_values -v
```

Expected: float16 is rejected and/or the error omits float16.

- [ ] **Step 3: Implement minimal dtype support**

Change the resolver to:

```python
supported = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
message = "dtype must be float16, bfloat16, or float32"
```

Use the same message for unsupported string and `torch.dtype` values.

- [ ] **Step 4: Verify focused and full tests**

Run:

```bash
python3 -m unittest tests.test_pretrained -v
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add toy_qwen/pretrained.py tests/test_pretrained.py
git commit -m "feat: support float16 checkpoint loading"
```

---

### Task 2: Implement the correctness-first StudentEngine adapter

**Files:**
- Modify: `student_release/student_engine.py`
- Create: `tests/test_student_engine.py`

**Interfaces:**
- Consumes: `QwenTokenizerAdapter.from_model_dir`, `load_pretrained_qwen`, and `greedy_generate`.
- Produces: benchmark-compatible `StudentEngine.__init__` and `StudentEngine.generate`.

- [ ] **Step 1: Write failing adapter tests**

Use `unittest.mock.patch` only at the external construction boundaries so the tests run without a 1 GB checkpoint:

```python
class StudentEngineTest(unittest.TestCase):
    @patch("student_release.student_engine.greedy_generate")
    @patch("student_release.student_engine.load_pretrained_qwen")
    @patch("student_release.student_engine.QwenTokenizerAdapter.from_model_dir")
    def test_generate_preserves_order_and_returns_only_continuations(
        self, tokenizer_factory, load_model, generate
    ):
        tokenizer = tokenizer_factory.return_value
        tokenizer.encode_chat.side_effect = [
            ("chat-a", [10, 11]),
            ("chat-b", [20, 21, 22]),
        ]
        tokenizer.decode.side_effect = ["answer-a", "answer-b"]
        load_model.return_value = (MagicMock(), MagicMock())
        generate.side_effect = [
            SimpleNamespace(generated_ids=(101, 102)),
            SimpleNamespace(generated_ids=(201, 202)),
        ]
        engine = StudentEngine("/model", device="cpu", dtype="float16", seed=7)

        outputs = engine.generate(["prompt-a", "prompt-b"], 2, batch_size=2, suite_name=None)

        self.assertEqual(outputs, ["answer-a", "answer-b"])
        self.assertEqual(generate.call_count, 2)
        for call in generate.call_args_list:
            self.assertIsNone(call.kwargs["eos_token_id"])
            self.assertEqual(call.kwargs["max_new_tokens"], 2)
        tokenizer.decode.assert_has_calls([
            call((101, 102), skip_special_tokens=True),
            call((201, 202), skip_special_tokens=True),
        ])
```

Add separate tests asserting:

- model loader receives the exact model path, device, and dtype;
- input tensors have shape `[1, T]`, dtype `torch.long`, and requested device;
- an empty prompt list fails with `ValueError("prompts")`;
- a non-string prompt fails with `TypeError("prompt")`;
- `max_new_tokens <= 0` fails with `ValueError("max_new_tokens")`;
- `batch_size <= 0` fails with `ValueError("batch_size")`;
- constructor exposes optional `seed` and retains `attn_implementation` for inspection.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest tests.test_student_engine -v
```

Expected: failures because the current constructor raises `NotImplementedError`.

- [ ] **Step 3: Implement repository-root imports and initialization**

Replace the skeleton with a thin adapter. Before importing `toy_qwen`, add:

```python
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
```

Initialization must be equivalent to:

```python
self.model_path = model_path
self.device = torch.device(device)
self.dtype = dtype
self.attn_implementation = attn_implementation
self.local_files_only = bool(local_files_only)
self.seed = int(seed)
torch.manual_seed(self.seed)
if self.device.type == "cuda":
    torch.cuda.manual_seed_all(self.seed)
self.tokenizer = QwenTokenizerAdapter.from_model_dir(model_path)
self.model, self.checkpoint_report = load_pretrained_qwen(
    model_path, device=self.device, dtype=dtype
)
```

Do not import Transformers in this file.

- [ ] **Step 4: Implement sequential fixed-step generate**

Validate arguments, delete `suite_name` without reading it, and for each prompt execute:

```python
_, token_ids = self.tokenizer.encode_chat([{"role": "user", "content": prompt}])
input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
result = greedy_generate(
    self.model,
    input_ids,
    eos_token_id=None,
    max_new_tokens=max_new_tokens,
    top_k=5,
)
outputs.append(self.tokenizer.decode(result.generated_ids, skip_special_tokens=True))
```

Return `outputs` in input order. Do not use `batch_size` to select a different computation path in Phase 1.

- [ ] **Step 5: Verify focused tests and static validation**

Run:

```bash
python3 -m unittest tests.test_student_engine -v
python3 student_release/scripts/validate_engine.py --skip-load
```

Expected: adapter tests pass and validator prints `Static strict-rule and signature checks passed.`

- [ ] **Step 6: Run full local regression and commit**

```bash
python3 -m unittest discover -s tests -v
git diff --check
git add student_release/student_engine.py tests/test_student_engine.py
git commit -m "feat: connect benchmark to toy qwen engine"
```

Expected: all tests pass and only `references/` plus `student_release.zip` remain unrelated/untracked.

---

### Task 3: Document and verify the server benchmark path

**Files:**
- Modify: `README.md`
- Modify: `student_release/README.md`

**Interfaces:**
- Consumes: the Phase 1 `StudentEngine` and existing `.venv-real` environment.
- Produces: reproducible runtime-validation and smoke commands plus captured verification evidence.

- [ ] **Step 1: Add exact Phase 1 commands to documentation**

Document that the adapter depends on the parent repository, Phase 1 is sequential/eager, and commands must run from `student_release`:

```bash
cd /ai/projects/Infer-DaseSS/student_release
source use_data_cache.sh

../.venv-real/bin/python scripts/validate_engine.py \
  --model /ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype float16 --local-files-only
```

Include the one-case smoke command from the design, with `--allow-stale-baseline`, batch-size-one flags, cache stress 32, and `results/smoke_test` output.

- [ ] **Step 2: Run local documentation and regression checks**

```bash
python3 student_release/scripts/validate_engine.py --skip-load
python3 -m unittest discover -s tests -v
git diff --check
```

Expected: static validation and every local test pass.

- [ ] **Step 3: Sync changed files to the server**

Use SCP with the already unlocked SSH agent, placing source and tests in their corresponding directories under `/ai/projects/Infer-DaseSS`. Do not sync `.git`, `references/`, `student_release.zip`, local virtual environments, or caches.

- [ ] **Step 4: Run server runtime validation**

From `/ai/projects/Infer-DaseSS/student_release`, run the documented runtime validator with CUDA FP16.

Expected:

```text
Signature check passed.
Runtime interface check passed.
```

- [ ] **Step 5: Run all-suite one-case smoke**

Run the documented smoke command with `--limit 1`, all fixed batch sizes set to 1, cache stress set to 32, process isolation, and a 1800-second worker timeout.

Expected:

- all six suites create result rows;
- no `NotImplementedError`, import failure, CUDA OOM, or empty-output interface failure;
- final summary files exist under `results/smoke_test`;
- any low performance score is recorded as a Phase 1 limitation, not treated as a correctness failure.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md student_release/README.md
git commit -m "docs: add benchmark phase one workflow"
```

---

### Task 4: Final review and handoff to Phase 2

**Files:**
- Review only: changes since `96cba6b`

**Interfaces:**
- Consumes: Phase 1 implementation and server evidence.
- Produces: a reviewed Phase 1 completion state and an explicit Phase 2 starting point.

- [ ] **Step 1: Request code review**

Review checkpoint strictness, static-rule compliance, prompt/continuation semantics, fixed-step behavior, and server evidence. Fix every Critical or Important finding with a new failing regression test before proceeding.

- [ ] **Step 2: Run final verification**

```bash
python3 -m unittest discover -s tests -v
python3 student_release/scripts/validate_engine.py --skip-load
python3 whiteboard_llm_inference.py --prompt 中国首都是 --trace-shapes
rg -n "AutoModel|AutoModelForCausalLM|\.generate\(|\.forward\(" student_release/student_engine.py toy_qwen
git diff --check
```

Expected: all tests and static validation pass, whiteboard output is `北`, forbidden-model scan has no matches, and diff check is clean.

- [ ] **Step 3: Record Phase 2 boundary**

The handoff must state that Phase 2 begins by adding an operational eager/SDPA selector and left-padded batched prefill/decode in `toy_qwen`, without changing `StudentEngine`'s public API.
