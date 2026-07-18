# Batched SDPA Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real SDPA attention backend and left-padded batched prefill/decode so `StudentEngine` completes the full public benchmark with batch sizes 1/2/4.

**Architecture:** Preserve eager attention as the explainable reference and add SDPA as a runtime-selected backend inside the same Qwen2 attention module. Add last-position logit projection, pure left-padding preparation, and a dedicated fixed-step batched generator; then replace `StudentEngine`'s sequential loop with chunked batch calls.

**Tech Stack:** Python 3.11, PyTorch SDPA/GQA, safetensors, tokenizers, unittest; server A800 with PyTorch 2.10.0+cu128; public inference-optimization benchmark.

## Global Constraints

- Keep one handwritten Qwen2 implementation under repository-root `toy_qwen`; do not copy model code into `student_release`.
- Default attention remains `eager`; only explicit `sdpa` selection changes the backend.
- Do not import or call Hugging Face model APIs in production.
- `StudentEngine` must not read or branch on `suite_name` or benchmark content.
- Batch generation remains greedy and fixed-step; ignore EOS for every suite.
- Preserve full-logit behavior by default; optimize projection only when explicitly requested.
- Phase 2 does not implement preallocated/paged KV Cache, prefix reuse, or `serve_requests`.
- Develop locally, sync exact files to `/ai/projects/Infer-DaseSS`, and verify with `.venv-real` plus the server-local Qwen checkpoint.

---

### Task 1: Project only requested LM Head positions

**Files:**
- Modify: `toy_qwen/modeling.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Produces: `QwenToyForCausalLM.forward(input_ids, num_logits_to_keep: int | None = None, **kwargs)`.
- Preserves: default logits shape `[B,T,V]`.

- [ ] **Step 1: Write failing tests**

Add tests using the whiteboard model:

```python
def test_num_logits_to_keep_projects_only_last_positions(self):
    model = build_whiteboard_model().eval()
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    full = model(ids, use_cache=False).logits
    last = model(ids, use_cache=False, num_logits_to_keep=1).logits
    last_two = model(ids, use_cache=False, num_logits_to_keep=2).logits
    self.assertEqual(last.shape, (1, 1, 9))
    self.assertEqual(last_two.shape, (1, 2, 9))
    torch.testing.assert_close(last, full[:, -1:])
    torch.testing.assert_close(last_two, full[:, -2:])

def test_num_logits_to_keep_must_be_positive(self):
    with self.assertRaisesRegex(ValueError, "num_logits_to_keep"):
        build_whiteboard_model()(torch.tensor([[0]]), num_logits_to_keep=0)
```

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_model -v`; expect the argument to leak into `QwenToyModel.forward` and fail.

- [ ] **Step 3: Implement hidden-state slicing before LM Head**

Use an explicit causal-LM argument:

```python
def forward(self, input_ids, num_logits_to_keep=None, **kwargs):
    if num_logits_to_keep is not None and num_logits_to_keep <= 0:
        raise ValueError("num_logits_to_keep must be positive")
    output = self.model(input_ids, **kwargs)
    hidden = output.last_hidden_state
    if num_logits_to_keep is not None:
        hidden = hidden[:, -num_logits_to_keep:, :]
    logits = self.lm_head(hidden)
    return CausalLMOutput(logits, ...)
```

- [ ] **Step 4: Run focused/full tests and commit**

```bash
python3 -m unittest tests.test_model -v
python3 -m unittest discover -s tests -v
git add toy_qwen/modeling.py tests/test_model.py
git commit -m "feat: project selected causal lm logits"
```

---

### Task 2: Add eager/SDPA attention backends

**Files:**
- Modify: `toy_qwen/modeling.py`
- Modify: `toy_qwen/pretrained.py`
- Modify: `tests/test_attention.py`
- Modify: `tests/test_pretrained.py`

**Interfaces:**
- Produces: `QwenToyForCausalLM.set_attention_implementation(name: str)`.
- Extends: `load_pretrained_qwen(..., attn_implementation: str = "eager")`.

- [ ] **Step 1: Write failing backend tests**

Cover:

```python
def test_rejects_unknown_attention_backend(self):
    with self.assertRaisesRegex(ValueError, "eager.*sdpa"):
        build_whiteboard_model().set_attention_implementation("flash_magic")

def test_sdpa_matches_eager(self):
    eager = build_whiteboard_model().eval()
    sdpa = copy.deepcopy(eager).set_attention_implementation("sdpa")
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    with torch.no_grad():
        expected = eager(ids, use_cache=False).logits
        actual = sdpa(ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
```

Add a 14Q/2KV comparison, future-token causal test under SDPA, and a two-row left-padding test asserting finite outputs and equality of each row's valid final token with independent unpadded eager execution.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_attention -v`; expect missing selector/backend failures.

- [ ] **Step 3: Build one allowed-attention mask helper**

Create a boolean mask from physical causal positions and key validity:

```python
query_positions = past_length + torch.arange(query_length, device=device)
key_positions = torch.arange(key_length, device=device)
allowed = key_positions[None, :] <= query_positions[:, None]
allowed = allowed.view(1, 1, query_length, key_length)
if attention_mask is not None:
    if attention_mask.shape != (batch_size, key_length):
        raise ValueError("attention_mask must match batch and full key length")
    allowed = allowed & attention_mask[:, None, None, :].bool()
```

Use the same allowed mask in eager and SDPA. In eager, call `masked_fill(~allowed, torch.finfo(dtype).min)` once instead of adding multiple minimum values. Zero attention output rows whose current query mask is padding so padded states remain finite.

- [ ] **Step 4: Implement SDPA without score allocation**

For `sdpa`, call:

```python
output = F.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=allowed,
    dropout_p=0.0,
    is_causal=False,
    enable_gqa=True,
)
```

If `enable_gqa` raises a runtime capability error, repeat K/V and call SDPA without GQA. Do not fall back to eager. Record logical `attention_scores=(B,Nq,T,S)` in trace.

- [ ] **Step 5: Add and apply the runtime selector**

Initialize every attention layer as eager. `set_attention_implementation` validates the name, assigns it to every layer, and returns `self` for tests. Extend the real loader with a defaulted backend argument and call the selector before returning.

Add a loader/unit test showing `attn_implementation="sdpa"` reaches the model selector and an invalid name fails before inference.

- [ ] **Step 6: Verify and commit**

```bash
python3 -m unittest tests.test_attention tests.test_pretrained -v
python3 -m unittest discover -s tests -v
git add toy_qwen/modeling.py toy_qwen/pretrained.py tests/test_attention.py tests/test_pretrained.py
git commit -m "feat: add sdpa attention backend"
```

---

### Task 3: Add left-padding preparation and batched generation

**Files:**
- Modify: `toy_qwen/generation.py`
- Create: `tests/test_batch_generation.py`

**Interfaces:**
- Produces: `PaddedBatch`, `BatchedGenerationResult`, `left_pad_token_ids`, and `batched_greedy_generate`.

- [ ] **Step 1: Write failing padding tests**

Specify the API:

```python
batch = left_pad_token_ids([[1, 2], [3, 4, 5]], pad_token_id=0, device="cpu")
self.assertEqual(batch.input_ids.tolist(), [[0, 1, 2], [3, 4, 5]])
self.assertEqual(batch.attention_mask.tolist(), [[0, 1, 1], [1, 1, 1]])
self.assertEqual(batch.position_ids.tolist(), [[0, 0, 1], [0, 1, 2]])
self.assertEqual(batch.lengths.tolist(), [2, 3])
```

Reject no sequences, empty rows, missing pad ID, and non-integer token IDs.

- [ ] **Step 2: Verify padding RED, implement, and verify GREEN**

Run the focused test, implement a frozen `PaddedBatch` dataclass and right-aligned tensor fill, then rerun it.

- [ ] **Step 3: Write failing batched generation tests**

Using a whiteboard model configured with SDPA, assert:

- two identical prompts generate the same IDs as batch-one generation;
- variable-length left-padded rows match independent unpadded runs;
- generated result has one tuple per batch row and exactly the requested length;
- prefill calls LM Head with sequence length one (capture LM Head input using a forward pre-hook);
- cached second-token IDs/logits agree with an uncached full forward using explicit per-row positions;
- malformed shapes, invalid masks, right-padded final columns, empty valid rows, and context overflow fail clearly.

- [ ] **Step 4: Verify generation RED**

Run `python3 -m unittest tests.test_batch_generation -v`; expect missing API failures.

- [ ] **Step 5: Implement fixed-step batch decode**

Create:

```python
@dataclass(frozen=True)
class BatchedGenerationResult:
    generated_ids: tuple[tuple[int, ...], ...]
    prefill_logits_shape: tuple[int, ...]
    first_cache_shapes: CacheShapes
    last_cache_shapes: CacheShapes
```

Prefill with full `input_ids`, mask, positions, cache enabled, and `num_logits_to_keep=1`. At each step use `argmax`, append one ID per row, extend the full key mask, and feed `[B,1]` IDs with positions `lengths + step`. Do not compute top-k and do not stop on EOS.

- [ ] **Step 6: Verify and commit**

```bash
python3 -m unittest tests.test_batch_generation -v
python3 -m unittest discover -s tests -v
git add toy_qwen/generation.py tests/test_batch_generation.py
git commit -m "feat: add left-padded batch generation"
```

---

### Task 4: Switch StudentEngine to true batching

**Files:**
- Modify: `student_release/student_engine.py`
- Modify: `tests/test_student_engine.py`

**Interfaces:**
- Consumes: `left_pad_token_ids` and `batched_greedy_generate`.
- Produces: unchanged benchmark `StudentEngine` API with real chunked batch execution.

- [ ] **Step 1: Rewrite mock expectations as failing batch tests**

For five prompts and `batch_size=2`, make tokenization return variable-length IDs and assert:

- `left_pad_token_ids` receives chunks of sizes 2,2,1;
- `batched_greedy_generate` is called exactly three times;
- loader receives `attn_implementation="sdpa"`;
- decoded continuations remain in original order;
- batch size 1 remains valid;
- no call to the batch-one `greedy_generate` remains.

- [ ] **Step 2: Verify RED**

Run `python3 -m unittest tests.test_student_engine -v`; expect sequential-call and loader-signature mismatches.

- [ ] **Step 3: Implement chunked batching**

Import the new batch functions. Pass the backend to `load_pretrained_qwen`. Encode prompts in input order, slice encoded lists in `range(0, len(prompts), batch_size)`, left-pad each chunk with `tokenizer.pad_token_id`, call the batch generator once, and decode each row's generated IDs.

Fail clearly if `pad_token_id is None`. Continue to delete `suite_name` without reading it.

- [ ] **Step 4: Verify static/full tests and commit**

```bash
python3 -m unittest tests.test_student_engine -v
python3 student_release/scripts/validate_engine.py --skip-load
python3 -m unittest discover -s tests -v
git diff --check
git add student_release/student_engine.py tests/test_student_engine.py
git commit -m "feat: batch benchmark inference requests"
```

---

### Task 5: Real-model alignment and smoke validation

**Files:**
- Modify: `verification/compare_transformers.py`
- Modify: `verification/README.md`
- Modify: `README.md`

**Interfaces:**
- Produces: eager/SDPA real-model comparison evidence and updated benchmark commands/status.

- [ ] **Step 1: Extend verification CLI**

Add `--attn-implementation {eager,sdpa}` and pass it only to the custom loader. Preserve Transformers eager oracle. Print the selected custom backend.

- [ ] **Step 2: Run local validation**

```bash
python3 -m unittest discover -s tests -v
python3 student_release/scripts/validate_engine.py --skip-load
python3 verification/compare_transformers.py --help
git diff --check
```

- [ ] **Step 3: Sync exact source/tests/docs to server**

Do not sync `.git`, `references/`, either ZIP, virtual environments, caches, or results.

- [ ] **Step 4: Run server correctness gates**

Run:

1. full server unittest discovery;
2. real FP16 runtime validator with SDPA;
3. batch-size 1/2/4 targeted generation on three unequal prompts;
4. real float32 custom-SDPA versus Transformers oracle;
5. six-suite one-case smoke.

Expected: tests and validator pass, all batch outputs are non-empty and ordered, no OOM, oracle top-10/greedy match with reviewed absolute tolerance, and smoke runtime/valid rates equal 1.0.

- [ ] **Step 5: Commit verification/docs**

```bash
git add verification/compare_transformers.py verification/README.md README.md
git commit -m "test: verify batched sdpa real inference"
```

---

### Task 6: Run the complete public benchmark

**Files:**
- Server output only: `student_release/results/phase2_full/`
- Modify after results: `README.md`

**Interfaces:**
- Produces: full Phase 2 performance baseline and documented metrics.

- [ ] **Step 1: Run exact public default configuration**

From the server `student_release` directory, source `use_data_cache.sh` and run without `--limit`, using local FP16 model, SDPA, baseline summary, timed repeats 3, process isolation, and worker timeout 1800. Do not override the public default batch sizes or token budgets.

- [ ] **Step 2: Validate result completeness**

Read `final_summary.json`, `final_summary.txt`, student summary, and CSV. Confirm six suites, requested batch groups, runtime success >=0.98, valid-output rate is no lower than the bundled public baseline, long partial >=0.90, no OOM/timeouts, and decode batch-4 TPS > batch-1 TPS. The valid-output field includes content scoring, so it must not be treated as a pure runtime-health threshold.

If any gate fails, use systematic debugging and do not relax the gate without evidence.

- [ ] **Step 3: Document measured metrics**

Add a Phase 2 results table to root README containing GPU, PyTorch, dtype, backend, score, runtime/valid, long partial, decode TPS for batch 1/2/4, TTFT, serving TPS, cache stress peak, and result directory. State that the public baseline uses different hardware.

- [ ] **Step 4: Commit results documentation**

```bash
git add README.md
git commit -m "docs: record phase two benchmark results"
```

---

### Task 7: Review and final verification

**Files:**
- Review: all changes since `1868d9f`

- [ ] **Step 1: Request code review**

Review mask semantics, GQA, padding finiteness, logical/physical positions, cache shape, last-logit projection, StudentEngine order, static rules, and benchmark evidence. Fix Critical/Important findings through RED/GREEN tests.

- [ ] **Step 2: Run final local verification**

```bash
python3 -m unittest discover -s tests -v
python3 student_release/scripts/validate_engine.py --skip-load
python3 whiteboard_llm_inference.py --prompt 中国首都是 --trace-shapes
rg -n "AutoModel|AutoModelForCausalLM|\.generate\(|\.forward\(" student_release/student_engine.py toy_qwen
git diff --check
```

- [ ] **Step 3: Run final server verification**

Repeat real runtime validation, SDPA oracle, and check the retained full benchmark summary. Mark Phase 2 complete only if every gate remains satisfied.
