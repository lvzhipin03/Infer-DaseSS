# Benchmark Phase 1 Adapter Design

## Goal

Connect the existing handwritten `toy_qwen` implementation to the inference-optimization benchmark through `student_release.StudentEngine`, prioritizing correctness and interface compliance before performance optimization.

Phase 1 must run the benchmark's runtime validation and one-case smoke suites with the real server-local Qwen2.5-0.5B-Instruct checkpoint. It does not need to achieve baseline throughput.

## Architectural Decision

`student_release` is not a standalone copy of the model implementation. Its `student_engine.py` is a thin benchmark adapter that imports and reuses the repository-root `toy_qwen` package.

```text
run_inference_benchmark.py
        |
        v
student_release/student_engine.py
        |
        +-- QwenTokenizerAdapter
        +-- load_pretrained_qwen
        +-- greedy_generate
        |
        v
toy_qwen handwritten Qwen2 forward
```

This keeps one source of truth for Embedding, RMSNorm, Qwen2 RoPE, GQA, SwiGLU, residual connections, Final RMSNorm, LM Head, checkpoint validation, and KV Cache behavior.

The adapter resolves the repository root relative to `student_engine.py` and adds it to `sys.path`. This is intentional: the benchmark is run from the checked-out repository and Phase 1 is not designed as a standalone `student_release` archive.

## StudentEngine Interface

`StudentEngine.__init__` retains the benchmark-compatible signature:

```python
StudentEngine(
    model_path,
    device="cuda",
    dtype="float16",
    attn_implementation="sdpa",
    local_files_only=False,
    seed=0,
)
```

Initialization performs these operations once per benchmark worker:

1. Validate that the requested model path is local.
2. Load `tokenizer.json` and `tokenizer_config.json` with `QwenTokenizerAdapter`.
3. Load config and raw safetensors through `load_pretrained_qwen`.
4. Move the handwritten model to the requested device and dtype and set evaluation mode.
5. Record `attn_implementation` for compatibility. Phase 1 continues to use the existing eager attention implementation even when the benchmark passes `sdpa`; Phase 2 will make this selector operational.
6. Seed PyTorch without introducing sampling. Generation remains greedy.

`local_files_only` is accepted for interface compatibility. The existing loader never downloads files, so all Phase 1 loading is local regardless of the flag value.

## Generate Data Flow

`generate(prompts, max_new_tokens, batch_size, suite_name)` accepts a list of prompts but processes each prompt sequentially:

```text
prompt
  -> default Qwen system + user Chat Template
  -> tokenizer IDs
  -> tensor [1, T]
  -> full-prompt prefill
  -> single-token cached greedy decode
  -> exactly max_new_tokens selections
  -> decode generated IDs only
  -> continuation string
```

Requirements:

- Return exactly one string for every input prompt.
- Preserve input order.
- Return continuation text only; never prepend the source prompt.
- Use the same inference path for all suites.
- Do not read or branch on `suite_name`.
- Accept `batch_size` but do not use it to change Phase 1 computation.
- Pass `eos_token_id=None` to the generator so generation uses the fixed token budget required by performance suites.
- Decode with special tokens skipped so generated `<|im_end|>` tokens do not appear in returned text.
- Reject an empty prompt list, non-string prompts, and non-positive `max_new_tokens` with clear errors.

Fixed-step decoding is a global engine policy, not benchmark-case detection. It avoids suite-specific branching and provides enough generated tokens for throughput and cache-stress scoring.

## Required Changes to toy_qwen

### Float16 loading

Extend the strict dtype resolver to accept `float16` / `torch.float16`. Preserve the existing `float32` and `bfloat16` paths and all checkpoint validation rules.

### No model math changes

Phase 1 does not change attention, RoPE, MLP, cache layout, or generation math. The already-verified BF16 and float32 behavior remains the reference implementation.

## Static Validation Constraints

All Python files under the student package are scanned. The adapter must therefore:

- not import or invoke Hugging Face model classes;
- not call `.generate()` or `.forward()` methods;
- not import vLLM or other inference engines;
- not contain benchmark suite names, public case identifiers, required answers, or keyword-specific branches;
- not load benchmark data;
- not read `suite_name`.

Importing the repository's handwritten `toy_qwen` modules is allowed. The verification-only Transformers oracle remains outside `student_release` and outside the production path.

## Error Handling

- Missing local model files fail during initialization with the existing explicit file list.
- Unsupported dtype or unavailable CUDA fails before generation.
- A prompt that exceeds the model context limit fails with the existing model error.
- CUDA OOM is allowed to propagate to the benchmark, which records it consistently.
- The adapter must not silently return empty strings after an internal failure.

## Tests and Acceptance

Development follows TDD.

Local unit tests cover:

1. `float16` dtype resolution and model loading behavior without changing existing dtype behavior.
2. `StudentEngine` signature, initialization wiring, prompt order, continuation-only decoding, fixed-step EOS policy, and argument validation using small controlled collaborators.
3. Static strict-rule validation.
4. The complete existing `toy_qwen` regression suite.

Server acceptance uses the real checkpoint at:

```text
/ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct
```

The gates are:

1. `scripts/validate_engine.py --skip-load` passes.
2. Runtime `validate_engine.py` passes with CUDA FP16 and two prompts.
3. One-case, batch-size-one smoke runs every public suite without exception.
4. Every smoke output is non-empty and continuation-only.
5. Existing whiteboard inference still predicts `北`.

Phase 1 makes no throughput or latency acceptance claim.

## Explicitly Deferred Work

Phase 2:

- operational eager/SDPA attention selector;
- left-padded batched prefill;
- batched cached decode;
- per-request attention masks and position IDs.

Phase 3:

- preallocated KV Cache;
- elimination of per-token `torch.cat` cache growth;
- cache memory and long-decode optimization.

Phase 4:

- `serve_requests`;
- length grouping and active-batch scheduling;
- prefix reuse and optional block/paged cache management.

These deferred features must extend `toy_qwen`; they must not create a second model implementation under `student_release`.
