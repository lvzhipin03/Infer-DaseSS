# Batched SDPA Inference Design

## Goal

Turn the correctness-first benchmark adapter into a real batched inference path. Phase 2 must make `attn_implementation=sdpa` operational, run left-padded batched prefill and cached decode for batch sizes 1/2/4, and complete the full public benchmark without changing the handwritten Qwen2 architecture or `StudentEngine` public API.

## Chosen Approach

Extend the repository-root `toy_qwen` implementation with a second attention backend and a dedicated batched generator. Keep eager attention as the explainable reference path and use PyTorch scaled dot-product attention as the benchmark path.

Do not copy model code into `student_release`. `student_release/student_engine.py` remains a thin adapter over `toy_qwen`.

```text
prompts
  -> Qwen chat template and tokenization
  -> chunks of batch_size
  -> left padding + masks + per-row positions
  -> batched prefill
  -> last-position LM Head only
  -> batched fixed-step cached decode
  -> per-row continuation decode
  -> outputs in original order
```

Varlen FlashAttention, PagedAttention, continuous batching, prefix reuse, and preallocated KV storage are explicitly deferred.

## Runtime Attention Backend

Every `QwenToyAttention` has a runtime `attn_implementation` value:

- `eager`: preserve the existing QK matmul, additive causal mask, float32 softmax, repeated KV heads, attention output, and trace semantics.
- `sdpa`: call `torch.nn.functional.scaled_dot_product_attention` with grouped-query attention enabled when supported by the installed PyTorch version.

The model exposes one public runtime selector that validates `"eager"` or `"sdpa"` and applies it to every decoder layer. `load_pretrained_qwen` accepts an optional attention implementation and configures the model before returning it. The default remains eager so existing whiteboard, CLI, and oracle behavior do not change implicitly.

The SDPA path must not materialize `[B,Nq,T,S]` attention scores. Shape tracing records the logical score shape without allocating it.

### SDPA mask

Use a boolean allowed-attention mask with shape broadcastable to `[B,Nq,T,S]`:

```text
allowed = causal_allowed AND valid_key
```

`valid_key` comes from the full key-side `attention_mask`. Query rows corresponding to left padding are not consumed by generation; fully masked padded query rows must produce finite outputs rather than NaNs.

During single-token cached decode, the query is at the newest physical cache location, so all preceding non-padding keys are causally visible. Per-row RoPE positions remain logical token positions and are independent of physical left padding.

If the runtime cannot use SDPA GQA directly, the implementation may repeat K/V heads before SDPA as a correctness fallback. This fallback must be visible in tests and must not change model output semantics.

## Left-Padded Batch Representation

Tokenize every prompt with the same default Qwen system/user chat template. Within each chunk, left-pad to the longest token length using `pad_token_id`.

Example:

```text
input_ids:
[PAD, PAD, a, b]
[c,   d,   e, f]

attention_mask:
[0, 0, 1, 1]
[1, 1, 1, 1]

position_ids:
[0, 0, 0, 1]
[0, 1, 2, 3]
```

Position IDs are computed from the mask:

```python
position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 0)
```

The final prompt token is at column `-1` for every row, so batched prefill selects the same last hidden-state column.

For decode step `i`, each row receives logical position:

```text
original_valid_length[row] + i
```

The physical KV cache length is shared across the batch and includes left-padding slots. The full key attention mask is extended by one valid column after each selected token.

## Last-Position Logit Projection

The base decoder still computes hidden states for all prompt positions because attention requires them. The LM Head must not project every prefill position into the 151,936-token vocabulary.

Add an optional causal-LM forward argument:

```python
num_logits_to_keep: int | None = None
```

- `None`: preserve current behavior and return logits for the full sequence.
- positive integer: slice the final hidden state to the last N positions before LM Head projection.
- zero or negative: raise `ValueError`.

The batched generator uses `num_logits_to_keep=1` for prefill and decode. Existing inference, trace tests, and Transformers oracle continue to receive full logits unless they explicitly opt in.

## Batched Generator

Add a separate API rather than changing the logging-oriented batch-one generator:

```python
batched_greedy_generate(
    model,
    input_ids,
    attention_mask,
    position_ids,
    max_new_tokens,
) -> BatchedGenerationResult
```

`BatchedGenerationResult.generated_ids` is a tuple of token-ID tuples, one per input row.

Validation:

- `input_ids`, `attention_mask`, and `position_ids` are rank-2 with identical shape;
- batch and sequence dimensions are non-empty;
- masks contain only zero/one values;
- every row contains at least one valid prompt token and ends in a valid token;
- `max_new_tokens` is positive;
- requested forward positions do not exceed the model context limit.

Algorithm:

1. Run one full batched prefill with cache enabled and last-position logits only.
2. Select one argmax ID per row.
3. Append IDs to per-row output lists.
4. Extend the key attention mask with one valid column.
5. Build one-token per-row position IDs from original valid lengths plus step index.
6. Feed `[B,1]` selected IDs with the shared batched cache.
7. Repeat for exactly `max_new_tokens` selections.

Phase 2 ignores EOS globally and never removes finished rows, keeping decode shapes stable and preserving the benchmark's fixed-step policy.

The benchmark path uses argmax directly and does not compute top-k logging on every step.

## StudentEngine Integration

`StudentEngine.__init__` passes `attn_implementation` to the handwritten model loader instead of only recording it.

`StudentEngine.generate`:

1. validates the existing public arguments;
2. divides prompts into chunks no larger than `batch_size`;
3. encodes each chunk with the existing Qwen chat template;
4. builds left-padded input IDs, masks, and positions;
5. calls `batched_greedy_generate` once per chunk;
6. decodes generated IDs only, with special tokens skipped;
7. appends results in original prompt order.

No code reads or branches on `suite_name`. Batch size changes tensor grouping only, not inference semantics.

## Correctness Tests

Development follows TDD. Required local coverage:

1. Eager versus SDPA outputs for the whiteboard model.
2. Eager versus SDPA outputs for 14 query heads / 2 KV heads.
3. Causal masking under SDPA.
4. Key-padding masking and finite padded query rows.
5. Full-logit default and last-N logit projection.
6. Batched prefill/decode versus independent batch-one generation.
7. Variable prompt lengths with left padding.
8. Per-row logical decode position IDs.
9. Cached batched decode versus an uncached full forward.
10. StudentEngine chunking for batch sizes 1/2/4, output order, and continuation-only decoding.
11. Existing whiteboard, real loader, CLI, and static validator regression.

Server correctness gates:

- real FP16 runtime validator passes;
- eager and SDPA first-token IDs agree for the fixed real-model prompt;
- verification-only Transformers comparison retains matching top-10 and greedy output;
- no forbidden model API enters `student_release` or `toy_qwen`.

## Full Public Benchmark Acceptance

Run from `/ai/projects/Infer-DaseSS/student_release` with the real server model, FP16, SDPA, local files only, process isolation, and the public baseline's default configuration:

```text
decode batch sizes: 1,2,4
TTFT batch size: 1
serving fallback batch size: 4
mixed batch sizes: 1,2
cache-stress batch sizes: 2,4
cache-stress decode lengths: 128,256,512
timed repeats: 3
seed: 0
```

Acceptance conditions:

- all six suites complete without CUDA OOM or worker timeout;
- runtime success and valid-output rates are at least 0.98;
- long-context partial score is at least 0.90 and does not regress materially from Phase 1;
- result summaries contain the requested batch-size groups;
- decode batch-4 TPS is greater than decode batch-1 TPS;
- `final_summary.json`, `final_summary.txt`, and student CSV/summary files exist;
- the complete summary is retained as the performance baseline for Phase 3.

The public score is recorded but is not itself a correctness gate because the included baseline was generated on different hardware.

## Error Handling

- Reject unknown attention backend names during model setup.
- Reject malformed batch tensors before model execution.
- Fail clearly when `pad_token_id` is unavailable.
- Propagate CUDA OOM to the benchmark so it can record the failure.
- Do not silently fall back from requested SDPA to eager. A GQA implementation fallback within SDPA is allowed and tested.
- Preserve existing strict checkpoint failures before partial loading.

## Deferred Work

Phase 3 will replace per-step KV `torch.cat` with preallocated cache storage and measure cache-stress memory/TPS improvements.

Phase 4 will implement `serve_requests`, length-aware scheduling, active batches, prefix reuse, and optional paged/block cache management.
