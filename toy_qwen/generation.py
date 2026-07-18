from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache import PastKeyValues
from .modeling import QwenToyForCausalLM


LayerCacheShapes = tuple[tuple[int, ...], tuple[int, ...]]
CacheShapes = tuple[LayerCacheShapes, ...]


@dataclass(frozen=True)
class PaddedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    lengths: torch.Tensor


@dataclass(frozen=True)
class GenerationStep:
    index: int
    token_id: int
    selected_logit: float
    top_ids: tuple[int, ...]
    top_logits: tuple[float, ...]


@dataclass(frozen=True)
class GenerationResult:
    generated_ids: tuple[int, ...]
    steps: tuple[GenerationStep, ...]
    prefill_logits_shape: tuple[int, ...]
    first_cache_shapes: CacheShapes
    last_cache_shapes: CacheShapes


@dataclass(frozen=True)
class BatchedGenerationResult:
    generated_ids: tuple[tuple[int, ...], ...]
    prefill_logits_shape: tuple[int, ...]
    first_cache_shapes: CacheShapes
    last_cache_shapes: CacheShapes


def _cache_shapes(cache: PastKeyValues) -> CacheShapes:
    return tuple((tuple(key.shape), tuple(value.shape)) for key, value in cache)


def left_pad_token_ids(
    sequences: list[list[int]],
    pad_token_id: int | None,
    device: str | torch.device,
) -> PaddedBatch:
    if not sequences:
        raise ValueError("at least one token sequence is required")
    if pad_token_id is None or type(pad_token_id) is not int:
        raise ValueError("pad_token_id must be an integer")
    if any(not sequence for sequence in sequences):
        raise ValueError("token sequences must not contain an empty row")
    if any(type(token_id) is not int for sequence in sequences for token_id in sequence):
        raise ValueError("every token id must be an integer")

    target_device = torch.device(device)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long, device=target_device)
    width = int(lengths.max().item())
    input_ids = torch.full(
        (len(sequences), width), pad_token_id, dtype=torch.long, device=target_device
    )
    attention_mask = torch.zeros_like(input_ids)
    position_ids = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, -length:] = torch.tensor(sequence, dtype=torch.long, device=target_device)
        attention_mask[row, -length:] = 1
        position_ids[row, -length:] = torch.arange(length, device=target_device)
    return PaddedBatch(input_ids, attention_mask, position_ids, lengths)


def _validate_padded_batch(batch: PaddedBatch) -> None:
    if not isinstance(batch, PaddedBatch):
        raise ValueError("batch must be a PaddedBatch")
    if batch.input_ids.ndim != 2 or batch.input_ids.shape[0] == 0 or batch.input_ids.shape[1] == 0:
        raise ValueError("input_ids must have non-empty rank-2 shape")
    expected_shape = batch.input_ids.shape
    if batch.attention_mask.shape != expected_shape or batch.position_ids.shape != expected_shape:
        raise ValueError("input_ids, attention_mask, and position_ids must have the same shape")
    if batch.lengths.shape != (expected_shape[0],):
        raise ValueError("lengths shape must match the batch dimension")
    tensors = (batch.input_ids, batch.attention_mask, batch.position_ids, batch.lengths)
    if any(tensor.device != batch.input_ids.device for tensor in tensors):
        raise ValueError("all padded batch tensors must be on the same device")
    if any(tensor.dtype != torch.long for tensor in tensors):
        raise ValueError("all padded batch tensors must use torch.long")
    if not torch.all((batch.attention_mask == 0) | (batch.attention_mask == 1)):
        raise ValueError("attention_mask must contain only zero or one")
    mask_lengths = batch.attention_mask.sum(dim=1)
    if torch.any(mask_lengths == 0):
        raise ValueError("padded batch must not contain an empty valid row")
    if torch.any(batch.attention_mask[:, 1:] < batch.attention_mask[:, :-1]):
        raise ValueError("attention_mask must describe left-padded rows")
    if not torch.equal(batch.lengths, mask_lengths):
        raise ValueError("lengths must equal attention_mask row sums")
    expected_positions = (batch.attention_mask.cumsum(dim=1) - 1).clamp_min(0)
    if not torch.equal(batch.position_ids, expected_positions):
        raise ValueError("position_ids must count valid tokens from zero")


def batched_greedy_generate(
    model: QwenToyForCausalLM,
    batch: PaddedBatch,
    max_new_tokens: int,
) -> BatchedGenerationResult:
    _validate_padded_batch(batch)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_width = batch.input_ids.shape[1]
    if prompt_width + max_new_tokens - 1 > model.config.max_position_embeddings:
        raise ValueError("generation requires a forward beyond max_position_embeddings")

    batch_size = batch.input_ids.shape[0]
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    current_ids = batch.input_ids
    current_positions = batch.position_ids
    full_attention_mask = batch.attention_mask
    past_key_values: PastKeyValues | None = None
    prefill_logits_shape: tuple[int, ...] | None = None
    first_cache_shapes: CacheShapes | None = None
    last_cache_shapes: CacheShapes | None = None

    with torch.inference_mode():
        for index in range(max_new_tokens):
            output = model(
                current_ids,
                attention_mask=full_attention_mask,
                position_ids=current_positions,
                past_key_values=past_key_values,
                use_cache=True,
                num_logits_to_keep=1,
            )
            if output.past_key_values is None:
                raise RuntimeError("model did not return a KV cache with use_cache=True")
            if prefill_logits_shape is None:
                prefill_logits_shape = tuple(output.logits.shape)

            next_ids = output.logits[:, -1, :].argmax(dim=-1)
            for row, token_id in enumerate(next_ids.detach().cpu().tolist()):
                generated[row].append(int(token_id))

            past_key_values = output.past_key_values
            last_cache_shapes = _cache_shapes(past_key_values)
            if first_cache_shapes is None:
                first_cache_shapes = last_cache_shapes
            if index + 1 < max_new_tokens:
                current_ids = next_ids[:, None]
                current_positions = (batch.lengths + index)[:, None]
                full_attention_mask = torch.cat(
                    (
                        full_attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            dtype=full_attention_mask.dtype,
                            device=full_attention_mask.device,
                        ),
                    ),
                    dim=1,
                )

    assert prefill_logits_shape is not None
    assert first_cache_shapes is not None
    assert last_cache_shapes is not None
    return BatchedGenerationResult(
        generated_ids=tuple(tuple(row) for row in generated),
        prefill_logits_shape=prefill_logits_shape,
        first_cache_shapes=first_cache_shapes,
        last_cache_shapes=last_cache_shapes,
    )


def greedy_generate(
    model: QwenToyForCausalLM,
    input_ids: torch.Tensor,
    eos_token_id: int | None,
    max_new_tokens: int,
    top_k: int = 5,
) -> GenerationResult:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [1, sequence_length]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    generated_ids: list[int] = []
    steps: list[GenerationStep] = []
    past_key_values: PastKeyValues | None = None
    current_ids = input_ids
    prefill_logits_shape: tuple[int, ...] | None = None
    first_cache_shapes: CacheShapes | None = None
    last_cache_shapes: CacheShapes | None = None

    with torch.inference_mode():
        for index in range(max_new_tokens):
            if input_ids.shape[1] + index > model.config.max_position_embeddings:
                raise ValueError("generation requires a forward beyond max_position_embeddings")
            output = model(current_ids, past_key_values=past_key_values, use_cache=True)
            if output.past_key_values is None:
                raise RuntimeError("model did not return a KV cache with use_cache=True")
            if prefill_logits_shape is None:
                prefill_logits_shape = tuple(output.logits.shape)

            next_logits = output.logits[:, -1, :]
            count = min(top_k, next_logits.shape[-1])
            top_values, top_ids = torch.topk(next_logits, k=count, dim=-1, sorted=True)
            ids_cpu = top_ids[0].detach().cpu()
            values_cpu = top_values[0].float().detach().cpu()
            token_id = int(ids_cpu[0])
            steps.append(GenerationStep(
                index=index,
                token_id=token_id,
                selected_logit=float(values_cpu[0]),
                top_ids=tuple(int(value) for value in ids_cpu.tolist()),
                top_logits=tuple(float(value) for value in values_cpu.tolist()),
            ))
            generated_ids.append(token_id)

            past_key_values = output.past_key_values
            last_cache_shapes = _cache_shapes(past_key_values)
            if first_cache_shapes is None:
                first_cache_shapes = last_cache_shapes
            if eos_token_id is not None and token_id == eos_token_id:
                break
            current_ids = torch.tensor([[token_id]], dtype=input_ids.dtype, device=input_ids.device)

    assert prefill_logits_shape is not None
    assert first_cache_shapes is not None
    assert last_cache_shapes is not None
    return GenerationResult(
        generated_ids=tuple(generated_ids),
        steps=tuple(steps),
        prefill_logits_shape=prefill_logits_shape,
        first_cache_shapes=first_cache_shapes,
        last_cache_shapes=last_cache_shapes,
    )
