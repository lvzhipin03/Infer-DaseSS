from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache import PastKeyValues
from .modeling import QwenToyForCausalLM


LayerCacheShapes = tuple[tuple[int, ...], tuple[int, ...]]
CacheShapes = tuple[LayerCacheShapes, ...]


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


def _cache_shapes(cache: PastKeyValues) -> CacheShapes:
    return tuple((tuple(key.shape), tuple(value.shape)) for key, value in cache)


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
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus generated tokens exceeds max_position_embeddings")

    generated_ids: list[int] = []
    steps: list[GenerationStep] = []
    past_key_values: PastKeyValues | None = None
    current_ids = input_ids
    prefill_logits_shape: tuple[int, ...] | None = None
    first_cache_shapes: CacheShapes | None = None
    last_cache_shapes: CacheShapes | None = None

    with torch.inference_mode():
        for index in range(max_new_tokens):
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
