from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from .config import QwenToyConfig

PastKeyValue: TypeAlias = tuple[torch.Tensor, torch.Tensor]
PastKeyValues: TypeAlias = tuple[PastKeyValue, ...]


@dataclass
class StaticKVCache:
    key_cache: tuple[torch.Tensor, ...]
    value_cache: tuple[torch.Tensor, ...]
    length: int
    capacity: int

    @classmethod
    def allocate(
        cls,
        config: QwenToyConfig,
        batch_size: int,
        capacity: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> StaticKVCache:
        if batch_size <= 0:
            raise ValueError("cache batch size must be positive")
        if capacity <= 0 or capacity > config.max_position_embeddings:
            raise ValueError("cache capacity must be within max_position_embeddings")
        shape = (
            batch_size,
            config.num_key_value_heads,
            capacity,
            config.head_dim,
        )
        keys = tuple(
            torch.empty(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        )
        values = tuple(
            torch.empty(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        )
        return cls(keys, values, length=0, capacity=capacity)

    def layer(self, layer_idx: int, length: int | None = None) -> PastKeyValue:
        logical_length = self.length if length is None else length
        if not 0 <= layer_idx < len(self.key_cache):
            raise ValueError("cache layer index is out of range")
        if not 0 <= logical_length <= self.capacity:
            raise ValueError("cache view length is out of range")
        return (
            self.key_cache[layer_idx][:, :, :logical_length, :],
            self.value_cache[layer_idx][:, :, :logical_length, :],
        )

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> PastKeyValue:
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("new cache K/V shapes must match and have rank 4")
        end = self.length + key.shape[2]
        if end > self.capacity:
            raise ValueError("KV cache capacity exceeded")
        key_buffer = self.key_cache[layer_idx]
        value_buffer = self.value_cache[layer_idx]
        expected = (
            key_buffer.shape[0],
            key_buffer.shape[1],
            key.shape[2],
            key_buffer.shape[3],
        )
        if key.shape != expected:
            raise ValueError("new cache K/V shape does not match the allocated cache")
        if key.device != key_buffer.device or key.dtype != key_buffer.dtype:
            raise ValueError("new cache K/V device or dtype does not match the allocated cache")
        key_buffer[:, :, self.length:end, :].copy_(key)
        value_buffer[:, :, self.length:end, :].copy_(value)
        return self.layer(layer_idx, end)

    def advance(self, token_count: int) -> None:
        if token_count <= 0 or self.length + token_count > self.capacity:
            raise ValueError("invalid KV cache length advance")
        self.length += token_count


KVCache: TypeAlias = PastKeyValues | StaticKVCache


def cache_length(cache: KVCache | None) -> int:
    if cache is None:
        return 0
    if isinstance(cache, StaticKVCache):
        return cache.length
    return 0 if not cache else cache[0][0].shape[2]


def validate_past_key_values(cache: KVCache | None, config: QwenToyConfig, batch_size: int) -> None:
    if cache is None:
        return
    if isinstance(cache, StaticKVCache):
        if len(cache.key_cache) != config.num_hidden_layers or len(cache.value_cache) != config.num_hidden_layers:
            raise ValueError("static cache layer count mismatch")
        if not 0 <= cache.length <= cache.capacity <= config.max_position_embeddings:
            raise ValueError("static cache length or capacity is invalid")
        expected = (
            batch_size,
            config.num_key_value_heads,
            cache.capacity,
            config.head_dim,
        )
        for index, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
            if key.shape != expected or value.shape != expected:
                raise ValueError(f"static cache layer {index} K/V shape mismatch")
            if key.device != value.device or key.dtype != value.dtype:
                raise ValueError(f"static cache layer {index} K/V device or dtype mismatch")
        return
    if len(cache) != config.num_hidden_layers:
        raise ValueError(f"cache has {len(cache)} layers; expected {config.num_hidden_layers}")
    length = cache_length(cache)
    for index, (key, value) in enumerate(cache):
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError(f"cache layer {index} K/V shapes must match and have rank 4")
        if key.shape[0] != batch_size:
            raise ValueError(f"cache layer {index} batch mismatch")
        if key.shape[1] != config.num_key_value_heads:
            raise ValueError(f"cache layer {index} KV heads mismatch")
        if key.shape[2] != length or key.shape[3] != config.head_dim:
            raise ValueError(f"cache layer {index} length or head dimension mismatch")
