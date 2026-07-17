from typing import TypeAlias
import torch
from .config import QwenToyConfig

PastKeyValue: TypeAlias = tuple[torch.Tensor, torch.Tensor]
PastKeyValues: TypeAlias = tuple[PastKeyValue, ...]


def cache_length(cache: PastKeyValues | None) -> int:
    return 0 if not cache else cache[0][0].shape[2]


def validate_past_key_values(cache: PastKeyValues | None, config: QwenToyConfig, batch_size: int) -> None:
    if cache is None:
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
