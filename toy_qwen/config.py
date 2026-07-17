from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QwenToyConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 64
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    attention_bias: bool = False
    mlp_bias: bool = False
    lm_head_bias: bool = False
    tie_word_embeddings: bool = False
    use_cache: bool = True
    use_sliding_window: bool = False
    sliding_window: int = 64
    max_window_layers: int = 1
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    model_type: str = "qwen2"

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.hidden_size, self.intermediate_size) <= 0:
            raise ValueError("vocab_size, hidden_size, and intermediate_size must be positive")
        if min(self.num_hidden_layers, self.num_attention_heads, self.num_key_value_heads) <= 0:
            raise ValueError("layer and head counts must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.hidden_act != "silu":
            raise ValueError("only hidden_act='silu' is supported")
        if self.use_sliding_window:
            raise ValueError("sliding-window attention is not implemented")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QwenToyConfig":
        accepted = {field.name for field in fields(cls)}
        values = {key: value for key, value in payload.items() if key in accepted}
        if payload.get("model_type") == "qwen2" and "attention_bias" not in payload:
            values["attention_bias"] = True
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "QwenToyConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def whiteboard_config() -> QwenToyConfig:
    return QwenToyConfig(9, 4, 8, 1, 1, 1)


def qwen25_05b_config() -> QwenToyConfig:
    return QwenToyConfig(
        151936, 896, 4864, 24, 14, 2,
        max_position_embeddings=32768, sliding_window=32768,
        max_window_layers=21, attention_bias=True,
        tie_word_embeddings=True, bos_token_id=151643, eos_token_id=151645,
    )
