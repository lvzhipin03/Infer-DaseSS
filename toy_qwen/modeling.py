from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .cache import PastKeyValue, PastKeyValues, cache_length, validate_past_key_values
from .config import QwenToyConfig


class QwenToyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * values.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class QwenToyRotaryEmbedding(nn.Module):
    def __init__(self, config: QwenToyConfig):
        super().__init__()
        exponent = torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim
        self.register_buffer("inv_freq", 1.0 / (config.rope_theta ** exponent), persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype).unsqueeze(1), angles.sin().to(dtype).unsqueeze(1)


def repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, heads, length, dimension = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(batch, heads, groups, length, dimension)
    return expanded.reshape(batch, heads * groups, length, dimension)


def _allowed_attention_mask(batch_size, query_length, key_length, past_length, attention_mask, device):
    query = past_length + torch.arange(query_length, device=device)
    key = torch.arange(key_length, device=device)
    allowed = key.unsqueeze(0) <= query.unsqueeze(1)
    allowed = allowed.view(1, 1, query_length, key_length)
    if attention_mask is not None:
        if attention_mask.shape != (batch_size, key_length):
            raise ValueError("attention_mask must match batch and full key length")
        allowed = allowed & attention_mask[:, None, None, :].bool()
    return allowed


class QwenToyAttention(nn.Module):
    def __init__(self, config: QwenToyConfig, layer_idx: int):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.attn_implementation = "eager"

    def forward(self, hidden_states, position_embeddings, past_key_value=None, attention_mask=None, use_cache=False):
        batch, length, _ = hidden_states.shape
        def shape(x, heads):
            return x.view(batch, length, heads, self.config.head_dim).transpose(1, 2)
        query = shape(self.q_proj(hidden_states), self.config.num_attention_heads)
        key = shape(self.k_proj(hidden_states), self.config.num_key_value_heads)
        value = shape(self.v_proj(hidden_states), self.config.num_key_value_heads)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        past_length = 0
        if past_key_value is not None:
            past_length = past_key_value[0].shape[2]
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present = (key, value) if use_cache else None
        key_length = key.shape[2]
        allowed = _allowed_attention_mask(
            batch, length, key_length, past_length, attention_mask, query.device
        )
        logical_scores_shape = (batch, self.config.num_attention_heads, length, key_length)
        if self.attn_implementation == "eager":
            repeated_key = repeat_kv(key, self.config.num_key_value_groups)
            repeated_value = repeat_kv(value, self.config.num_key_value_groups)
            scores = torch.matmul(query, repeated_key.transpose(2, 3)) * (self.config.head_dim ** -0.5)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores.float(), dim=-1).to(value.dtype)
            output = torch.matmul(probabilities, repeated_value)
        else:
            try:
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=allowed,
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=True,
                )
            except TypeError as error:
                if "enable_gqa" not in str(error):
                    raise
                repeated_key = repeat_kv(key, self.config.num_key_value_groups)
                repeated_value = repeat_kv(value, self.config.num_key_value_groups)
                output = F.scaled_dot_product_attention(
                    query,
                    repeated_key,
                    repeated_value,
                    attn_mask=allowed,
                    dropout_p=0.0,
                    is_causal=False,
                )
            except RuntimeError as error:
                message = str(error).lower()
                if not any(marker in message for marker in ("gqa", "grouped query", "no available kernel")):
                    raise
                repeated_key = repeat_kv(key, self.config.num_key_value_groups)
                repeated_value = repeat_kv(value, self.config.num_key_value_groups)
                output = F.scaled_dot_product_attention(
                    query,
                    repeated_key,
                    repeated_value,
                    attn_mask=allowed,
                    dropout_p=0.0,
                    is_causal=False,
                )
        if attention_mask is not None:
            query_is_valid = attention_mask[:, key_length - length :].bool()
            output = output * query_is_valid[:, None, :, None]
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        output = self.o_proj(output)
        trace = {"query": tuple(query.shape), "key": tuple(key.shape), "value": tuple(value.shape),
                 "repeated_key": logical_scores_shape[:-2] + (key_length, self.config.head_dim),
                 "attention_scores": logical_scores_shape,
                 "attention_output": tuple(output.shape)}
        return output, present, trace


class QwenToyMLP(nn.Module):
    def __init__(self, config: QwenToyConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenToyDecoderLayer(nn.Module):
    def __init__(self, config: QwenToyConfig, layer_idx: int):
        super().__init__()
        self.self_attn = QwenToyAttention(config, layer_idx)
        self.mlp = QwenToyMLP(config)
        self.input_layernorm = QwenToyRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = QwenToyRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, attention_mask=None, use_cache=False):
        residual = hidden_states
        attention, present, trace = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, past_key_value, attention_mask, use_cache)
        hidden_states = residual + attention
        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(mlp_input)
        trace["mlp_input"] = tuple(mlp_input.shape)
        trace["layer_output"] = tuple(hidden_states.shape)
        return hidden_states, present, trace


@dataclass
class BaseModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: PastKeyValues | None
    hidden_states: tuple[torch.Tensor, ...] | None
    trace: dict[str, tuple[int, ...]] | None


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    past_key_values: PastKeyValues | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    trace: dict[str, tuple[int, ...]] | None = None


class QwenToyModel(nn.Module):
    def __init__(self, config: QwenToyConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(QwenToyDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = QwenToyRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = QwenToyRotaryEmbedding(config)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=None, output_hidden_states=False, trace_shapes=False):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty rank-2 tensor")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input token id is outside the vocabulary")
        batch, length = input_ids.shape
        validate_past_key_values(past_key_values, self.config, batch)
        past_length = cache_length(past_key_values)
        if past_length + length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if position_ids is None:
            position_ids = torch.arange(past_length, past_length + length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        hidden = self.embed_tokens(input_ids)
        trace = {"embedding": tuple(hidden.shape)} if trace_shapes else None
        states = [hidden] if output_hidden_states else None
        positions = self.rotary_emb(position_ids, hidden.dtype)
        presents = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden, present, layer_trace = layer(hidden, positions, past, attention_mask, use_cache)
            if use_cache:
                presents.append(present)
            if states is not None:
                states.append(hidden)
            if trace is not None:
                trace.update({f"layer_{index}.{key}": value for key, value in layer_trace.items()})
        hidden = self.norm(hidden)
        if states is not None:
            states.append(hidden)
        if trace is not None:
            trace["final_norm"] = tuple(hidden.shape)
        return BaseModelOutput(hidden, tuple(presents) if use_cache else None,
                               tuple(states) if states is not None else None, trace)


class QwenToyForCausalLM(nn.Module):
    def __init__(self, config: QwenToyConfig):
        super().__init__()
        self.config = config
        self.model = QwenToyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def set_attention_implementation(self, name: str):
        if name not in {"eager", "sdpa"}:
            raise ValueError("attention implementation must be eager or sdpa")
        for layer in self.model.layers:
            layer.self_attn.attn_implementation = name
        return self

    def forward(self, input_ids, num_logits_to_keep=None, **kwargs):
        if num_logits_to_keep is not None and num_logits_to_keep <= 0:
            raise ValueError("num_logits_to_keep must be positive")
        output = self.model(input_ids, **kwargs)
        hidden_states = output.last_hidden_state
        if num_logits_to_keep is not None:
            hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        return CausalLMOutput(self.lm_head(hidden_states), output.past_key_values,
                              output.hidden_states, output.trace)
