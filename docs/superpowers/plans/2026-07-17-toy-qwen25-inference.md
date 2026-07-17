# Toy Qwen2.5 Minimal Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-PyTorch, CPU-runnable, configuration-driven Qwen2-style toy causal LM that loads the supplied whiteboard weights and predicts “北” after “中国首都是” through a real forward pass.

**Architecture:** Keep Hugging Face Qwen2 module names and pre-norm decoder math while allowing the whiteboard dimensions (`H=4`, one layer, one head) and the full Qwen2.5-0.5B dimensions to come from configuration. Isolate tokenization, cache validation, model math, whiteboard weight conversion, and inference so a full tokenizer and safetensors loader can be added without changing forward.

**Tech Stack:** Python 3.10+, PyTorch, `dataclasses`, `unittest`, standard-library `argparse` and `json`; no GPU, training, Transformers, NumPy, pytest, or network dependency.

## Global Constraints

- The authoritative toy weights are in `references/whiteboard_weights_summary.md`; do not train or silently alter them.
- The required example is exactly input `中国首都是` and greedy next token `北`.
- `model.forward()` accepts tensors only and must not contain prompt strings, token strings, or post-hoc logit overrides.
- Whiteboard runtime is CPU float32 and depends only on PyTorch plus the Python standard library.
- Qwen2 math includes float32 RMS variance, Qwen `rotate_half` RoPE, causal attention, generalized GQA, SwiGLU, two residuals, final RMSNorm, and a bias-free LM Head.
- Model-internal token IDs are dense `0..8`; source IDs `10..90` are trace-only legacy IDs.
- Whiteboard matrices documented as `[in,out]` must be transposed when copied into `nn.Linear.weight` (`[out,in]`).
- Preserve Hugging Face-style state-dict paths such as `model.layers.0.self_attn.q_proj.weight`.
- Do not implement training, sampling, padding batches, sliding-window attention, quantization, safetensors loading, or a Hugging Face tokenizer in this version.

---

## File Map

| File | Responsibility |
|---|---|
| `toy_qwen/__init__.py` | Public imports only |
| `toy_qwen/config.py` | Config dataclass, JSON loading, whiteboard/full presets, validation |
| `toy_qwen/tokenizer.py` | Nine-character tokenizer and legacy-ID trace |
| `toy_qwen/cache.py` | Cache aliases, length lookup, structural validation |
| `toy_qwen/modeling.py` | RMSNorm, RoPE, GQA attention, SwiGLU, decoder, model, causal LM |
| `toy_qwen/weights.py` | Exact whiteboard tensors, orientation conversion, shape-checked loading |
| `toy_qwen/inference.py` | Prediction record and greedy next-token orchestration |
| `configs/whiteboard_toy.json` | Runnable toy preset snapshot |
| `configs/qwen2_5_0_5b.json` | Future full-size architecture snapshot |
| `whiteboard_llm_inference.py` | CLI and human-readable shape/logit output |
| `tests/test_config.py` | Config parsing and invalid combinations |
| `tests/test_tokenizer.py` | Dense IDs, legacy IDs, decoding, errors |
| `tests/test_primitives.py` | RMSNorm, RoPE, cache validation |
| `tests/test_attention.py` | Causality, shape, 14Q/2KV expansion, cache equality |
| `tests/test_model.py` | SwiGLU, residual/model shapes, tied-weight behavior, state paths |
| `tests/test_weights.py` | Reference tensors and transposition |
| `tests/test_inference.py` | Reference logits, next token, anti-hardcoding, CLI |
| `README.md` | Run instructions, tensor flow, limitations, full-model migration |

---

### Task 1: Configuration and package foundation

**Files:**
- Create: `toy_qwen/__init__.py`
- Create: `toy_qwen/config.py`
- Create: `configs/whiteboard_toy.json`
- Create: `configs/qwen2_5_0_5b.json`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `QwenToyConfig`, `QwenToyConfig.from_json(path)`, `whiteboard_config()`, `qwen25_05b_config()`.
- Consumers: Every later model, weight, and inference task.

- [ ] **Step 1: Write config tests first**

Create `tests/test_config.py` with these concrete cases:

```python
import json
import tempfile
import unittest
from pathlib import Path

from toy_qwen.config import QwenToyConfig, qwen25_05b_config, whiteboard_config


class ConfigTest(unittest.TestCase):
    def test_whiteboard_values_and_derived_head_dim(self):
        config = whiteboard_config()
        self.assertEqual((config.vocab_size, config.hidden_size), (9, 4))
        self.assertEqual((config.num_attention_heads, config.num_key_value_heads), (1, 1))
        self.assertEqual(config.head_dim, 4)
        self.assertFalse(config.attention_bias)
        self.assertFalse(config.tie_word_embeddings)

    def test_full_preset_matches_official_structural_values(self):
        config = qwen25_05b_config()
        self.assertEqual(config.hidden_size, 896)
        self.assertEqual(config.intermediate_size, 4864)
        self.assertEqual(config.num_hidden_layers, 24)
        self.assertEqual((config.num_attention_heads, config.num_key_value_heads), (14, 2))
        self.assertEqual(config.rope_theta, 1_000_000.0)
        self.assertTrue(config.attention_bias)
        self.assertTrue(config.tie_word_embeddings)

    def test_json_loader_ignores_known_non_architecture_metadata(self):
        payload = dict(whiteboard_config().to_dict(), architectures=["Qwen2ForCausalLM"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(QwenToyConfig.from_json(path), whiteboard_config())

    def test_official_qwen_config_implies_qkv_bias(self):
        payload = qwen25_05b_config().to_dict()
        del payload["attention_bias"]
        self.assertTrue(QwenToyConfig.from_dict(payload).attention_bias)

    def test_invalid_head_combinations_fail_early(self):
        with self.assertRaisesRegex(ValueError, "hidden_size.*num_attention_heads"):
            QwenToyConfig(vocab_size=9, hidden_size=5, intermediate_size=8,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=1)
        with self.assertRaisesRegex(ValueError, "even"):
            QwenToyConfig(vocab_size=9, hidden_size=6, intermediate_size=8,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=1)
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            QwenToyConfig(vocab_size=9, hidden_size=8, intermediate_size=8,
                          num_hidden_layers=1, num_attention_heads=4,
                          num_key_value_heads=3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the intended failure**

Run:

```bash
python -m unittest tests.test_config -v
```

Expected: `ERROR` with `ModuleNotFoundError: No module named 'toy_qwen'`.

- [ ] **Step 3: Implement the validated config**

Implement `toy_qwen/config.py` as a frozen dataclass with these exact public fields:

```python
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
        if self.vocab_size <= 0 or self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("vocab_size, hidden_size, and intermediate_size must be positive")
        if self.num_hidden_layers <= 0 or self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
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
            raise ValueError("sliding-window attention is outside the toy implementation")

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
        # Official Qwen2 config omits this field although Q/K/V use bias.
        if payload.get("model_type") == "qwen2" and "attention_bias" not in payload:
            values["attention_bias"] = True
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "QwenToyConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def whiteboard_config() -> QwenToyConfig:
    return QwenToyConfig(vocab_size=9, hidden_size=4, intermediate_size=8,
                         num_hidden_layers=1, num_attention_heads=1,
                         num_key_value_heads=1)


def qwen25_05b_config() -> QwenToyConfig:
    return QwenToyConfig(
        vocab_size=151936, hidden_size=896, intermediate_size=4864,
        num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2,
        max_position_embeddings=32768, rope_theta=1_000_000.0,
        rms_norm_eps=1e-6, attention_dropout=0.0, initializer_range=0.02,
        attention_bias=True, tie_word_embeddings=True, use_cache=True,
        sliding_window=32768, max_window_layers=21,
        bos_token_id=151643, eos_token_id=151645,
    )
```

Export `QwenToyConfig`, `whiteboard_config`, and `qwen25_05b_config` from `toy_qwen/__init__.py`. Write both JSON files from the exact `to_dict()` values; do not invent divergent hand-maintained keys.

- [ ] **Step 4: Run config tests**

Run `python -m unittest tests.test_config -v`.

Expected: `Ran 5 tests ... OK`.

- [ ] **Step 5: Commit the foundation**

```bash
git add toy_qwen/__init__.py toy_qwen/config.py configs tests/test_config.py
git commit -m "feat: add parameterized qwen toy configuration"
```

---

### Task 2: Character tokenizer and legacy trace

**Files:**
- Create: `toy_qwen/tokenizer.py`
- Modify: `toy_qwen/__init__.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces: `ToyTokenizer.encode(text) -> list[int]`, `decode(ids) -> str`, `legacy_ids(text) -> list[int]`, `token(id) -> str`.
- Consumes: no model code.

- [ ] **Step 1: Add failing tokenizer tests**

```python
import unittest
from toy_qwen.tokenizer import ToyTokenizer


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ToyTokenizer()

    def test_required_prompt_uses_dense_and_legacy_ids(self):
        self.assertEqual(self.tokenizer.encode("中国首都是"), [0, 1, 2, 3, 4])
        self.assertEqual(self.tokenizer.legacy_ids("中国首都是"), [10, 20, 30, 40, 50])
        self.assertEqual(self.tokenizer.decode([5]), "北")

    def test_round_trip_all_tokens(self):
        text = "中国首都是北京上海"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_unknown_character_reports_position(self):
        with self.assertRaisesRegex(ValueError, "未知.*位置 1"):
            self.tokenizer.encode("中法")

    def test_invalid_decode_id_fails(self):
        with self.assertRaisesRegex(ValueError, "token id 9"):
            self.tokenizer.decode([9])
```

- [ ] **Step 2: Verify failure**

Run `python -m unittest tests.test_tokenizer -v`.

Expected: import error for `toy_qwen.tokenizer`.

- [ ] **Step 3: Implement the tokenizer without model coupling**

```python
class ToyTokenizer:
    TOKENS = ("中", "国", "首", "都", "是", "北", "京", "上", "海")
    LEGACY_IDS = (10, 20, 30, 40, 50, 60, 70, 80, 90)

    def __init__(self) -> None:
        self._token_to_id = {token: index for index, token in enumerate(self.TOKENS)}

    @property
    def vocab_size(self) -> int:
        return len(self.TOKENS)

    def encode(self, text: str) -> list[int]:
        ids = []
        for position, token in enumerate(text):
            if token not in self._token_to_id:
                raise ValueError(f"未知字符 {token!r}，位置 {position}")
            ids.append(self._token_to_id[token])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.token(token_id) for token_id in token_ids)

    def token(self, token_id: int) -> str:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"invalid token id {token_id}")
        return self.TOKENS[token_id]

    def legacy_ids(self, text: str) -> list[int]:
        return [self.LEGACY_IDS[token_id] for token_id in self.encode(text)]
```

- [ ] **Step 4: Run tests and commit**

Run `python -m unittest tests.test_tokenizer -v`; expect four passing tests.

```bash
git add toy_qwen/tokenizer.py toy_qwen/__init__.py tests/test_tokenizer.py
git commit -m "feat: add whiteboard character tokenizer"
```

---

### Task 3: RMSNorm, Qwen RoPE, and cache validation

**Files:**
- Create: `toy_qwen/cache.py`
- Create: `toy_qwen/modeling.py`
- Test: `tests/test_primitives.py`

**Interfaces:**
- Produces: `PastKeyValue`, `PastKeyValues`, `cache_length`, `validate_past_key_values`, `QwenToyRMSNorm`, `QwenToyRotaryEmbedding`, `rotate_half`, `apply_rotary_pos_emb`.
- Consumes: `QwenToyConfig`.

- [ ] **Step 1: Write mathematical primitive tests**

Tests must compare RMSNorm to the direct formula, verify Qwen half-split rotation, and reject malformed cache:

```python
import unittest
import torch
from toy_qwen.cache import cache_length, validate_past_key_values
from toy_qwen.config import whiteboard_config
from toy_qwen.modeling import QwenToyRMSNorm, apply_rotary_pos_emb, rotate_half


class PrimitiveTest(unittest.TestCase):
    def test_rmsnorm_matches_direct_float32_formula(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        layer = QwenToyRMSNorm(4, eps=1e-6)
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(layer(x), expected)

    def test_rotate_half_uses_qwen_front_back_pairs(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))

    def test_rope_preserves_norm(self):
        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        cos = torch.zeros_like(q)
        sin = torch.ones_like(q)
        rotated, _ = apply_rotary_pos_emb(q, q, cos, sin)
        torch.testing.assert_close(rotated.square().sum(-1), q.square().sum(-1))

    def test_cache_shape_validation(self):
        config = whiteboard_config()
        valid = ((torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),)
        self.assertEqual(cache_length(valid), 3)
        validate_past_key_values(valid, config, batch_size=1)
        invalid = ((torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),)
        with self.assertRaisesRegex(ValueError, "KV heads"):
            validate_past_key_values(invalid, config, batch_size=1)
```

- [ ] **Step 2: Verify import failures**

Run `python -m unittest tests.test_primitives -v`; expect missing symbol/module errors.

- [ ] **Step 3: Implement cache helpers**

`toy_qwen/cache.py` must define tuple aliases and validate layer count, equal K/V shapes, rank 4, batch, KV heads, head dimension, and a common cached length. `cache_length(None)` and `cache_length(())` return zero; a valid cache returns `shape[2]`.

```python
from typing import TypeAlias
import torch
from .config import QwenToyConfig

PastKeyValue: TypeAlias = tuple[torch.Tensor, torch.Tensor]
PastKeyValues: TypeAlias = tuple[PastKeyValue, ...]


def cache_length(cache: PastKeyValues | None) -> int:
    return 0 if not cache else cache[0][0].shape[2]


def validate_past_key_values(cache, config, batch_size):
    if cache is None:
        return
    if len(cache) != config.num_hidden_layers:
        raise ValueError(f"cache has {len(cache)} layers; expected {config.num_hidden_layers}")
    expected_length = cache_length(cache)
    for index, (key, value) in enumerate(cache):
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError(f"cache layer {index} K/V shapes must match and have rank 4")
        if key.shape[0] != batch_size:
            raise ValueError(f"cache layer {index} batch mismatch")
        if key.shape[1] != config.num_key_value_heads:
            raise ValueError(f"cache layer {index} KV heads mismatch")
        if key.shape[2] != expected_length or key.shape[3] != config.head_dim:
            raise ValueError(f"cache layer {index} length or head dimension mismatch")
```

- [ ] **Step 4: Implement RMSNorm and Qwen RoPE exactly**

Add to `toy_qwen/modeling.py`:

```python
class QwenToyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


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
```

- [ ] **Step 5: Run and commit**

Run `python -m unittest tests.test_primitives -v`; expect four passing tests.

```bash
git add toy_qwen/cache.py toy_qwen/modeling.py tests/test_primitives.py
git commit -m "feat: add qwen normalization rope and cache primitives"
```

---

### Task 4: Generalized grouped-query causal attention

**Files:**
- Modify: `toy_qwen/modeling.py`
- Test: `tests/test_attention.py`

**Interfaces:**
- Produces: `repeat_kv(hidden, groups)`, `QwenToyAttention.forward(hidden_states, position_embeddings, past_key_value=None, attention_mask=None, use_cache=False)` returning `(output, present, trace)`.
- Consumes: config, RoPE helpers, `PastKeyValue`.

- [ ] **Step 1: Write attention tests**

Include these exact behavioral assertions:

```python
class AttentionTest(unittest.TestCase):
    def test_repeat_kv_14_query_2_kv_heads(self):
        source = torch.arange(2 * 3 * 2.0).reshape(1, 2, 3, 2)
        repeated = repeat_kv(source, 7)
        self.assertEqual(repeated.shape, (1, 14, 3, 2))
        for kv_head in range(2):
            for offset in range(7):
                torch.testing.assert_close(repeated[:, kv_head * 7 + offset], source[:, kv_head])

    def test_future_token_cannot_change_earlier_output(self):
        config = QwenToyConfig(vocab_size=9, hidden_size=4, intermediate_size=8,
                               num_hidden_layers=1, num_attention_heads=1,
                               num_key_value_heads=1)
        attention = QwenToyAttention(config, layer_idx=0).eval()
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
            projection.weight.data.copy_(torch.eye(4))
        x = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
        changed = x.clone(); changed[:, 1] = torch.tensor([9., 9., 9., 9.])
        positions = torch.tensor([[0, 1]])
        rope = QwenToyRotaryEmbedding(config)
        cos, sin = rope(positions, x.dtype)
        original, _, _ = attention(x, (cos, sin))
        modified, _, _ = attention(changed, (cos, sin))
        torch.testing.assert_close(original[:, 0], modified[:, 0])

    def test_attention_trace_has_whiteboard_shapes(self):
        config = whiteboard_config()
        attention = QwenToyAttention(config, layer_idx=0)
        x = torch.zeros(1, 5, 4)
        rope = QwenToyRotaryEmbedding(config)
        cos, sin = rope(torch.arange(5).unsqueeze(0), x.dtype)
        output, cache, trace = attention(x, (cos, sin), use_cache=True)
        self.assertEqual(output.shape, (1, 5, 4))
        self.assertEqual(cache[0].shape, (1, 1, 5, 4))
        self.assertEqual(trace["attention_scores"], (1, 1, 5, 5))
```

- [ ] **Step 2: Run tests and observe missing implementation**

Run `python -m unittest tests.test_attention -v`; expected import/name failures.

- [ ] **Step 3: Implement GQA and causal masking**

Implementation requirements, expressed as the exact computation order:

```python
def repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, kv_heads, groups, length, head_dim)
    return expanded.reshape(batch, kv_heads * groups, length, head_dim)


def _causal_mask(query_length, key_length, past_length, dtype, device):
    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    blocked = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    mask = torch.zeros(query_length, key_length, dtype=dtype, device=device)
    return mask.masked_fill(blocked, torch.finfo(dtype).min).view(1, 1, query_length, key_length)
```

`QwenToyAttention.__init__` creates Q output `Nq*D`, K/V output `Nkv*D`, and O output H. Q/K/V use `config.attention_bias`; O never uses bias. `forward` must project, reshape/transposes to `[B,N,T,D]`, apply RoPE, append past K/V on sequence dimension, construct `present` before `repeat_kv`, compute scaled scores by `D**-0.5`, add causal and optional padding mask, softmax in float32, multiply V, merge heads, and apply O. The trace must contain `query`, `key`, `value`, `repeated_key`, `attention_scores`, and `attention_output` shapes.

- [ ] **Step 4: Run attention tests and the existing suite**

Run:

```bash
python -m unittest tests.test_attention -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add toy_qwen/modeling.py tests/test_attention.py
git commit -m "feat: add grouped-query causal attention"
```

---

### Task 5: SwiGLU decoder stack and causal LM

**Files:**
- Modify: `toy_qwen/modeling.py`
- Modify: `toy_qwen/__init__.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `CausalLMOutput`, `QwenToyMLP`, `QwenToyDecoderLayer`, `QwenToyModel`, `QwenToyForCausalLM`.
- Consumes: config, cache validation, attention.

- [ ] **Step 1: Write model tests**

Tests must verify the direct SwiGLU formula, full output/cache shapes, tied weights, and HF-like names:

```python
class ModelTest(unittest.TestCase):
    def test_mlp_matches_swiglu_formula(self):
        config = whiteboard_config()
        mlp = QwenToyMLP(config)
        x = torch.randn(1, 2, 4)
        expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
        torch.testing.assert_close(mlp(x), expected)

    def test_whiteboard_model_shapes_and_cache(self):
        model = QwenToyForCausalLM(whiteboard_config()).eval()
        output = model(torch.tensor([[0, 1, 2, 3, 4]]), use_cache=True,
                       output_hidden_states=True, trace_shapes=True)
        self.assertEqual(output.logits.shape, (1, 5, 9))
        self.assertEqual(output.past_key_values[0][0].shape, (1, 1, 5, 4))
        self.assertEqual(len(output.hidden_states), 3)  # embedding, layer, final norm
        self.assertEqual(output.trace["embedding"], (1, 5, 4))

    def test_tied_config_shares_parameter_object(self):
        config = replace(whiteboard_config(), tie_word_embeddings=True)
        model = QwenToyForCausalLM(config)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_hugging_face_style_parameter_paths(self):
        names = set(QwenToyForCausalLM(whiteboard_config()).state_dict())
        self.assertIn("model.embed_tokens.weight", names)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", names)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", names)
        self.assertIn("model.norm.weight", names)
        self.assertIn("lm_head.weight", names)

    def test_14q_2kv_parameterized_forward(self):
        config = QwenToyConfig(vocab_size=9, hidden_size=28, intermediate_size=152,
                               num_hidden_layers=2, num_attention_heads=14,
                               num_key_value_heads=2)
        output = QwenToyForCausalLM(config)(torch.tensor([[0, 1, 2]]))
        self.assertEqual(output.logits.shape, (1, 3, 9))
```

- [ ] **Step 2: Verify tests fail before model classes exist**

Run `python -m unittest tests.test_model -v` and expect missing-symbol failures.

- [ ] **Step 3: Implement records, MLP, and decoder**

```python
@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    past_key_values: PastKeyValues | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    trace: dict[str, tuple[int, ...]] | None = None


class QwenToyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

`QwenToyDecoderLayer.forward` must save `residual`, normalize/attend/add, save the second residual, normalize/MLP/add, then return hidden state, optional present cache, and namespaced attention/MLP trace. It must not mutate input cache.

- [ ] **Step 4: Implement model and causal LM validation**

`QwenToyModel` must own `embed_tokens`, `layers`, `norm`, and `rotary_emb`. Validate rank-2 integer input IDs, non-empty T, ID range, cache shape, and total sequence length. Generate positions starting at cache length, calculate RoPE once, collect cache and optional hidden states, then final-normalize.

`QwenToyForCausalLM` must own `model` and bias-configured `lm_head`; if tying is enabled, assign the exact embedding `Parameter` object to `lm_head.weight`. Its forward delegates all model arguments and projects every returned position to `[B,T,V]`.

- [ ] **Step 5: Run all model tests and commit**

Run `python -m unittest tests.test_model -v` and `python -m unittest discover -s tests -v`; expect all passing.

```bash
git add toy_qwen/modeling.py toy_qwen/__init__.py tests/test_model.py
git commit -m "feat: add qwen decoder and causal language model"
```

---

### Task 6: Exact whiteboard weight loader

**Files:**
- Create: `toy_qwen/weights.py`
- Test: `tests/test_weights.py`

**Interfaces:**
- Produces: `load_whiteboard_weights(model) -> None`, `build_whiteboard_model() -> QwenToyForCausalLM`.
- Consumes: whiteboard config and model.

- [ ] **Step 1: Write exact-value and orientation tests**

```python
class WeightTest(unittest.TestCase):
    def test_embedding_and_lm_rows_match_reference(self):
        model = build_whiteboard_model()
        torch.testing.assert_close(model.model.embed_tokens.weight[0], torch.tensor([2., 0., 0., 0.]))
        torch.testing.assert_close(model.model.embed_tokens.weight[5], torch.tensor([1., 1., 0., 0.]))
        torch.testing.assert_close(model.lm_head.weight[5], torch.tensor([2., 2., -1., -1.]))
        torch.testing.assert_close(model.lm_head.weight[:5], torch.zeros(5, 4))

    def test_projection_and_norm_values(self):
        model = build_whiteboard_model()
        layer = model.model.layers[0]
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                           layer.self_attn.v_proj, layer.self_attn.o_proj):
            torch.testing.assert_close(projection.weight, torch.eye(4))
        torch.testing.assert_close(layer.input_layernorm.weight, torch.ones(4))
        torch.testing.assert_close(layer.post_attention_layernorm.weight, torch.ones(4))
        torch.testing.assert_close(model.model.norm.weight, torch.ones(4))

    def test_ffn_source_matrix_is_transposed_for_linear(self):
        model = build_whiteboard_model()
        self.assertAlmostEqual(model.model.layers[0].mlp.gate_proj.weight[0, 0].item(), -0.2252, places=4)
        self.assertAlmostEqual(model.model.layers[0].mlp.gate_proj.weight[1, 0].item(), -0.2305, places=4)

    def test_loader_rejects_non_whiteboard_shape(self):
        other = QwenToyForCausalLM(QwenToyConfig(vocab_size=9, hidden_size=8,
            intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1))
        with self.assertRaisesRegex(ValueError, "whiteboard.*hidden_size"):
            load_whiteboard_weights(other)
```

- [ ] **Step 2: Verify failure**

Run `python -m unittest tests.test_weights -v`; expect import failure.

- [ ] **Step 3: Implement deterministic source tensors**

Use this exact embedding and LM data:

```python
EMBEDDING = torch.tensor([
    [2., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 2., 0.],
    [0., 0., 0., 2.], [1., 1., -1., -1.], [1., 1., 0., 0.],
    [.8, .8, 0., 0.], [0., 0., 1., 1.], [0., 0., .8, .8],
])
LM_CANDIDATES = torch.tensor([
    [2., 2., -1., -1.], [1.2, 1.2, -.5, -.5],
    [-1., -1., 2., 2.], [-.8, -.8, 1.2, 1.2],
])
```

Generate FFN matrices locally with a dedicated generator so global RNG state is not changed:

```python
generator = torch.Generator(device="cpu").manual_seed(0)
w_gate = torch.randn(4, 8, generator=generator) * 0.2
w_up = torch.randn(4, 8, generator=generator) * 0.2
w_down = torch.randn(8, 4, generator=generator) * 0.2
```

Within `torch.no_grad()`, copy embedding, unit projections, norm ones, transposed FFN tensors, and a zeroed `[9,4]` LM matrix whose rows 5:9 receive `LM_CANDIDATES`. Check config fields before any copy so a failure cannot partially load a model.

- [ ] **Step 4: Run weight and regression suites**

Run `python -m unittest tests.test_weights -v` and the full discovery command; expect all passing.

- [ ] **Step 5: Commit**

```bash
git add toy_qwen/weights.py tests/test_weights.py
git commit -m "feat: load deterministic whiteboard weights"
```

---

### Task 7: Greedy inference, cache equivalence, and CLI

**Files:**
- Create: `toy_qwen/inference.py`
- Create: `whiteboard_llm_inference.py`
- Modify: `toy_qwen/__init__.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Produces: `Prediction`, `predict_next_token(text, model, tokenizer)`, CLI exit code 0.
- Consumes: tokenizer, loaded model, output logits and trace.

- [ ] **Step 1: Write end-to-end tests before inference code**

```python
class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.model = build_whiteboard_model().eval()
        self.tokenizer = ToyTokenizer()

    def test_reference_logits_and_required_next_token(self):
        prediction = predict_next_token("中国首都是", self.model, self.tokenizer)
        self.assertEqual((prediction.token, prediction.token_id), ("北", 5))
        self.assertGreater(prediction.logit, prediction.runner_up_logit)
        expected = torch.tensor([6.199933, 3.560251, -5.495592, -3.757640])
        torch.testing.assert_close(prediction.logits[[5, 6, 7, 8]], expected,
                                   atol=1e-5, rtol=1e-5)

    def test_prefill_plus_cached_decode_matches_full_forward(self):
        full_ids = torch.tensor([self.tokenizer.encode("中国首都是")])
        with torch.no_grad():
            full = self.model(full_ids).logits[:, -1]
            prefill = self.model(full_ids[:, :-1], use_cache=True)
            cached = self.model(full_ids[:, -1:], past_key_values=prefill.past_key_values,
                                use_cache=True).logits[:, -1]
        torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)

    def test_prediction_follows_weights_not_prompt_branch(self):
        original = self.model.lm_head.weight.detach().clone()
        self.model.lm_head.weight.data[[5, 6]] = original[[6, 5]]
        prediction = predict_next_token("中国首都是", self.model, self.tokenizer)
        self.assertEqual(prediction.token, "京")

    def test_empty_prompt_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            predict_next_token("", self.model, self.tokenizer)

    def test_cli_prints_expected_token(self):
        completed = subprocess.run(
            [sys.executable, "whiteboard_llm_inference.py", "--prompt", "中国首都是",
             "--trace-shapes"], text=True, capture_output=True, check=True)
        self.assertIn("dense ids: [0, 1, 2, 3, 4]", completed.stdout)
        self.assertTrue(completed.stdout.rstrip().endswith("next token: 北"))
```

- [ ] **Step 2: Verify inference import/CLI failures**

Run `python -m unittest tests.test_inference -v`; expected missing module or script errors.

- [ ] **Step 3: Implement prediction without logit mutation**

```python
@dataclass(frozen=True)
class Prediction:
    token: str
    token_id: int
    logit: float
    runner_up_token: str
    runner_up_logit: float
    logits: torch.Tensor
    trace: dict[str, tuple[int, ...]]


def predict_next_token(text, model, tokenizer, trace_shapes=False):
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError("prompt must not be empty")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids, use_cache=model.config.use_cache,
                       trace_shapes=trace_shapes)
    logits = output.logits[0, -1].detach().cpu()
    ranking = torch.argsort(logits, descending=True)
    winner, runner_up = ranking[:2].tolist()
    return Prediction(tokenizer.token(winner), winner, logits[winner].item(),
                      tokenizer.token(runner_up), logits[runner_up].item(), logits,
                      output.trace or {})
```

- [ ] **Step 4: Implement the CLI**

The CLI must parse `--prompt` (default `中国首都是`) and `--trace-shapes`; build the model, print prompt/dense/legacy IDs, optionally print sorted trace entries, print all nine token logits in token order, print winner and runner-up margin, and make `next token: 北` the final line.

- [ ] **Step 5: Run end-to-end verification and commit**

```bash
python -m unittest tests.test_inference -v
python whiteboard_llm_inference.py --prompt 中国首都是 --trace-shapes
python -m unittest discover -s tests -v
```

Expected: all tests pass; CLI final line is `next token: 北`.

```bash
git add toy_qwen/inference.py toy_qwen/__init__.py whiteboard_llm_inference.py tests/test_inference.py
git commit -m "feat: add whiteboard next-token inference"
```

---

### Task 8: Documentation, config snapshots, and final compatibility audit

**Files:**
- Create: `README.md`
- Verify: `configs/whiteboard_toy.json`
- Verify: `configs/qwen2_5_0_5b.json`
- Verify: all source and tests

**Interfaces:**
- Produces: reproducible user instructions and documented migration boundary.
- Consumes: all prior tasks.

- [ ] **Step 1: Write README acceptance checks as executable commands**

Document prerequisites, exact run command, expected token, tensor-shape table, whiteboard matrix orientation, why 1Q/1KV is the degenerate GQA case, cache prefill/decode behavior, and the two config presets. State explicitly that full safetensors/tokenizer/runtime optimization are future adapters rather than current claims.

- [ ] **Step 2: Check config snapshots against constructors**

Run:

```bash
python -c "from toy_qwen.config import QwenToyConfig,whiteboard_config,qwen25_05b_config; assert QwenToyConfig.from_json('configs/whiteboard_toy.json') == whiteboard_config(); assert QwenToyConfig.from_json('configs/qwen2_5_0_5b.json') == qwen25_05b_config()"
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Audit forbidden hardcoding and parameter paths**

Run:

```bash
rg -n "中国首都是|next token.*北|logits\[.*\] =" toy_qwen
python -c "from toy_qwen.weights import build_whiteboard_model; names=set(build_whiteboard_model().state_dict()); assert 'model.layers.0.self_attn.q_proj.weight' in names; assert 'lm_head.weight' in names"
```

Expected: `rg` returns no matches in `toy_qwen`; parameter-path check exits 0.

- [ ] **Step 4: Run final clean verification**

```bash
python -m unittest discover -s tests -v
python whiteboard_llm_inference.py --prompt 中国首都是 --trace-shapes
git diff --check
git status --short
```

Expected: every test is `ok`; CLI ends with `next token: 北`; `git diff --check` has no output; status contains only intended files.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md configs docs toy_qwen tests whiteboard_llm_inference.py
git commit -m "docs: explain toy qwen inference and extension path"
```

---

## Final Acceptance Checklist

- [ ] `python -m unittest discover -s tests -v` exits 0.
- [ ] CPU CLI inference exits 0 without model downloads.
- [ ] Input `中国首都是` produces dense IDs `[0,1,2,3,4]` and next token `北`.
- [ ] The four supplied candidate logits match the reviewed Qwen-RoPE baseline within `1e-5`.
- [ ] Cached and uncached final logits match within `1e-5`.
- [ ] A 14Q/2KV parameterized forward test passes.
- [ ] Swapping “北/京” LM rows changes the prediction, proving weight-driven behavior.
- [ ] All whiteboard FFN values originate from an isolated seed-0 generator and correct transpose.
- [ ] No prompt text or forced token appears in `toy_qwen/` implementation code.
- [ ] HF-style module paths and tied/untied embedding configurations are both tested.
