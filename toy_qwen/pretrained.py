from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .config import QwenToyConfig
from .modeling import QwenToyForCausalLM


REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
)


@dataclass(frozen=True)
class CheckpointReport:
    tensor_count: int
    expected_tied_missing: tuple[str, ...]


def validate_checkpoint(
    model: QwenToyForCausalLM,
    state_dict: Mapping[str, torch.Tensor],
) -> CheckpointReport:
    expected = model.state_dict()
    expected_keys = set(expected)
    actual_keys = set(state_dict)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    mismatched = {
        key: (tuple(state_dict[key].shape), tuple(expected[key].shape))
        for key in expected_keys & actual_keys
        if state_dict[key].shape != expected[key].shape
    }
    tied_alias_conflict = False
    if model.config.tie_word_embeddings and {
        "model.embed_tokens.weight", "lm_head.weight"
    } <= actual_keys:
        embedding = state_dict["model.embed_tokens.weight"]
        lm_head = state_dict["lm_head.weight"]
        tied_alias_conflict = embedding.shape == lm_head.shape and not torch.equal(embedding, lm_head)
    invalid_missing = missing - allowed_missing
    if invalid_missing or unexpected or mismatched or tied_alias_conflict:
        details = []
        if invalid_missing:
            details.append(f"missing={sorted(invalid_missing)}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)}")
        if mismatched:
            details.append(f"shape_mismatches={mismatched}")
        if tied_alias_conflict:
            details.append("tied aliases model.embed_tokens.weight and lm_head.weight contain different values")
        raise ValueError("invalid checkpoint: " + "; ".join(details))
    return CheckpointReport(
        tensor_count=len(state_dict),
        expected_tied_missing=tuple(sorted(missing & allowed_missing)),
    )


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    supported = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if isinstance(dtype, torch.dtype):
        if dtype not in supported.values():
            raise ValueError("dtype must be float32 or bfloat16")
        return dtype
    try:
        return supported[dtype]
    except KeyError as error:
        raise ValueError("dtype must be 'float32' or 'bfloat16'") from error


def load_pretrained_qwen(
    model_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "bfloat16",
) -> tuple[QwenToyForCausalLM, CheckpointReport]:
    model_dir = Path(model_path)
    missing_files = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(f"model directory is missing required files: {missing_files}")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    target_dtype = _resolve_dtype(dtype)

    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("install safetensors==0.4.5 to load real Qwen weights") from error

    config = QwenToyConfig.from_json(model_dir / "config.json")
    model = QwenToyForCausalLM(config).to(dtype=target_dtype)
    state_dict = load_file(str(model_dir / "model.safetensors"), device="cpu")
    report = validate_checkpoint(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if set(incompatible.missing_keys) != set(report.expected_tied_missing) or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint changed during loading: {incompatible}")
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.to(target_device)
    model.eval()
    return model, report
