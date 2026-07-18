#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toy_qwen.pretrained import load_pretrained_qwen
from toy_qwen.qwen_tokenizer import QwenTokenizerAdapter


DEFAULT_MODEL_PATH = "/ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare custom Qwen logits with a Transformers oracle")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="中国的首都是哪里？")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("install requirements-verify.txt to run the oracle") from error

    tokenizer = QwenTokenizerAdapter.from_model_dir(args.model_path)
    rendered, token_ids = tokenizer.encode_chat([{"role": "user", "content": args.prompt}])
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    print(f"rendered_chat={rendered!r}")
    print(f"input_ids={token_ids}")

    custom_model, report = load_pretrained_qwen(
        args.model_path,
        args.device,
        "float32",
        attn_implementation=args.attn_implementation,
    )
    oracle_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device).eval()

    with torch.inference_mode():
        custom_logits = custom_model(input_ids, use_cache=False).logits[:, -1, :]
        oracle_logits = oracle_model(input_ids, use_cache=False).logits[:, -1, :]

    difference = (custom_logits - oracle_logits).abs()
    max_absolute_error = float(difference.max().cpu())
    max_relative_error = float((difference / oracle_logits.abs().clamp_min(1e-8)).max().cpu())
    custom_values, custom_ids = torch.topk(custom_logits, k=10, dim=-1)
    oracle_values, oracle_ids = torch.topk(oracle_logits, k=10, dim=-1)
    custom_top_ids = tuple(int(value) for value in custom_ids[0].cpu().tolist())
    oracle_top_ids = tuple(int(value) for value in oracle_ids[0].cpu().tolist())

    print(f"checkpoint_tensors={report.tensor_count}")
    print(f"custom_attention_backend={args.attn_implementation}")
    print(f"expected_tied_missing={list(report.expected_tied_missing)}")
    print(f"max_absolute_error={max_absolute_error:.9g}")
    print(f"max_relative_error={max_relative_error:.9g}")
    print(f"custom_top10_ids={custom_top_ids}")
    print(f"oracle_top10_ids={oracle_top_ids}")
    print(f"custom_top10_logits={tuple(float(v) for v in custom_values[0].cpu().tolist())}")
    print(f"oracle_top10_logits={tuple(float(v) for v in oracle_values[0].cpu().tolist())}")
    print(f"custom_greedy_id={custom_top_ids[0]} oracle_greedy_id={oracle_top_ids[0]}")

    ids_match = custom_top_ids == oracle_top_ids
    greedy_match = custom_top_ids[0] == oracle_top_ids[0]
    error_ok = max_absolute_error <= args.tolerance
    print(f"parity: top10={ids_match} greedy={greedy_match} abs_error_ok={error_ok}")
    if not (ids_match and greedy_match and error_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
