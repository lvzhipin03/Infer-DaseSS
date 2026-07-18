#!/usr/bin/env python3
from __future__ import annotations

import argparse


DEFAULT_MODEL_PATH = "/ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real Qwen2.5 weights with the project's own forward implementation")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default="中国的首都是哪里？")
    parser.add_argument("--system-prompt", default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--trace-shapes", action="store_true")
    return parser


def _format_cache_shapes(cache_shapes) -> str:
    return ", ".join(
        f"layer_{index}: K{key_shape} V{value_shape}"
        for index, (key_shape, value_shape) in enumerate(cache_shapes)
    )


def main() -> None:
    args = build_parser().parse_args()

    import torch

    from toy_qwen.generation import greedy_generate
    from toy_qwen.pretrained import load_pretrained_qwen
    from toy_qwen.qwen_tokenizer import QwenTokenizerAdapter

    tokenizer = QwenTokenizerAdapter.from_model_dir(args.model_path)
    rendered_chat, token_ids = tokenizer.encode_chat([
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ])
    print("=== INPUT ===")
    print(rendered_chat, end="")
    print(f"input_ids ({len(token_ids)}): {token_ids}")

    try:
        model, report = load_pretrained_qwen(args.model_path, args.device, args.dtype)
        config = model.config
        print("=== MODEL ===")
        print(
            f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
            f"intermediate={config.intermediate_size} q_heads={config.num_attention_heads} "
            f"kv_heads={config.num_key_value_heads} vocab={config.vocab_size} "
            f"dtype={args.dtype} device={args.device}"
        )
        print(
            f"checkpoint_tensors={report.tensor_count} "
            f"expected_tied_missing={list(report.expected_tied_missing)}"
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
        result = greedy_generate(
            model,
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            top_k=5,
        )
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory: stop other GPU jobs or retry with --device cpu --dtype bfloat16")
        raise

    print("=== SHAPES ===")
    print(f"prefill_logits={result.prefill_logits_shape}")
    print(f"first_cache={_format_cache_shapes(result.first_cache_shapes)}")
    print(f"last_cache={_format_cache_shapes(result.last_cache_shapes)}")
    if args.trace_shapes:
        for index, (key_shape, value_shape) in enumerate(result.last_cache_shapes):
            print(f"cache.layer_{index}.key={key_shape} cache.layer_{index}.value={value_shape}")

    print("=== GENERATION STEPS ===")
    for step in result.steps:
        candidates = []
        for token_id, logit in zip(step.top_ids, step.top_logits):
            candidates.append(
                f"id={token_id} token={tokenizer.token(token_id)!r} "
                f"text={tokenizer.decode([token_id], skip_special_tokens=False)!r} logit={logit:.6f}"
            )
        print(f"step={step.index} selected={step.token_id} top5=[{' | '.join(candidates)}]")

    print("=== RESULT ===")
    print(f"generated_ids={list(result.generated_ids)}")
    print(f"generated_text={tokenizer.decode(result.generated_ids)!r}")


if __name__ == "__main__":
    main()
