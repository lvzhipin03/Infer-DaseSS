import argparse

from toy_qwen.inference import predict_next_token
from toy_qwen.tokenizer import ToyTokenizer
from toy_qwen.weights import build_whiteboard_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whiteboard Qwen2-style toy model")
    parser.add_argument("--prompt", default="中国首都是")
    parser.add_argument("--trace-shapes", action="store_true")
    args = parser.parse_args()
    tokenizer = ToyTokenizer()
    model = build_whiteboard_model().eval()
    prediction = predict_next_token(args.prompt, model, tokenizer, args.trace_shapes)
    print(f"text: {args.prompt}")
    print(f"dense ids: {tokenizer.encode(args.prompt)}")
    print(f"legacy ids: {tokenizer.legacy_ids(args.prompt)}")
    if args.trace_shapes:
        for name, shape in sorted(prediction.trace.items()):
            print(f"shape {name}: {shape}")
    for token_id, logit in enumerate(prediction.logits.tolist()):
        print(f"logit {tokenizer.token(token_id)}: {logit:.6f}")
    print(f"margin over {prediction.runner_up_token}: {prediction.logit - prediction.runner_up_logit:.6f}")
    print(f"next token: {prediction.token}")


if __name__ == "__main__":
    main()
