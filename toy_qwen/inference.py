from dataclasses import dataclass
import torch

from .modeling import QwenToyForCausalLM
from .tokenizer import ToyTokenizer


@dataclass(frozen=True)
class Prediction:
    token: str
    token_id: int
    logit: float
    runner_up_token: str
    runner_up_logit: float
    logits: torch.Tensor
    trace: dict[str, tuple[int, ...]]


def predict_next_token(text: str, model: QwenToyForCausalLM, tokenizer: ToyTokenizer,
                       trace_shapes: bool = False) -> Prediction:
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError("prompt must not be empty")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids, use_cache=model.config.use_cache, trace_shapes=trace_shapes)
    logits = output.logits[0, -1].detach().cpu()
    winner, runner_up = torch.argsort(logits, descending=True)[:2].tolist()
    return Prediction(tokenizer.token(winner), winner, logits[winner].item(),
                      tokenizer.token(runner_up), logits[runner_up].item(), logits,
                      output.trace or {})
