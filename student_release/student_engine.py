"""Correctness-first benchmark adapter for the handwritten toy_qwen engine."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from toy_qwen.generation import greedy_generate
from toy_qwen.pretrained import load_pretrained_qwen
from toy_qwen.qwen_tokenizer import QwenTokenizerAdapter


class StudentEngine:
    """Expose the repository's manual Qwen2 forward through the course API."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
        seed: int = 0,
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.local_files_only = bool(local_files_only)
        self.seed = int(seed)

        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        self.tokenizer = QwenTokenizerAdapter.from_model_dir(model_path)
        self.model, self.checkpoint_report = load_pretrained_qwen(
            model_path,
            device=self.device,
            dtype=dtype,
        )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int = 1,
        suite_name: str | None = None,
    ) -> list[str]:
        """Generate fixed-step greedy continuations in original prompt order."""
        del suite_name
        if not prompts:
            raise ValueError("prompts must not be empty")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("every prompt must be a string")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        outputs: list[str] = []
        for prompt in prompts:
            _, token_ids = self.tokenizer.encode_chat([
                {"role": "user", "content": prompt},
            ])
            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=self.device,
            )
            result = greedy_generate(
                self.model,
                input_ids,
                eos_token_id=None,
                max_new_tokens=max_new_tokens,
                top_k=5,
            )
            outputs.append(
                self.tokenizer.decode(
                    result.generated_ids,
                    skip_special_tokens=True,
                )
            )
        return outputs
