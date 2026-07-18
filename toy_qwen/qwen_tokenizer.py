from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SUPPORTED_ROLES = {"system", "user", "assistant"}


def render_qwen_chat(messages: Sequence[dict[str, str]], add_generation_prompt: bool = True) -> str:
    if not messages:
        raise ValueError("messages must not be empty")
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in SUPPORTED_ROLES:
            raise ValueError(f"unsupported chat role {role!r} at position {index}; tool calling is not supported")
        if not isinstance(content, str):
            raise ValueError(f"message content at position {index} must be a string")

    parts: list[str] = []
    start = 0
    if messages[0]["role"] == "system":
        parts.append(f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n")
        start = 1
    else:
        parts.append(f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n")
    for message in messages[start:]:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


@dataclass
class QwenTokenizerAdapter:
    _tokenizer: Any
    eos_token_id: int
    pad_token_id: int | None

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "QwenTokenizerAdapter":
        model_dir = Path(model_dir)
        tokenizer_path = model_dir / "tokenizer.json"
        config_path = model_dir / "tokenizer_config.json"
        missing = [str(path) for path in (tokenizer_path, config_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing tokenizer files: {', '.join(missing)}")
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError("real tokenizer requires: pip install tokenizers==0.19.1") from error
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        eos_id = tokenizer.token_to_id(config["eos_token"])
        pad_id = tokenizer.token_to_id(config["pad_token"]) if config.get("pad_token") else None
        if eos_id is None:
            raise ValueError(f"eos token {config['eos_token']!r} is absent from tokenizer.json")
        return cls(tokenizer, eos_id, pad_id)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode_chat(self, messages: Sequence[dict[str, str]], add_generation_prompt: bool = True) -> tuple[str, list[int]]:
        rendered = render_qwen_chat(messages, add_generation_prompt)
        return rendered, self.encode(rendered)

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def token(self, token_id: int) -> str:
        token = self._tokenizer.id_to_token(token_id)
        if token is None:
            raise ValueError(f"invalid token id {token_id}")
        return token
