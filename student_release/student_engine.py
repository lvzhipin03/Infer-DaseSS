"""
Student submission entry point for the strict inference-engine track.

You must implement the model computation yourself: load tokenizer/config/raw
weights, then write your own prefill, greedy decode, KV cache, batching, and any
optimization strategy. The benchmark imports StudentEngine once and calls
generate() for fixed-batch workloads.

Advanced submissions may also implement serve_requests(requests, batch_size)
for the request-stream serving/scheduling suite. If absent, the benchmark
falls back to generate().

Do not call Hugging Face AutoModel/AutoModelForCausalLM, model.forward,
model(...), or model.generate(). Those APIs are intentionally outside the
student release boundary for this assignment.
"""

from __future__ import annotations


class StudentEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.local_files_only = local_files_only

        # Recommended starting point:
        # - use AutoTokenizer only for tokenization;
        # - use utils.load_weights.load_config_and_state_dict() to read config
        #   and raw safetensors weights;
        # - implement Qwen embedding/RMSNorm/QKV/RoPE/attention/MLP/LM head,
        #   prefill, greedy decode, and KV cache in this file or your helpers.
        raise NotImplementedError("Implement your strict manual inference engine here.")

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int = 1,
        suite_name: str | None = None,
    ) -> list[str]:
        """
        Return generated continuations in the same order as prompts.

        Hard requirements:
        - Return type must be list[str].
        - len(outputs) must equal len(prompts).
        - Outputs must correspond to the original prompt order.
        - Outputs must be continuations only; do not prepend the full prompt.
        - Do not read benchmark answer fields or hidden data.
        - Do not call external LLM/API services.
        - Do not call Hugging Face model.forward/model.generate or full
          inference frameworks such as vLLM, llama.cpp, or TGI.
        """
        raise NotImplementedError("Implement generate().")

    # Optional high-level serving interface for request-stream scheduling.
    # The benchmark passes batch_size=None for serving_schedule, so optimized
    # engines may choose their own active-batch policy.
    # def serve_requests(self, requests: list[dict], batch_size: int | None = None):
    #     return self.generate(
    #         [request["prompt"] for request in requests],
    #         max_new_tokens=max(int(request.get("max_new_tokens", 1)) for request in requests),
    #         batch_size=batch_size or len(requests),
    #         suite_name=None,
    #     )
