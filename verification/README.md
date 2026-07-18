# Transformers 数值对照

这个目录只用于验证，不会被 `toy_qwen` 或生产 CLI 导入。它使用相同 chat template
和 input IDs，以 float32 eager attention 分别运行自研模型和 Transformers，并比较最后
位置的完整词表 logits、top-10 IDs 与 greedy token。

```bash
.venv-real/bin/pip install -r requirements-verify.txt
.venv-real/bin/python verification/compare_transformers.py \
  --model-path /ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct \
  --prompt 中国的首都是哪里？ --device cuda --tolerance 1e-3
```

脚本只在 top-10 顺序一致、greedy token 一致且最大绝对误差不超过阈值时返回 0。
