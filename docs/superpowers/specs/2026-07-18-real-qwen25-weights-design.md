# Qwen2.5-0.5B 真实权重加载设计

## 目标

让项目自研的 `QwenToyForCausalLM` 从服务器本地 ModelScope 模型目录加载
Qwen2.5-0.5B-Instruct 的真实配置、BPE tokenizer 和 290 个 BF16 safetensors
权重，先独立完成 chat prompt 的 CUDA greedy 推理，再以 Hugging Face
`Qwen2ForCausalLM` 作为可选 oracle 做数值对照。

正式推理不得调用 Transformers 模型实现。现有白板权重与单元测试保留，作为教学和
快速回归模式。

## 模型与运行环境

服务器模型路径：

```text
/ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct
```

该路径是符号链接，包含：

```text
config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
model.safetensors
```

checkpoint 约 988 MB，共 290 个 BF16 tensor。结构为 hidden size 896、FFN
4864、24 层、14 query heads、2 KV heads、head dimension 64、词表 151936、
`rope_theta=1000000`、Q/K/V 有 bias、embedding 与 LM Head tied。

服务器为 Python 3.8.10、PyTorch 1.13.1+cu117，GPU 是 A800 80GB。正式新增依赖：

```text
safetensors==0.4.5
tokenizers==0.19.1
```

`transformers==4.43.1` 仅属于可选验证依赖。

## 生产架构

新增 `toy_qwen/pretrained.py`，负责：

1. 检查模型目录所需文件。
2. 从官方 `config.json` 创建 `QwenToyConfig`。
3. 根据 CLI dtype 构建完整 `QwenToyForCausalLM`。
4. 通过 `safetensors.torch.load_file` 读取 checkpoint。
5. 在复制前校验全部名称和 shape。
6. 允许 checkpoint 唯一缺少 tied alias `lm_head.weight`。
7. 拒绝任何其他 missing/unexpected/mismatched key。
8. 加载完成后再次保证 `lm_head.weight is model.embed_tokens.weight`。
9. 移动到目标 device，调用 `eval()`。

生产模型仍执行项目自己的 RMSNorm、Qwen RoPE、GQA、causal mask、SwiGLU、
residual、final RMSNorm、LM Head 和 KV Cache。

新增 `toy_qwen/qwen_tokenizer.py`，使用 `tokenizers.Tokenizer.from_file` 加载真实
BPE。首版支持 system、user、assistant 三种角色和 `add_generation_prompt`，模板严格
遵循模型 `tokenizer_config.json` 的无 tools 分支。默认 system prompt 为：

```text
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

新增 `toy_qwen/generation.py`，实现 prompt prefill、逐 token cached decode、greedy
argmax、EOS 停止和最大新 token 限制。它接收 tokenizer/model 接口，不包含具体模型
路径或测试 prompt。

新增 `real_qwen_inference.py` CLI，参数包括 model path、prompt、system prompt、
device、dtype、max new tokens 和 trace shapes。默认测试 prompt 是
“中国的首都是哪里？”。

## 输出和日志

真实模型默认打印：

```text
配置摘要：H、I、layers、Q heads、KV heads、dtype
checkpoint tensor 数量
missing/unexpected/mismatched key 报告
chat 文本和 token IDs
prefill logits shape
第 0 层及最后一层 KV cache shape
每步 top-5 token、ID 和 logit
生成 token IDs
最终生成文本
```

不打印完整 151936 维 logits，也不保存 hidden states，避免无意义输出和显存占用。

## 正式推理依赖边界

生产代码不得 import `transformers`。`torch + safetensors + tokenizers` 足够完成真实
推理。缺少可选依赖时，模块必须给出包含安装命令的错误，而不是在包导入阶段让白板模式
失效。

`verification/compare_transformers.py` 是唯一允许 import Transformers 的文件。它
使用同一模型目录、同一 chat input IDs、float32 和 eager attention，对比：

- 最后位置完整 logits 的最大绝对/相对误差；
- top-10 token IDs；
- greedy 首 token；
- 若单步对齐，再比较短序列 greedy token IDs。

## 错误处理

- 缺失 config/tokenizer/checkpoint 时列出缺失文件。
- 不支持的 dtype/device 给出明确错误。
- 请求 CUDA 但不可用时在构建模型前失败。
- checkpoint tensor 名称或 shape 不匹配时，不允许部分加载。
- chat messages 为空、角色不支持或顺序非法时失败。
- prompt 加生成长度超过 32768 时失败。
- CUDA OOM 时保留原始异常并提示改用 BF16、减少长度或 CPU。
- 生成遇到 `eos_token_id=151645` 时停止。

## 测试策略

本地测试继续只要求 PyTorch。对真实依赖的单元测试在依赖缺失时 skip，但 chat template
字符串渲染、配置解析、key/shape 校验逻辑必须可用小型 fake state dict 测试。

服务器验收严格按以下顺序：

1. 创建 `--system-site-packages` venv，复用 CUDA PyTorch。
2. 安装 safetensors/tokenizers 正式依赖。
3. 运行现有全部白板测试。
4. 加载真实 tokenizer，检查 chat prompt 和 token IDs。
5. 加载 290 个真实 tensor，检查 tied LM Head。
6. 用自研模型 BF16 CUDA 单步 forward。
7. 用自研 KV Cache greedy 生成，输出非空文本和关键 shape。
8. 安装可选 Transformers 4.43.1。
9. 使用 float32 eager 模式执行 oracle 对照。

真实模式的核心验收是自研模型能够独立生成；语义答案不通过硬编码断言。数值正确性以
oracle logits/top-k/token IDs 对照为准。

## 本地与服务器工作流

所有源码先在本地 `/mnt/d/vDesktop/Infer-DaseSS` 修改和测试，再通过 rsync 同步到：

```text
/ai/projects/Infer-DaseSS
```

同步排除 `.git/`、`.venv/` 和 `__pycache__/`。服务器只用于安装真实依赖、加载本地
模型文件和执行 GPU 验收，不直接编辑源码。

## 非目标

本次不实现 sampling、beam search、量化、训练、tool calling、批量 padding、长上下文
优化、FlashAttention、分片 checkpoint 或模型下载。单文件 0.5B checkpoint 和短 chat
prompt 是本阶段唯一真实模型范围。
