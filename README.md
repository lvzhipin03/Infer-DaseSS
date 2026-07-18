# Toy Qwen2.5 白板推理器

这是一个只依赖 PyTorch 的 Qwen2 风格最小 causal LM。它使用手写白板权重在 CPU
上执行完整的 Embedding、RMSNorm、RoPE、causal GQA、SwiGLU、残差、final
RMSNorm 和 LM Head 链路。

## 运行白板模型

```bash
python3 whiteboard_llm_inference.py --prompt 中国首都是 --trace-shapes
```

最后一行应为：

```text
next token: 北
```

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 运行真实 Qwen2.5-0.5B-Instruct 权重

生产推理仍使用本项目自己的 Qwen2 forward，不依赖 Transformers。服务器使用
Python 3.11，并让虚拟环境复用服务器已有的 CUDA PyTorch：

```bash
/root/.pyenv/shims/python3.11 -m venv --system-site-packages .venv-real
.venv-real/bin/pip install -r requirements-real.txt
.venv-real/bin/python real_qwen_inference.py \
  --model-path /ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct \
  --prompt 中国的首都是哪里？ --max-new-tokens 16 --trace-shapes
```

输出包含配置、checkpoint 张量数量、输入 token IDs、prefill logits shape、每层 KV
Cache shape、逐步 top-5 token/logit，以及最终文本。`requirements-verify.txt` 中的
Transformers 只用于独立数值对照，不进入生产推理路径。

本地修改后同步到服务器：

```bash
rsync -avz --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' \
  /mnt/d/vdesktop/infer-dasess/ dase314-server:/ai/projects/Infer-DaseSS/
```

## Benchmark 第二阶段

`student_release/student_engine.py` 直接复用仓库根目录的 `toy_qwen`，不是一套独立
模型实现。当前实现保留可解释的 eager attention 作为数值参考，并增加 PyTorch SDPA
运行后端；多个不同长度的 prompt 会按 `batch_size` 左填充，使用显式 attention mask
和每行逻辑 position IDs 完成批量 prefill、KV Cache decode。prefill 只把最后位置送入
LM Head，生成仍采用固定步数 greedy decode 并忽略 EOS，符合公开 benchmark 契约。

完整公开 benchmark 的阶段二实测结果会在完成服务器正确性门禁后记录在本节。

服务器运行时检查：

```bash
cd /ai/projects/Infer-DaseSS/student_release
source use_data_cache.sh
../.venv-real/bin/python scripts/validate_engine.py \
  --model /ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda --dtype float16 --local-files-only
```

全 suite 单样本 smoke：

```bash
../.venv-real/bin/python -u scripts/run_inference_benchmark.py \
  --model /ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct \
  --local-files-only --device cuda --dtype float16 \
  --attn-implementation sdpa --limit 1 \
  --decode-batch-sizes 1 --ttft-batch-sizes 1 \
  --serving-fallback-batch-size 1 --mixed-batch-sizes 1 \
  --cache-stress-batch-sizes 1 --max-new-tokens-cache-stress 32 \
  --baseline-summary data/public_baseline_summary.json \
  --allow-stale-baseline --suite-isolation process \
  --worker-timeout-s 1800 --output-dir results/smoke_test
```

## 白板配置

默认配置为 9 个字符、hidden size 4、SwiGLU intermediate size 8、1 层、1 个
query head 和 1 个 KV head。`1Q/1KV` 是通用 GQA 的退化形式；同一个实现另有
`14Q/2KV` 参数测试。

白板资料用 `[in,out]` 记录线性矩阵，加载器会转置为 PyTorch 的
`nn.Linear.weight=[out,in]`。LM Head 扩成完整 `[9,4]`：输入字符对应行置零，
“北、京、上、海”保留参考权重。

## 张量流

```text
input_ids [B,T]
→ embedding [B,T,H]
→ Q [B,Nq,T,D], K/V [B,Nkv,T,D]
→ attention scores [B,Nq,T,S]
→ decoder hidden [B,T,H]
→ logits [B,T,V]
```

KV Cache 保存每层 `[B,Nkv,cached_length,D]` 的 K/V。测试验证完整输入与
prefill“`中国首都`”再 cached decode“`是`”的最终 logits 一致。

## 扩展为完整 Qwen2.5-0.5B

`configs/qwen2_5_0_5b.json` 保留官方结构参数。模块路径使用 Qwen/Hugging Face
风格，例如 `model.layers.0.self_attn.q_proj.weight`。QKV bias、tied embedding、
层数、head 数和维度均由配置控制。

真实权重适配器当前支持单个 `model.safetensors`、Qwen tokenizer.json、FP16/BF16/float32
以及 greedy KV Cache 生成。量化、safetensors 分片和滑动窗口仍未实现；它们可作为
独立 adapter 添加，不需要修改当前 forward 数学链路。
