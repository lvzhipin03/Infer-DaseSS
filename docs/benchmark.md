# Infer-DaseSS 远程 2080 Ti Benchmark 工作流

本文记录从本地仓库向 SeetaCloud RTX 2080 Ti 服务器增量同步代码、运行标准 benchmark、监控进度和取回结果的固定流程。

## 1. 固定环境

### 本地

```text
项目目录: /home/lzp/workspace/Infer-DaseSS
SSH 别名: seetacloud-2080ti
```

连接命令：

```bash
ssh seetacloud-2080ti
```

### 远程

```text
GPU:      NVIDIA GeForce RTX 2080 Ti, 11264 MiB
代码:     /root/autodl-tmp/Infer-DaseSS
模型:     /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct
Python:   /root/autodl-tmp/Infer-DaseSS/.venv310/bin/python
缓存:     /root/autodl-tmp/cache
日志:     /root/autodl-tmp/Infer-DaseSS/benchmark_logs
结果:     /root/autodl-tmp/Infer-DaseSS/student_release/results
```

环境版本：

```text
Python       3.10.20
PyTorch      2.4.1+cu121
CUDA runtime 12.1
Transformers 4.43.1
tokenizers   0.19.1
safetensors  0.4.5
```

NVIDIA driver 为 550.90.07，可运行上述 CUDA 12.1 PyTorch。模型和 `.venv310` 均放在数据盘，后续阶段不需要重新上传或安装。

## 2. 每个优化阶段：同步代码

在本地项目根目录执行：

```bash
cd /home/lzp/workspace/Infer-DaseSS

rsync -az --info=progress2 \
  --include='/toy_qwen/***' \
  --include='/student_release/***' \
  --include='/configs/***' \
  --include='/verification/***' \
  --include='/tests/***' \
  --include='/README.md' \
  --include='/requirements-real.txt' \
  --include='/requirements-verify.txt' \
  --include='/real_qwen_inference.py' \
  --include='/whiteboard_llm_inference.py' \
  --exclude='/student_release/results/***' \
  --exclude='/student_release/.cache/***' \
  --exclude='*' \
  /home/lzp/workspace/Infer-DaseSS/ \
  seetacloud-2080ti:/root/autodl-tmp/Infer-DaseSS/
```

该命令只同步代码白名单，不上传 `.git`、本地 `.venv`、`tmp`、缓存、历史结果、zip 或 PDF。rsync 是增量传输，没有变化的文件不会重复上传。

注意：该命令不会删除远程存在但本地已删除的文件。若某阶段涉及删除或重命名模块，应先人工核对远程目录，不要直接增加 `--delete`。

## 3. 运行前验证

每次大改后建议先验证接口：

```bash
ssh seetacloud-2080ti '
  cd /root/autodl-tmp/Infer-DaseSS &&
  .venv310/bin/python student_release/scripts/validate_engine.py \
    --model /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct \
    --device cuda \
    --dtype float16 \
    --attn-implementation sdpa \
    --local-files-only
'
```

成功标志：

```text
Signature check passed.
Runtime interface check passed.
```

## 4. 后台运行标准正式 Benchmark

先为本阶段确定唯一名称。例如：

```text
stage0_2080ti_20260718
stage1_paged_kv_20260719
stage2_serving_20260720
```

以下示例使用 `stage1_paged_kv_20260719`。结果目录和日志名必须使用相同阶段名：

```bash
ssh seetacloud-2080ti '
  mkdir -p /root/autodl-tmp/cache \
           /root/autodl-tmp/Infer-DaseSS/benchmark_logs

  cd /root/autodl-tmp/Infer-DaseSS/student_release
  export INFERENCE_OPT_CACHE_ROOT=/root/autodl-tmp/cache

  nohup ../.venv310/bin/python -u scripts/run_inference_benchmark.py \
    --model /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct \
    --local-files-only \
    --device cuda \
    --dtype float16 \
    --attn-implementation sdpa \
    --timed-repeats 3 \
    --baseline-summary data/public_baseline_summary.json \
    --suite-isolation process \
    --worker-timeout-s 1800 \
    --output-dir results/stage1_paged_kv_20260719 \
    > /root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage1_paged_kv_20260719.log \
    2>&1 < /dev/null &

  echo BENCHMARK_PID=$!
'
```

这是标准测试口径：

- 六个公开 suite；
- 不使用 `--limit`；
- 使用 benchmark 默认 batch sizes 和 token budgets；
- 每个计时 batch 重复 3 次并取中位数；
- 每个 suite 使用独立 worker 进程；
- FP16、SDPA、greedy decode、local files only；
- 不添加本地 A6000 使用过的 11 GiB allocator 限制。

`nohup` 使 benchmark 在 SSH 断线后继续运行。

## 5. 查看后台进度

查看 GPU：

```bash
ssh seetacloud-2080ti nvidia-smi
```

实时查看阶段日志：

```bash
ssh seetacloud-2080ti \
  'tail -f /root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage1_paged_kv_20260719.log'
```

按 `Ctrl-C` 只会停止本地 `tail`，不会停止远程 benchmark。

查看 benchmark 进程：

```bash
ssh seetacloud-2080ti \
  "ps -eo pid,etime,stat,cmd | grep '[r]un_inference_benchmark.py'"
```

判断是否完成：

```bash
ssh seetacloud-2080ti \
  "grep -E 'FINAL SCORE|SERVER/GPU|RUN CONFIG' \
  /root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage1_paged_kv_20260719.log"
```

正常完成后日志包含 `FINAL SCORE`，且对应 benchmark 进程不再存在。

## 6. 结果文件

每个正式阶段应保留下列四个核心文件：

```text
student_release/results/<stage>/final_summary.json
student_release/results/<stage>/final_summary.txt
student_release/results/<stage>/student/summary.json
student_release/results/<stage>/student/results.csv
```

各 suite 的子目录还会保存独立 `summary.json` 和 `results.csv`，用于定位性能回退或 OOM。

## 7. 下载阶段结果到本地

在本地执行：

```bash
mkdir -p /home/lzp/workspace/Infer-DaseSS/tmp/remote_results

rsync -az --info=progress2 \
  seetacloud-2080ti:/root/autodl-tmp/Infer-DaseSS/student_release/results/stage1_paged_kv_20260719/ \
  /home/lzp/workspace/Infer-DaseSS/tmp/remote_results/stage1_paged_kv_20260719/

rsync -az \
  seetacloud-2080ti:/root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage1_paged_kv_20260719.log \
  /home/lzp/workspace/Infer-DaseSS/tmp/remote_results/
```

结果放在本地 `tmp`，不会进入 Git。

## 8. 当前 Stage 0 运行

当前后台正式测试：

```text
阶段: stage0_2080ti_20260718
PID:  8212（PID 仅对本次实例有效）
日志: /root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage0_2080ti_20260718.log
结果: /root/autodl-tmp/Infer-DaseSS/student_release/results/stage0_2080ti_20260718
```

查看当前进度：

```bash
ssh seetacloud-2080ti \
  'tail -50 /root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage0_2080ti_20260718.log'
```

## 9. 推荐的阶段循环

每轮优化固定执行：

1. 本地实现并运行单元测试；
2. 使用白名单 rsync 增量同步；
3. 在远程运行 `validate_engine.py`；
4. 为阶段选择新的、不可复用的名称；
5. 后台启动标准 benchmark；
6. 检查 `FINAL SCORE`、OOM、timeout 和六个 suite；
7. 将完整结果下载到 `tmp/remote_results/<stage>`；
8. 对比 Long、Decode、TTFT、Serving、Runtime 和峰值显存后再开始下一轮。

不要覆盖旧阶段目录，否则会失去可比较的实验记录。

## 10. 优化原则与当前瓶颈

课程材料 `Day2&3-推理优化.pdf` 强调两个基本事实：Prefill 通常更偏向 Compute-Bound，Decode 通常更偏向 Memory-Bound。对当前 Qwen2.5-0.5B 手写引擎而言，应先减少 Decode 的数据搬运、同步和临时分配，再考虑复杂调度或自研算子。

当前代码中已经确认的主要问题：

1. 每生成一个 token 都执行 `next_ids.detach().cpu().tolist()`，造成逐 token CUDA 同步和 GPU→CPU 传输；
2. 每层、每个 Decode step 都用 `torch.cat` 扩展 K/V，历史 KV 被反复复制；
3. `full_attention_mask` 每生成一个 token 也通过 `torch.cat` 扩展；
4. `_cache_shapes()` 在生产生成路径逐 token遍历所有层并构造 Python tuple；
5. 每层重复构造相同的 causal/padding attention mask；
6. 2080 Ti 上 SDPA 的 GQA 支持可能退化到 `repeat_kv`，把 2 个 KV heads 扩展为 14 个 heads，增加临时显存和带宽；
7. 不同长度 prompt 左填充到同一宽度，可能产生大量无效 Prefill；
8. 尚未实现 `serve_requests()`，Serving suite 只能使用 `generate()` fallback；
9. 尚未实现共享前缀缓存、continuous batching、chunked prefill 或分页 KV；
10. 当前结构尚不适合 CUDA Graph，因为 KV、mask 和输出张量地址在 Decode 中不断变化。

Stage 0 的真实 2080 Ti CSV 进一步表明，当前首要问题不是显存容量，而是变长 batch 的正确性：

```text
Decode batch=1: 多数样本可生成接近 128 个有效 tokens
Decode batch=2/4: 大量失败样本只有 16 个有效 tokens
Serving batch=12: 9/12 请求只有 12 个有效 tokens
Cache Stress: 大量输出长度退化到预算的约 1/8
全局峰值显存: 约 2.5GB，远低于 11GB
OOM / timeout: 0 / 0
```

因此，优化必须先恢复 batched generation 正确性，再讨论 batch TPS。任何只提高速度但维持低 valid 的改动都不能 accepted。

所有优化都必须遵守：

- 不调用 Hugging Face model forward 或 `model.generate()`；
- greedy 输出和当前正确实现保持一致；
- 不针对 suite 名、公开样本、关键词、case id 或数据文件做特判；
- 每个阶段先运行单元测试和真实接口验证，再运行标准 benchmark；
- 不在同一阶段混入过多互不相关的改动；
- 发现正确率、有效率或稳定性下降时，先修复再进入下一阶段。

## 11. 阶段 0：冻结 2080 Ti 基线

### 目标

保留当前未优化实现在真实 2080 Ti 11GB 上的完整标准结果，作为后续所有阶段的比较基准。

### 必须记录

- Git commit 或工作区代码指纹；
- Python、PyTorch、CUDA、GPU 和驱动版本；
- 最终总分和五个评分项；
- 六个 suite 的 runtime、valid、TPS/latency；
- Decode batch=1/2/4 的独立 TPS；
- TTFT 各长度 bucket 的 avg 和 p95；
- OOM、timeout 和失败样本原因；
- 峰值 allocated/reserved 显存；
- 完整 `final_summary.json`、日志和 CSV。

### 当前 Stage 0 重点

已观察到的部分指标：

```text
Long Context: valid=0.933, tps≈30.1
Decode:       best batch=4, best_tps≈51.5, valid≈0.583
TTFT:         avg≈0.126s, valid=1.000
Serving:      tps≈39.5, valid≈0.250
```

逐条 CSV 已确认低 valid 主要伴随有效生成长度异常缩短，不是 OOM、timeout 或 Python/CUDA exception。问题集中在 batch>1，而 Long Context 和 TTFT 的 batch=1 路径正常。

## 11.1 阶段 0.5：修复变长 Batch 与左 Padding 正确性

### 优先级

最高。该阶段必须在所有性能优化之前完成。

### 已知现象

- 单请求或无 padding 路径基本正常；
- batch=2/4/12 时大量输出有效 token 数异常缩短；
- runtime success 为 1.000，错误记录中没有 OOM 或异常；
- Long Context batch=1 和 TTFT batch=1 正常；
- 问题与变长 batch、左 padding、SDPA backend 的组合高度相关。

### 第一怀疑点：无效 Query 的 NaN 传播

当前 Attention 使用：

```python
output = output * query_is_valid[:, None, :, None]
```

若 SDPA/math backend 对 fully-masked padding query 返回 NaN，则 `NaN * 0` 仍为 NaN。NaN 会进入 residual，随后污染下一层 Q/K/V。2080 Ti 使用的 Turing SDPA backend 与 A6000 不同，因此同一代码可能只在 2080 Ti 暴露。

### 排查步骤

1. 构造长度明显不同的 2/4 请求 batch；
2. 在每层记录 input hidden、normalized hidden、Q/K/V、SDPA output、masked output、layer output 和 final logits 是否存在 NaN/Inf；
3. 对比 batch=1 逐条 SDPA、batch=2/4 左 padding SDPA、batch=2/4 eager、batch=2/4 等长 prompt SDPA；
4. 记录 PyTorch 实际选择的 SDPA backend；
5. 确认 `enable_gqa=True` 是成功执行还是进入 `repeat_kv` fallback；
6. 对比 batched 最后位置完整 logits、top-k 和 greedy token；
7. 检查特殊 token，确认 12/16 tokens 是特殊 token 污染还是其他截断。

### 首选修复

使用选择操作真正清零无效 query，而不是乘法：

```python
output = torch.where(
    query_is_valid[:, None, :, None],
    output,
    torch.zeros((), dtype=output.dtype, device=output.device),
)
```

必要时同时：

- 在进入下一层前通过 `torch.where` 清零 padding hidden states；
- 保证 padding K/V 是有限值；
- 避免把 fully-masked query 交给不稳定 backend；
- 为 Prefill padding query 和 Decode valid query 使用不同路径。

不能仅通过全局 `nan_to_num` 吞掉异常。应先定位来源，再在语义上无效的 padding 位置清零。

### 必须新增的回归测试

对同一组 prompts 验证：

```text
batch=1 逐条输出 == batch=2 输出 == batch=4 输出
```

覆盖相同/极端不同 prompt 长度、左 padding、`max_new_tokens=1/16/128`、eager/SDPA、CPU/CUDA、真实 Qwen 权重，以及每步 cache length、position IDs 和 mask。

### Accepted 门槛

- batch=1/2/4 greedy token IDs 一致；
- 2080 Ti Decode valid 恢复到接近 1.000；
- Serving/Mixed/Cache Stress 不再大量退化为 12/16/32/64 tokens；
- Long partial/exact 不下降；
- runtime success 保持 1.000；
- 不通过减小 batch 或逐条执行来掩盖 correctness bug。

## 12. 阶段 1：消除逐 Token CPU/GPU 同步

### 目标

减少 Decode loop 的 Python 开销和 CUDA synchronization，不改变模型数学路径。

### 建议改动

1. 在 GPU 上预分配生成结果：

   ```text
   generated_ids: [batch_size, max_new_tokens]
   ```

2. 每步执行 `generated_ids[:, step] = next_ids`；
3. 整个 Decode 完成后只执行一次 `.cpu()`；
4. 生产 benchmark 路径不再每步调用 `_cache_shapes()`；
5. trace、top-k、逐 token selected logit 只保留在教学/调试入口；
6. 避免在 Decode hot path 中反复创建 Python list、tuple 和 dataclass。

### 验证

- greedy token IDs 与阶段 0 完全一致；
- `tests/test_generation.py` 和 batch generation 测试通过；
- Decode batch=1/2/4 TPS 不下降；
- Long partial/exact 不下降；
- 峰值显存不增加。

### 风险

生成结果长度、顺序或 continuation 解码可能因张量切片处理错误而改变。必须覆盖 batch>1 和不同 prompt 长度。

## 13. 阶段 2：预分配连续 KV Cache

### 目标

移除每层、每步扩展 K/V 的 `torch.cat`，把 Decode 的历史 KV 复制改为原地写入。

Stage 0 已证明当前峰值显存约 2.5GB，因此该阶段的首要目标是减少历史 KV 复制、allocator 开销和显存带宽消耗，提高 Decode TPS；不是解决当前 OOM。预分配仍会为 CUDA Graph、continuous batching 和分页缓存提供基础。

### 推荐数据结构

每层预分配：

```text
K: [batch, num_kv_heads, capacity, head_dim]
V: [batch, num_kv_heads, capacity, head_dim]
```

记录逻辑状态：

```text
current_length
prompt_lengths
capacity
batch_size
```

### 写入方式

Prefill：

```text
cache[:, :, :prompt_width, :] = prefill_kv
```

Decode：

```text
cache[:, :, current_position, :] = new_kv
attention 读取 cache[:, :, :current_position + 1, :]
```

### 同时预分配

- 完整 attention mask buffer；
- position IDs 或 position counter；
- generated token buffer；
- 如有必要，预分配 layer presents 容器，避免每步构造 tuple。

### Capacity 策略

第一版按当前调用的最大需求分配：

```text
capacity = padded_prompt_width + max_new_tokens
```

必须在分配前验证不超过 `max_position_embeddings`。不要在实际 2080 Ti 上按模型最大 32768 无条件预分配所有请求，否则会浪费显存。

### 验证

- 与旧 `torch.cat` cache 做逐层 K/V 数值对齐；
- 单条和 batch greedy token 完全一致；
- cache length、position IDs 和左 padding 语义正确；
- Decode TPS 提升；
- Cache Stress 的显存增长与 OOM 数量下降；
- 2080 Ti 峰值显存不超过可用范围。

### 风险

左 padding batch 中，物理 cache width 与每行逻辑 token position 不完全相同。RoPE 使用逻辑 position，Attention mask 使用物理 cache slot，两者不能混淆。

## 14. 阶段 3：预分配 Mask 与共享 Attention Mask

### 目标

移除 Decode 中 `full_attention_mask = torch.cat(...)`，并避免 24 层重复生成相同 mask。

### 建议改动

1. 在生成开始时创建 `[batch, capacity]` mask；
2. Prefill 区域写入原始 padding mask；
3. 每个 Decode step 原地把对应新 slot 写为 1；
4. 每次 model forward 只构造一次 SDPA mask，再传给所有 decoder layers；
5. `query_length=1` 的 Decode 走轻量专用路径；
6. 无 padding 的单请求 Decode 不创建完整 causal mask；
7. 如果 backend 支持，Prefill 无 padding时优先使用 `is_causal=True`。

### 验证

- eager 与 SDPA logits 继续对齐；
- 左 padding batch token 完全一致；
- TTFT 和 Decode latency 不下降；
- mask 临时显存显著减少。

## 15. 阶段 4：2080 Ti 兼容的 GQA Decode Attention

### 目标

在不物化重复 K/V 的情况下计算 Qwen GQA，降低显存带宽和临时显存。

模型结构：

```text
query heads = 14
KV heads    = 2
groups      = 7
head dim    = 64
```

### 问题

当前 SDPA 在 backend 不支持 `enable_gqa=True` 时会退化到 `repeat_kv(key, 7)` 和 `repeat_kv(value, 7)`。在 2080 Ti/PyTorch 2.4 上应通过 profiling 确认实际 backend 和是否发生物化复制。

### 第一版实现方向

Decode Query reshape：

```text
[B, 14, 1, D]
→ [B, 2, 7, 1, D]
```

K/V 保持：

```text
[B, 2, sequence, D]
```

在 KV head 维度内为 7 个 Query groups 计算 attention，避免把完整历史 K/V 复制 7 份。

### 路径划分

- Prefill：继续优先使用 PyTorch SDPA；
- Decode：使用专门的 grouped-query attention；
- eager：继续作为数值参考；
- 所有路径共享 RoPE、mask 和 KV Cache。

### 验证

- 完整 logits 最大误差在既定 tolerance 内；
- top-10 和 greedy token 一致；
- batch=1/2/4 全覆盖；
- Long、Cache Stress 不 OOM；
- 比较 repeat K/V 路径的显存和 TPS。

### 后续选择

若 PyTorch 算子仍有明显开销，再评估 Triton/自定义 CUDA online softmax。不要在没有 profiling 证据时先写复杂 kernel。

## 16. 阶段 5：Prompt 长度分桶

### 目标

减少不同长度 prompt 被左填充到最长宽度造成的无效 Prefill。

### 基础桶

```text
0–256
257–512
513–1024
1025–2048
2049–4096
4097+
```

### 执行流程

1. tokenizer 编码后保留原始输入下标；
2. 按 token 长度排序；
3. 相近长度组成 batch；
4. 生成完成后恢复原始顺序；
5. 保证返回数量与 prompt 顺序完全一致。

### 动态拆桶条件

固定 batch benchmark 不应无条件拆成小 batch。可根据 padding waste 判断：

```text
padding_waste = 1 - sum(real_lengths) / (batch_size * max_length)
```

只有浪费超过阈值时才拆桶，并对阈值进行实测。

### 验证

- 输出顺序严格保持；
- Mixed Serving TPS 和峰值显存改善；
- Decode 固定 batch 的吞吐不因过度拆分下降；
- TTFT 各长度 bucket 单独比较。

## 17. 阶段 6：实现基础 `serve_requests()`

### 目标

不再使用 benchmark 的 `generate()` fallback，为 Serving suite 提供通用请求调度。

### 第一版功能

- 解析每个请求的 prompt、`max_new_tokens`、arrival、priority 和 group；
- 按 priority/arrival 排序；
- 按 prompt 长度和 `max_new_tokens` 分组；
- 根据 token budget 和预计 KV 显存决定实际 batch；
- 为不同 `max_new_tokens` 的请求正确截断结果；
- 恢复 request_id 和输入顺序；
- 不读取答案、关键词或 case id 来决定输出。

### 显存预算

优先进行保守的显存估算，而不是把 OOM 当作正常控制流。保留现有 OOM 二分作为最后恢复手段。

### 返回格式

先选择 benchmark 最容易验证的格式，覆盖：

- 返回 `list[str]`；或
- 返回包含 `request_id` 和 continuation 的通用结构。

必须通过公开脚本对返回格式的解析测试。

### 验证

- Serving `iface` 从 `generate` 变为 `serve_requests`；
- runtime success 不下降；
- 每请求生成预算正确；
- p95 latency 和 TPS 改善；
- 无请求丢失、重复或乱序。

## 18. 阶段 7：共享前缀 KV Cache

### 目标

对完全相同的 token 前缀复用 Prefill KV，提升 Serving 和重复系统提示场景。

### 实施顺序

1. batch 内计算最长公共 token 前缀；
2. 公共前缀只 Prefill 一次；
3. 为各请求复制或共享只读前缀 cache 视图；
4. 再扩展到跨调用缓存；
5. 使用 block 对齐前缀；
6. 增加引用计数和 LRU 淘汰。

### Cache Key

必须基于真实 token IDs、模型/config 标识、dtype 和必要的 chat template 信息。禁止使用 prompt 关键词、suite 名、case id 或公开数据字段。

### 显存控制

- 设置缓存最大字节数；
- 只缓存完整 block；
- 引用计数为 0 的 block 才允许淘汰；
- 从叶子/最长后缀开始 LRU 回收；
- OOM 前主动回收。

### 验证

- cache hit/miss 的输出完全一致；
- prefix_shared 指标提升；
- Serving TPS/p95 改善；
- 不因缓存常驻导致 11GB OOM。

## 19. 阶段 8：Chunked Prefill 与 Continuous Batching

### 目标

降低长 Prefill 对活跃 Decode 的阻塞，并把等待请求动态加入 batch。

### 调度规则

每个 iteration：

1. 先为所有 active decode 分配 1 token budget；
2. 从 `max_num_batched_tokens` 扣除 decode 数量；
3. 用剩余 budget 接纳 waiting prefill；
4. 超长 prompt 按 `chunk_size` 切分；
5. 每个 chunk 写入同一请求的预分配/分页 KV；
6. 请求 Prefill 完成后进入 active decode；
7. 完成请求及时释放或转为可淘汰 cache。

初始候选：

```text
chunk_size = 256 或 512
max_num_batched_tokens 根据 2080 Ti profiling 选择
```

### Attention 正确性

当前 chunk 的 Query 必须能看到：

- 所有已完成 chunk 的 KV；
- 当前 chunk 中不晚于自身位置的 KV。

块间部分是 cross-attention，块内部分必须保留 causal mask。输出必须与一次完整 Prefill 数学等价。

### 验证

- chunked 与非 chunked 完整 logits/greedy token 对齐；
- Serving p95 改善；
- active decode 不因长 Prefill 长时间停顿；
- TTFT 可能有权衡，需同时报告吞吐和延迟。

## 20. 阶段 9：Paged KV Block Pool

### 目标

支持动态请求加入/退出、减少碎片、共享前缀 block，并为更完整的 continuous batching 提供存储基础。

### 演进顺序

```text
torch.cat KV
→ 预分配连续 KV
→ 每请求独立连续 KV
→ 固定大小 block pool
→ block table
→ prefix block 引用计数
→ LRU 回收
```

### 基础结构

```text
block_size: 16 或 32 tokens
free_block_pool
request_id -> block_table
physical_block -> ref_count
physical_block -> last_access
prefix_hash -> physical blocks
```

### 注意

仅有 block table 不等于高性能 PagedAttention。如果每步仍把离散 block `torch.cat` 成连续 K/V，再交给 SDPA，主要复制开销仍然存在。真正高效需要能直接读取 block table 的 attention kernel。

因此应先比较：

- 连续预分配 cache 是否已经满足公开 benchmark；
- 分页管理带来的调度/共享收益；
- 自研 paged attention kernel 的开发成本。

## 21. 阶段 10：算子融合、`torch.compile` 与 CUDA Graph

### 前置条件

必须先完成静态/预分配 KV、mask 和输出 buffer，使 Decode step 的主要张量地址保持稳定。

### 优化候选

1. 合并 MLP `gate_proj` 与 `up_proj` 为一次大 GEMM，再切分输出；
2. 融合 SiLU 和逐元素乘法；
3. 融合 residual add + RMSNorm；
4. 预计算完整 RoPE cos/sin table，通过 position 索引读取；
5. 尝试 `torch.compile` 编译 Decode step；
6. 对固定 batch/capacity 的 Decode 捕获 CUDA Graph；
7. 缓存不同 batch size 的 graph，例如 1/2/4；
8. profiling 后再决定是否编写 Triton kernel。

### 验证

- 首次编译/warmup 不进入正式计时；
- 不因动态 shape 反复 recompile；
- graph replay 使用正确 position/cache slot；
- batch size 或 capacity 变化时安全 fallback；
- 数值和 greedy token 不变。

## 22. 阶段 11：量化评估

Decode 是 memory-bound，权重量化理论上能减少每 token 的模型权重读取，但这是高风险后期项目。

### 可评估方向

- INT8 weight-only；
- INT4/AWQ 风格 groupwise weight-only；
- KV Cache INT8/FP8（2080 Ti 不具备 H100 FP8 Tensor Core，应谨慎）；
- LM head 单独保持 FP16，降低 greedy token 变化风险。

### 进入条件

- FP16 路径已完成结构优化；
- 有可靠量化 GEMM kernel，而不是每层即时反量化回 FP16；
- Long Context 正确率和 greedy 一致性有清晰门槛；
- 评分收益足以补偿正确率风险。

### 不建议当前优先做

- H100 专属 FP8 Transformer Engine；
- Hopper TMA；
- FlashAttention-3；
- 单卡场景的 Prefill/Decode 多机分离；
- 未经 profiling 直接自研完整 FlashAttention/PagedAttention kernel。

## 23. 每阶段统一验收表

每个优化阶段建立一条记录：

| 字段 | 内容 |
| --- | --- |
| Stage | 唯一阶段名 |
| 目标 | 本阶段只解决的核心问题 |
| Commit/代码指纹 | 可复现版本 |
| Correctness | Long partial/exact、oracle 误差、greedy 一致性 |
| Decode | batch 1/2/4 TPS 与 scaling |
| TTFT | bucket avg、bucket p95、总体 avg/p95 |
| Serving | iface、TPS、p95、prefix shared |
| Runtime | success、valid、OOM、timeout |
| Memory | peak allocated/reserved、cache growth |
| Score | 总分及五项得分 |
| 结论 | accepted / rejected 及原因 |

### Accepted 门槛

- runtime success 不下降；
- Long partial/exact 不下降或变化在可解释容差内；
- valid 不下降；
- 目标指标有稳定收益，而不是单次噪声；
- 不通过公开数据特判获得收益；
- 完整结果文件已保存并下载。

## 24. 推荐实施顺序

```text
Stage 0  冻结 2080 Ti 基线
Stage 0.5 修复变长 batch / 左 padding / SDPA 正确性
Stage 1  去除逐 token CPU/GPU 同步和诊断开销
Stage 2  预分配连续 KV Cache
Stage 3  预分配/共享 attention mask
Stage 4  2080 Ti 兼容 GQA Decode Attention
Stage 5  Prompt 长度分桶
Stage 6  基础 serve_requests 调度
Stage 7  共享前缀 KV Cache
Stage 8  Chunked Prefill + Continuous Batching
Stage 9  Paged KV Block Pool
Stage 10 torch.compile + CUDA Graph + 算子融合
Stage 11 评估 INT8/INT4 量化
```

第一轮必须只实施 Stage 0.5，并在真实 2080 Ti 上证明 batch=1/2/4 输出一致。Stage 0.5 中同时查明并固定 SDPA/GQA backend 行为；随后 Stage 1 去除逐 token 同步。验证正确性与收益后再单独实施 Stage 2。不要把“正确性修复、去同步、预分配 KV、GQA kernel 和调度”合并成一个阶段，否则发生回退时难以定位原因。

## 25. 2026-07-19 当前最佳基线

已完成并保留的组合：

```text
Stage 0.5  left-padding fully-masked Query correctness
Stage 1    移除逐 token 同步和诊断开销
Stage 2    Static KV Cache
Stage 3A   预分配 generation attention mask
Stage 3B   跨 24 层共享 allowed attention mask
Decode     past_length>0 时跳过 padding-query torch.where
Fusion     MLP gate_proj + up_proj
```

已撤回：

```text
Static Cache validation signature  无收益
token-major direct K/V write       RoPE clone/copy 导致变慢
Fused QKV                          batch=4 TPS 回退 20.4%
Native F.rms_norm                  真实 Decode valid=0.000
```

标准六套件：

| 项目 | 当前结果 |
| --- | ---: |
| Long | 28.50 / 30 |
| Decode | 23.17 / 25，172.5 TPS |
| TTFT | 16.16 / 20，0.125s avg，0.409s p95 |
| Serving | 9.32 / 15，117.7 TPS，8.867s p95 |
| Runtime | 10.00 / 10 |
| **总分** | **87.16 / 100** |

相比 Stage 0.5 的 86.49 分，总分提升 0.67；Decode、Serving、Mixed 和 Cache TPS 分别提升 15.8%、11.9%、20.0% 和 15.0%。

当前结果：

```text
/root/autodl-tmp/Infer-DaseSS/student_release/results/stage_best_full_20260719
/root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage_best_full_20260719.log
```
