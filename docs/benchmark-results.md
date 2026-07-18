# Benchmark 实验记录

本文按版本追加实验结果，不覆盖历史。根目录 [README](../README.md) 负责项目架构与运行方法，
本文只记录可复现实验环境、分数、运行情况和精简分析。

## Phase 2：Batched SDPA（accepted）

### 版本

| 项目 | 值 |
| --- | --- |
| 实验日期 | 2026-07-18 |
| 实现版本 | Git commit `5af72a7e9c2d7cc72d0373612e3611e3aca4ffb3` |
| Benchmark 包 | `student_release(1).zip` |
| Benchmark SHA-256 | `c88840a4f36e5d7776496fa8677067f36c60850fa94caad67b4776f60e0a4722` |
| Score profile | `vllm_reference_teacher_v7_component_tiered` |
| 模型 | Qwen2.5-0.5B-Instruct，服务器外部 checkpoint |

### 系统与服务器规格

| 项目 | 值 |
| --- | --- |
| OS | Ubuntu 20.04，Linux `5.4.0-216-generic` |
| CPU | 64 vCPU，Intel Xeon Gold 6326 @ 2.90GHz |
| 内存 | 128 GiB |
| GPU | 1 × NVIDIA A800-SXM4-80GB（81920 MiB） |
| NVIDIA driver | 570.211.01 |
| Python | 3.11.10 |
| PyTorch / CUDA | 2.10.0+cu128 / CUDA 12.8 |
| 推理设置 | FP16、SDPA、seed 0、local files only |
| Suite isolation | 每个 suite 独立进程 |

当前服务器示例路径：

```text
project: /ai/projects/Infer-DaseSS
model:   /ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct
result:  /ai/projects/Infer-DaseSS/student_release/results/phase2_full_retry_20260718
```

路径不是实现约束；换服务器时通过 `PROJECT_ROOT` 和 `MODEL_PATH` 替换。

### 正式运行配置

- 不使用 `--limit`，完整运行六个公开 suite。
- 保留公开默认 batch sizes：decode `1,2,4`，TTFT `1`，mixed `1,2`，cache stress `2,4`。
- 保留公开默认 token budgets：Long 96、Decode 128、TTFT 1、Serving 96、Mixed 64、Cache Stress 128/256/512。
- warmup 1 次，`timed_repeats=3`，计时取中位数，worker timeout 1800 秒。

```bash
export PROJECT_ROOT=/ai/projects/Infer-DaseSS
export MODEL_PATH=/ai/llm/models/models/Qwen/Qwen2.5-0.5B-Instruct
export PYTHON_BIN="$PROJECT_ROOT/.venv-real/bin/python"
export INFERENCE_OPT_CACHE_ROOT=/path/to/writable/cache

cd "$PROJECT_ROOT/student_release"
source use_data_cache.sh

"$PYTHON_BIN" -u scripts/run_inference_benchmark.py \
  --model "$MODEL_PATH" \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --attn-implementation sdpa \
  --timed-repeats 3 \
  --baseline-summary data/public_baseline_summary.json \
  --suite-isolation process \
  --worker-timeout-s 1800 \
  --output-dir results/phase2_full_retry_20260718
```

### 正式分数

**Final score：86.03 / 100**（原始值 `86.0268153815`，cap 100）。

| 评分项 | 得分 | 关键指标 |
| --- | ---: | --- |
| Long Context Correctness | 28.50 / 30 | partial 0.950，exact 0.933 |
| Decode TPS | 23.35 / 25 | best batch=4，182.45 TPS，baseline ratio 0.812× |
| TTFT / Prefill | 15.36 / 20 | avg 0.222 s，p95 0.912 s，4 个长度 bucket |
| Serving / Scheduling | 8.82 / 15 | 95.94 TPS，p95 10.882 s，fallback `generate` |
| Runtime Robustness | 10.00 / 10 | runtime success 1.000，OOM count 0 |

总体共 119 条计分记录：runtime success `1.000`，valid output `0.933`，accuracy `0.924`，
peak allocated `27069.83 MiB`。valid output 与随包提供的 public baseline
`0.9327731092` 完全一致；它包含内容正确性判定，不能误写为运行失败率。

### Suite 指标

| Suite | 记录数 | Runtime | Valid | 主要结果 |
| --- | ---: | ---: | ---: | --- |
| long_context | 15 | 1.000 | 1.000 | score 0.950，33.28 TPS |
| decode_throughput | 36 | 1.000 | 1.000 | best batch=4，182.45 TPS |
| ttft_prefill | 8 | 1.000 | 1.000 | avg 0.222 s，p95 0.912 s |
| serving_schedule | 12 | 1.000 | 0.917 | 95.94 TPS |
| mixed_serving | 24 | 1.000 | 0.917 | best batch=2，60.52 TPS |
| decode_cache_stress | 24 | 1.000 | 0.792 | primary batch=4，cache metric 174.18 TPS |

Decode 的真实 batch scaling：

| Batch size | TPS | 平均 batch latency | p95 latency | Valid |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 44.90 | 2.808 s | 2.847 s | 1.000 |
| 2 | 91.51 | 2.756 s | 2.781 s | 1.000 |
| 4 | 182.45 | 2.764 s | 2.768 s | 1.000 |

batch=4 TPS 是 batch=1 的 `4.06×`，同时 batch latency 基本不变，说明 Phase 2 的 batch
prefill/decode 是真实并行执行，不是逐条循环计时。

### 运行历史

| 运行 | 配置/结果 | 结论 |
| --- | --- | --- |
| `phase2_smoke_20260718` | `--limit 1`，84.28 | 六套件链路 smoke；不能作为正式分数 |
| `phase2_full_20260718` | 完整运行，76.33 | 外部进程占用约 64 GiB GPU，serving prefill OOM；诊断结果，不采纳 |
| `phase2_serving_retry_20260718` | 仅 serving，runtime 1.0 | 验证通用 OOM chunk 二分恢复；单 suite 分数不可与总分比较 |
| `phase2_full_retry_20260718` | 完整运行，**86.03** | 六套件无 OOM/timeout；本阶段 accepted 结果 |

首轮 OOM 后只修改了 `StudentEngine` 的通用错误恢复：先按请求 batch 执行，只有
`torch.cuda.OutOfMemoryError` 才二分当前 chunk；单请求仍 OOM 时继续抛错。没有修改
benchmark 脚本、public JSONL、baseline summary、评分规则、默认 batch sizes 或 token budgets。

### 结果文件

accepted run 的服务器证据：

```text
results/phase2_full_retry_20260718/final_summary.json       147536 bytes
results/phase2_full_retry_20260718/final_summary.txt          2106 bytes
results/phase2_full_retry_20260718/student/summary.json
results/phase2_full_retry_20260718/student/results.csv       112192 bytes
```

正式分数和表格数据均从保留的 `final_summary.json` 提取，并与 `final_summary.txt` 交叉检查。

### 简要分析

- 正确性：真实 SDPA 与 Transformers eager 的最大 logits 绝对误差为 `3.50e-05`；Long partial 0.950。
- Decode：batch 1→2→4 接近线性扩展，是当前最明显的收益。
- TTFT：长 prompt 的 padding mask/math SDPA 临时张量使 p95 和显存偏高，是主要短板。
- Serving：当前仍使用 benchmark 的 `generate` fallback，没有请求流调度或 shared-prefix KV 复用。
- Cache：已有逐层 KV Cache，但 decode 时仍用 `torch.cat` 扩展，尚未预分配或分页管理。

### 下一步优化方向

1. 实现预分配/paged KV Cache，减少 decode 拼接与 allocator 压力。
2. 增加长度分桶及可使用 FlashAttention 的 padding-mask 路径，降低 TTFT 和峰值显存。
3. 实现 `serve_requests` 与 continuous batching，提升 Serving 分项并降低 p95。
4. 增加 shared-prefix KV 复用，改善 serving 和长上下文重复前缀场景。

