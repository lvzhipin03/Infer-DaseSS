# Infer-DaseSS Benchmark 测试记录

## Stage 0：RTX 2080 Ti 正式基线

### 测试状态

```text
状态:      已完成
退出状态:  正常
OOM:       0
Timeout:   0
Realism:   通过
最终分数:  61.58 / 100
```

### 测试环境

| 项目 | 值 |
| --- | --- |
| 服务器 | SeetaCloud AutoDL |
| Host | `autodl-container-1d0242a1d3-a7c2d7f3` |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| GPU 显存 | 11264 MiB |
| NVIDIA driver | 550.90.07 |
| Python | 3.10.20 |
| PyTorch | 2.4.1+cu121 |
| CUDA runtime | 12.1 |
| Transformers | 4.43.1 |
| tokenizers | 0.19.1 |
| safetensors | 0.4.5 |
| dtype | float16 |
| Attention | SDPA |
| seed | 0 |
| 模型 | Qwen2.5-0.5B-Instruct 本地权重 |

### 运行配置

```text
suites: long_context, decode_throughput, ttft_prefill,
        serving_schedule, mixed_serving, decode_cache_stress
limit: None
timed_repeats: 3
suite_isolation: process
worker_timeout: 1800s
local_files_only: true
```

远程运行命令：

```bash
cd /root/autodl-tmp/Infer-DaseSS/student_release
export INFERENCE_OPT_CACHE_ROOT=/root/autodl-tmp/cache

../.venv310/bin/python -u scripts/run_inference_benchmark.py \
  --model /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --attn-implementation sdpa \
  --timed-repeats 3 \
  --baseline-summary data/public_baseline_summary.json \
  --suite-isolation process \
  --worker-timeout-s 1800 \
  --output-dir results/stage0_2080ti_20260718
```

本次未使用额外的 11GB PyTorch allocator 限制，直接使用真实 RTX 2080 Ti 11GB 显存。

## 正式得分

| 评分项 | 得分 | 关键指标 |
| --- | ---: | --- |
| Long Context Correctness | 28.50 / 30 | partial=0.950，exact=0.933 |
| Decode TPS | 4.97 / 25 | best batch=4，51.5 TPS，speedup=0.23x |
| TTFT / Prefill | 16.22 / 20 | avg=0.126s，p95=0.420s |
| Serving / Scheduling | 1.88 / 15 | 39.5 TPS，p95=9.795s，iface=generate |
| Runtime Robustness | 10.00 / 10 | runtime=1.000，valid=0.613 |
| **总分** | **61.58 / 100** | cap=100 |

TTFT Breakdown：

```text
bucket avg:  9.52 / 12
bucket p95:  4.70 / 6
quality:     2.00 / 2
valid:       1.000
```

## Suite 汇总

| Suite | Runtime | Valid | 性能 | 峰值显存 |
| --- | ---: | ---: | --- | ---: |
| long_context | 1.000 | 0.933 | 30.1 TPS | 1140 MB |
| decode_throughput | 1.000 | 0.583 | best batch=4，51.5 TPS | 约 1–2 GB |
| ttft_prefill | 1.000 | 1.000 | avg=0.126s，8.0 first-token TPS | 1347 MB |
| serving_schedule | 1.000 | 0.250 | 39.5 TPS | 2524 MB |
| mixed_serving | 1.000 | 0.708 | best batch=2，35.7 TPS | 1513 MB |
| decode_cache_stress | 1.000 | 0.375 | cache TPS=44.1，primary batch=4 | 1275 MB |

全局峰值显存约 2524 MB。

## Diagnostics

```text
batch scaling:       1.33x
mixed TPS:           35.7 (0.35x baseline)
prefix shared:       0.59x
copy prefix:         0.050
cache metric TPS:    53.2 (0.46x baseline)
cache growth:        0.00 MB / 100 tokens
cache peak extra:    1573 MB
Realism Guard:       OK
```

## 初步结论

1. 真实 2080 Ti 运行没有 OOM 或 timeout，运行成功率为 1.000；
2. 峰值显存只有约 2.5GB，11GB 容量目前不是主要瓶颈；
3. Long Context 正确性较好，partial=0.950；
4. TTFT 表现相对稳定，valid=1.000；
5. Decode 的主要问题是吞吐低且 valid 低，best TPS 只有 51.5；
6. Serving 尚未实现 `serve_requests()`，使用 `generate()` fallback，得分很低；
7. Mixed 和 Cache Stress 的 valid 同样较低，需要检查逐条 CSV；
8. 2080 Ti 上 batch scaling 只有 1.33x，远弱于 A6000；
9. 应重点核验 PyTorch 2.4/Turing 的 SDPA GQA fallback，以及逐 token CPU 同步、KV `torch.cat` 等热点。

下一步应先分析 Decode、Serving 和 Cache Stress 的无效记录，区分内容质量失败、生成长度不足、数值分歧或接口问题，再进入 Stage 1 优化。

## 远程证据

日志：

```text
/root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage0_2080ti_20260718.log
```

结果目录：

```text
/root/autodl-tmp/Infer-DaseSS/student_release/results/stage0_2080ti_20260718
```

核心文件：

```text
final_summary.json
final_summary.txt
student/summary.json
student/results.csv
student/suite_long_context/summary.json
student/suite_decode_throughput/summary.json
student/suite_ttft_prefill/summary.json
student/suite_serving_schedule/summary.json
student/suite_mixed_serving/summary.json
student/suite_decode_cache_stress/summary.json
```

## 后续追加规则

后续每个优化阶段在本文末尾追加，不覆盖 Stage 0。每次记录：

- 阶段名称和代码版本；
- 具体优化内容；
- 完整运行配置；
- 总分和五项得分；
- 六套件指标；
- 峰值显存、OOM 和 timeout；
- 与 Stage 0 及上一阶段的差值；
- accepted/rejected 结论及原因。
