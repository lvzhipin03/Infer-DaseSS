# Stage 3B：Share Allowed Attention Mask Across Layers

## 1. 问题背景

Qwen2.5-0.5B 的 24 个 Decoder 层在同一次 forward 中使用完全相同的：

```text
batch size
query length
key length
past length
padding attention mask
```

但原实现在每层 Attention 内都重新调用 `_allowed_attention_mask()`，重复执行：

- query/key `arange`；
- causal `<=` 比较；
- reshape；
- padding mask 布尔化；
- causal mask 与 padding mask 的 `bitwise_and`。

这些结果在 24 层之间没有差异，因此属于纯重复工作。

## 2. 优化方式

Stage 3B 在 `QwenToyModel.forward()` 开始时只构造一次 allowed mask：

```python
allowed_attention_mask = _allowed_attention_mask(
    batch,
    length,
    past_length + length,
    past_length,
    attention_mask,
    input_ids.device,
)
```

然后把同一个 tensor 传给 24 个 Decoder Layer 和 Attention。

`QwenToyAttention` 仍保留独立调用能力：外部直接调用 Attention 而没有传入预计算 mask 时，会按原路径自行构造 mask。

## 3. 正确性边界

该修改不改变：

- causal mask 的数值；
- left-padding key mask；
- fully-masked padding Query；
- Stage 0.5 `torch.where` 清理；
- Prefill/Decode 的 past length；
- SDPA/GQA backend。

传入 Attention 前会检查共享 mask shape 是否为：

```text
[batch, 1, query_length, key_length]
```

## 4. Profiler 结果

远程 RTX 2080 Ti，batch=2、32-token Decode：

| 指标 | Static Cache | + Shared Mask | 变化 |
| --- | ---: | ---: | ---: |
| CPU total | 1.300s | 1.179s | -9.3% |
| CUDA total | 194.496ms | 185.858ms | -4.4% |
| `arange` calls | 3076 | 132 | -95.7% |
| `bitwise_and` calls | 768 | 32 | -95.8% |
| causal `le` calls | 768 | 32 | -95.8% |

本地与远程完整单元测试均为：

```text
Ran 87 tests
OK
```

## 5. Stage 2 实验结论

在实施 Stage 3B 前，还验证了两个 Stage 2 候选改进：

1. 缓存 Static Cache 结构验证：没有可确认收益，增加复杂度，已撤回；
2. K/V projection 直接写入 token-major Cache slot：K 的原地 RoPE 需要额外 clone/copy 和小算子，CUDA 时间从 194.50ms 增加到 196.63ms，已撤回。

这说明在纯 PyTorch 小算子中模拟 K/V projection + RoPE 融合并不划算。真正的直接 Cache 写入需要 fused kernel，不应用更多 clone 和 elementwise launch 代替一次 copy。

## 6. 当前组合

当前保留：

```text
Stage 1   移除逐 token CPU/GPU 同步
Stage 2   Static KV Cache
Stage 3A  预分配 generation attention mask
Stage 3B  24 层共享 allowed attention mask
```

下一步应先运行快速 Decode benchmark；只有吞吐结果与 profiler 方向一致时，才运行完整六套件。

## 7. Decode 定向结果

完整 12 个 Decode 样本、batch=1/2/4、128 tokens、3 次计时的远程结果：

| Batch | Stage 0.5 | Stage 1+2+3A | + Stage 3B |
| ---: | ---: | ---: | ---: |
| 1 | 38.39 TPS | 33.90 TPS | 38.03 TPS |
| 2 | 76.10 TPS | 73.69 TPS | 79.55 TPS |
| 4 | 151.73 TPS | 146.01 TPS | 157.34 TPS |

```text
Decode valid: 1.000
Stage 3B batch=4 vs Stage 0.5:      +3.70%
Stage 3B batch=4 vs Stage 1+2+3A:  +7.76%
```

结果目录：

```text
/root/autodl-tmp/Infer-DaseSS/student_release/results/stage1_2_3ab_decode_20260719
```

## 8. 后续优化：Decode 跳过 Padding Query 清理

Stage 0.5 的 `torch.where` 是为了清理 left-padding Prefill 中 fully-masked PAD Query 的非有限输出。

cached Decode 的当前 Query 是上一步 greedy argmax 生成的真实 token，其 attention mask 新 slot 固定为 1，不是左侧 PAD Query。因此在：

```text
past_length > 0
```

时不需要每层再执行 `torch.where`。清理仍完整保留在 Prefill（`past_length == 0`）路径，所以 Stage 0.5 的 left-padding correctness 保护不变。

## 9. Decode 跳过 `torch.where` 的 Profiler

RTX 2080 Ti，batch=2、32-token Decode：

| 指标 | Shared Mask | + Decode skip `where` | 变化 |
| --- | ---: | ---: | ---: |
| `aten::where` calls | 768 | 24 | -96.9% |
| `aten::where` CUDA | 2.427ms | 0.079ms | -96.7% |
| CPU total | 1.179s | 1.040s | -11.8% |
| CUDA total | 185.858ms | 180.570ms | -2.8% |

其中剩余 24 次 `where` 来自 24 层 Prefill，它们必须保留以防止 fully-masked left-padding Query 的非有限值传播。

## 10. 完整六套件结果

加入 Fused MLP 后的当前最佳组合：

| 指标 | Stage 0.5 | 当前最佳 | 变化 |
| --- | ---: | ---: | ---: |
| 总分 | 86.49 | **87.16** | +0.67 |
| Decode TPS | 149.0 | **172.5** | +15.8% |
| Serving TPS | 105.2 | **117.7** | +11.9% |
| Serving p95 | 9.920s | **8.867s** | -10.6% |
| Mixed TPS | 55.0 | **66.0** | +20.0% |
| Cache TPS | 149.6 | **172.1** | +15.0% |
| Long TPS | 29.3 | **34.9** | +19.1% |

正确性：

```text
Long partial / exact: 0.950 / 0.933
Decode valid:         1.000
TTFT valid:           1.000
Aggregate valid:      0.958
Runtime success:      1.000
Realism Guard:        OK
```

结果和日志：

```text
/root/autodl-tmp/Infer-DaseSS/student_release/results/stage_best_full_20260719
/root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage_best_full_20260719.log
```
