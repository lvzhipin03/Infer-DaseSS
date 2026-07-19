# Stage 0.5：Left Padding Batch Correctness

## 1. 阶段目标

Stage 0.5 的目标是修复真实 RTX 2080 Ti 上变长 batch 推理的正确性问题：同一组 prompts 使用 batch=1、2、4 时，应生成一致的 greedy token 序列。

本阶段不是性能优化。只有 batch 输出正确后，Decode TPS、Serving TPS 和 batch scaling 才有评分意义。

## 2. 问题背景

Stage 0 在真实 RTX 2080 Ti 11GB 上完成了标准六套件 benchmark：

```text
FINAL SCORE:             61.58 / 100
Long Context partial:    0.950
Long Context exact:      0.933
Decode best TPS:         51.5
Decode valid:            低
TTFT avg:                0.126s
TTFT valid:              1.000
Serving TPS:             39.5
Serving valid:           0.250
Runtime success:         1.000
OOM / timeout:           0 / 0
全局峰值显存:           约 2.5GB
```

逐条 CSV 显示：

```text
Decode batch=1:
  多数样本能得到接近 128 个有效 tokens

Decode batch=2/4:
  大量失败样本只得到 16 个有效 tokens

Serving batch=12:
  9/12 请求只得到 12 个有效 tokens

Cache Stress:
  大量输出按 16/32/64 tokens 退化，
  约为对应 128/256/512 token budget 的 1/8
```

这些记录没有 Python exception、CUDA exception、OOM 或 timeout。因此，当前主要问题不是 11GB 显存容量，而是 batch 推理产生了无效内容或大量特殊 token。

另一个重要现象是：

- Long Context 使用 batch=1，结果正常；
- TTFT 默认使用 batch=1，valid=1.000；
- Decode、Serving、Mixed、Cache Stress 在 batch>1 时明显恶化。

这使问题高度指向变长 batch、左 padding、Attention mask、SDPA backend 或 GQA fallback 的组合。

## 3. 当前 Batch Padding 方式

当前实现位于：

```text
toy_qwen/generation.py
```

`left_pad_token_ids()` 将不同长度的 prompts 左填充到 batch 内最长宽度。

假设最长 prompt 有 5 个 tokens，短 prompt 只有 2 个 tokens：

```text
物理位置:  0    1    2    3    4
短请求:   PAD  PAD  PAD   A    B
长请求:    C    D    E    F    G
```

短请求对应：

```text
attention_mask: [0, 0, 0, 1, 1]
position_ids:   [0, 0, 0, 0, 1]
```

这里需要区分两种位置：

- 物理位置：token 在 padded tensor 中的列下标；
- 逻辑位置：真实 token 在原始 prompt 中的位置，用于 RoPE。

短请求中的 A 位于物理位置 3，但逻辑 position ID 为 0；B 位于物理位置 4，但逻辑 position ID 为 1。

## 4. Fully-Masked Row 如何产生

Attention 同时使用 causal mask 和 padding mask。

### Padding Query 0

物理位置 0 的 Query 根据 causal 规则只能看位置 0：

```text
causal 允许:  [1, 0, 0, 0, 0]
padding 允许: [0, 0, 0, 1, 1]
最终 allowed: [0, 0, 0, 0, 0]
```

这一行没有任何允许访问的 Key，因此是 fully-masked row。

### Padding Query 1

物理位置 1 的 Query 可以看物理位置 0、1，但它们都是 PAD：

```text
causal 允许:  [1, 1, 0, 0, 0]
padding 允许: [0, 0, 0, 1, 1]
最终 allowed: [0, 0, 0, 0, 0]
```

仍然是 fully masked。

### Padding Query 2

```text
causal 允许:  [1, 1, 1, 0, 0]
padding 允许: [0, 0, 0, 1, 1]
最终 allowed: [0, 0, 0, 0, 0]
```

仍然是 fully masked。

### 第一个有效 Query A

A 位于物理位置 3：

```text
causal 允许:  [1, 1, 1, 1, 0]
padding 允许: [0, 0, 0, 1, 1]
最终 allowed: [0, 0, 0, 1, 0]
```

A 至少可以看到自己，因此有效 Query 不会整行被 mask。

结论：fully-masked row 发生在短请求左侧补出的 PAD Query，而不是发生在 A、B 等有效 token 上。

## 5. 为什么 Fully-Masked Row 可能有风险

标准 Attention 可以写成：

```text
Attention(Q, K, V) = softmax(QK^T + mask) V
```

一整行全部被 mask 时，逻辑上接近：

```text
softmax([-∞, -∞, -∞, ...])
```

部分实现会对这种情况做安全处理并返回 0；部分 backend 可能产生 NaN。具体行为可能随以下条件变化：

- GPU 架构；
- PyTorch 版本；
- Flash、memory-efficient 或 math SDPA backend；
- bool mask 或 additive mask；
- GQA 是否原生支持；
- dtype。

当前 A6000 和 2080 Ti 环境不同：

```text
A6000:
  PyTorch 2.6.0+cu124
  Ampere

2080 Ti:
  PyTorch 2.4.1+cu121
  Turing
```

同一段 SDPA 代码可能选择不同 kernel，因此不能因为 A6000 正常就推断 2080 Ti 一定正常。

## 6. 当前清零方式的潜在问题

当前 Attention 在 SDPA 后执行：

```python
query_is_valid = attention_mask[:, key_length - length :].bool()
output = output * query_is_valid[:, None, :, None]
```

设计意图是把 PAD Query 的输出乘 0 清除。

但如果 PAD Query 的 SDPA output 已经是 NaN：

```text
NaN × 0 = NaN
```

因此乘法不能保证清除无效位置的非有限值。

潜在传播路径：

```text
Padding SDPA output = NaN
          ↓
乘 0 后仍是 NaN
          ↓
Residual hidden state = NaN
          ↓
下一层 RMSNorm = NaN
          ↓
下一层 padding K/V = NaN
          ↓
后续 Attention 可能被污染
          ↓
有效 token logits 错误
          ↓
模型生成大量特殊 token
          ↓
skip_special_tokens=True 删除特殊 token
          ↓
benchmark 只看到 12/16 个有效 tokens
```

## 7. 排查阶段的限定

目前已经确认：

- fully-masked PAD Query 在当前左 padding + causal mask 组合中客观存在；
- batch>1 的有效输出率显著下降；
- 问题不是 OOM 或 runtime exception。

目前尚未确认：

- 2080 Ti 的实际 SDPA output 是否在 fully-masked row 上产生 NaN；
- NaN 是否进入下一层 K/V；
- 12/16-token 退化是否完全由特殊 token 引起；
- GQA fallback 是否参与问题；
- 是否还存在 cache position 或 mask 对齐问题。

因此在实施修复前，Stage 0.5 没有直接把 `torch.where` 当作已证明的答案，而是把它作为需要通过单元测试和真实 2080 Ti benchmark 验证的候选修复。后续结果见第 10、11 节。

## 8. 计划排查的位置

主要代码：

```text
toy_qwen/generation.py
toy_qwen/modeling.py
toy_qwen/cache.py
student_release/student_engine.py
```

重点检查：

1. `left_pad_token_ids()` 生成的 input IDs、mask、position IDs；
2. `_allowed_attention_mask()` 的物理 causal 位置和 padding mask；
3. SDPA 前的 Q/K/V 是否有限；
4. SDPA output 是否有限；
5. 当前乘法清零后是否仍存在 NaN；
6. residual 后 padding hidden 是否有限；
7. 下一层 padding K/V 是否有限；
8. batch 与逐条的最后位置 logits 从哪一层开始分歧；
9. Decode cache length、逻辑 position 和物理 slot 是否一致；
10. `enable_gqa=True` 是否成功，或是否进入 `repeat_kv` fallback。

## 9. 实际代码修改

修改位于：

```text
toy_qwen/modeling.py
```

修改前，SDPA 输出通过乘法清理无效 Query：

```python
output = output * query_is_valid[:, None, :, None]
```

修改后，使用 `torch.where` 直接选择标量零：

```python
output = torch.where(
    query_is_valid[:, None, :, None],
    output,
    output.new_zeros(()),
)
```

两者在有限数输入上的目标相同，但关键区别是：

```text
NaN * 0       = NaN
where(False, NaN, 0) = 0
```

因此，即使 2080 Ti 的 SDPA backend 在 fully-masked padding Query 上返回非有限值，这些值也不再进入 residual stream 和后续层的 K/V Cache。有效 Query 的 Attention 输出不变。

修复提交为：

```text
9a9098c fix: clear non-finite outputs for padded attention queries
```

## 10. 回归测试

新增测试位于：

```text
tests/test_attention.py
```

测试不依赖当前 GPU backend 恰好产生 NaN，而是模拟 SDPA 在 fully-masked row 上返回 NaN，然后断言：

- padding Query 的最终 Attention output 为有限的零；
- 有效 Query 仍保持有限；
- 修复前测试失败，修复后通过。

本地完整回归测试结果：

```text
87 tests passed
```

## 11. RTX 2080 Ti 完整六套件结果

测试在远程 RTX 2080 Ti 11GB 上运行，使用 Qwen2.5-0.5B-Instruct 真实权重和标准六套件，未施加 11GB 以外的人为显存限制。

| 测试项 | Stage 0 | Stage 0.5 | 变化 |
| --- | ---: | ---: | ---: |
| Long Context | 28.50 / 30 | 28.50 / 30 | 0.00 |
| Decode | 4.97 / 25 | 22.75 / 25 | +17.78 |
| TTFT | 16.22 / 20 | 16.19 / 20 | -0.03 |
| Serving | 1.88 / 15 | 9.05 / 15 | +7.17 |
| Runtime | 10.00 / 10 | 10.00 / 10 | 0.00 |
| **总分** | **61.58** | **86.49** | **+24.91** |

关键指标：

```text
Long partial / exact:     0.950 / 0.933
Decode best batch:        4
Decode best TPS:          149.0
Decode valid:             1.000
Decode batch scaling:     4.07x
TTFT avg / p95:           0.126s / 0.419s
TTFT valid:               1.000
Serving TPS:              105.2
Serving valid:            0.917
Mixed best TPS:           55.0
Mixed valid:              0.917
Cache primary TPS:        100.2
Cache metric TPS:         149.6
Cache valid:              0.917
Aggregate valid:          0.958
Peak GPU memory:          2524 MB
OOM / timeout:            0 / 0
Realism Guard:            OK
```

远程结果和日志：

```text
/root/autodl-tmp/Infer-DaseSS/student_release/results/stage0_5_mask_fix_full_20260719
/root/autodl-tmp/Infer-DaseSS/benchmark_logs/stage0_5_mask_fix_full_20260719.log
```

## 12. 结论

Stage 0.5 已验收。最终结果支持以下归因：

1. Stage 0 的主要丢分不是显存容量或 runtime crash；
2. 问题与变长 left-padding batch 中的 fully-masked padding Query 相关；
3. 乘零无法清除非有限值，`torch.where` 阻断了它们向 residual 和后续 K/V 的传播；
4. 修复后 Decode valid 恢复到 1.000，Decode 与 Serving 的吞吐和得分大幅恢复；
5. Long Context、TTFT 和 Runtime 基本不变，说明修复没有引入明显的单请求正确性或性能回退。

## 附录 A：排查时的候选修复方向

排查阶段将“把乘法清零改为条件选择”作为首选候选修复：

```python
output = torch.where(
    query_is_valid[:, None, :, None],
    output,
    torch.zeros((), dtype=output.dtype, device=output.device),
)
```

语义是：

```text
有效 Query: 保留 SDPA output
无效 Query: 明确选择有限的 0
```

如果 NaN 在 output 清零前已经通过其他路径传播，还可能需要：

- 在每层结束时清零 padding hidden；
- 保证 padding K/V 为有限的 0；
- 为 fully-masked query 提供安全的 attention 处理；
- 把 Prefill padding query 与有效 query 分开处理；
- 在 2080 Ti 上选择稳定的 SDPA backend。

不能仅对全模型使用 `torch.nan_to_num()`。这会隐藏有效 token 上的真实数值错误，也不能解释根因。

## 附录 B：为什么必须优先修复

当前系统的问题不是单纯“速度慢”，而是 batch 越大，输出越不可靠。

如果直接进入后续性能优化：

- 提高 Decode TPS 没有意义，因为大量输出无效；
- `serve_requests()` 会使用更复杂、更动态的 batch，可能放大错误；
- Continuous Batching 会频繁混合不同长度请求；
- Prefix Cache 和 Paged KV 会让错误更难定位；
- CUDA Graph 可能把错误路径固化；
- 长度分桶可能暂时减少触发次数，但会掩盖而不是修复 bug。

正确顺序应当是：

```text
修复 batch correctness
        ↓
去除逐 token CPU/GPU 同步
        ↓
预分配 KV Cache 和 mask
        ↓
优化 GQA Decode
        ↓
实现 Serving、Prefix Cache、Continuous Batching
```

## 附录 C：Stage 0.5 验收标准

必须满足：

```text
同一组 prompts:
batch=1 逐条 greedy tokens
== batch=2 greedy tokens
== batch=4 greedy tokens
```

同时要求：

- Decode valid 恢复到接近 1.000；
- Serving 不再大量退化为 12 tokens；
- Decode 不再大量退化为 16 tokens；
- Cache Stress 不再按 16/32/64 tokens 退化；
- Long partial/exact 不下降；
- TTFT valid 保持 1.000；
- runtime success 保持 1.000；
- 不出现 OOM 或 timeout；
- 不通过强制 batch=1、逐条 fallback 或样本特判掩盖问题。

## 附录 D：本阶段不做的优化

Stage 0.5 不包含：

- 预分配 KV Cache；
- PagedAttention；
- Prefix Cache；
- `serve_requests()`；
- Continuous Batching；
- Chunked Prefill；
- CUDA Graph；
- `torch.compile`；
- INT8/INT4 量化；
- 自定义高性能 GQA kernel。

这些优化将在 batch correctness 恢复后按独立阶段实施。
