# Stage 1：Remove Per-Token Synchronization Overhead

## 1. 阶段目标

Stage 1 的目标是减少 Decode 每生成一个 token 时发生的 CPU/GPU 同步、设备间传输和 Python 元数据处理。

本阶段不改变 Attention 数学、KV Cache 存储结构、mask 语义、GQA 实现或请求调度。所有 greedy token IDs 必须与 Stage 0.5 完全一致。

## 2. 前置条件

Stage 1 必须在 Stage 0.5 通过后开始。

Stage 0.5 修复了变长 batch 左 padding 的非有限值传播，2080 Ti 四套件诊断结果为：

```text
Decode valid:          0.583 → 1.000
Decode best TPS:       51.5  → 151.5
Serving valid:         0.250 → 0.917
Serving TPS:           39.5  → 104.6
Mixed valid:           0.708 → 0.917
Mixed best TPS:        35.7  → 56.8
Cache Stress valid:    0.375 → 0.917
Cache Stress TPS:      44.1  → 95.9
Runtime success:       1.000
OOM / timeout:         0 / 0
```

Stage 0.5 完整六套件最终为 86.49 分，Decode best TPS=149.0，valid=1.000，已作为 Stage 1 及后续组合优化的正式比较基线。

## 3. CPU 与 GPU 异步执行背景

CUDA kernel launch 通常是异步的：CPU 提交 GPU 工作后可以继续执行，不必立即等待 kernel 完成。

理想 Decode 流程：

```text
CPU 提交 layer kernels
CPU 提交 argmax
CPU 立即准备下一次 GPU 工作
GPU 按队列执行
整段生成结束后再同步结果
```

如果每个 token 都把 GPU tensor 读取到 CPU：

```text
CPU 提交 GPU 工作
        ↓
CPU 请求读取 GPU tensor
        ↓
强制等待前面所有 GPU 工作完成
        ↓
执行 D2H copy
        ↓
CPU 处理 Python list
        ↓
再提交下一 token
```

这会破坏 CUDA 的异步流水，使 Decode 变成频繁的“提交—等待—复制—再提交”。

## 4. 当前逐 Token D2H 同步

当前代码位于：

```text
toy_qwen/generation.py
```

每个 Decode step 执行：

```python
next_ids = output.logits[:, -1, :].argmax(dim=-1)

for row, token_id in enumerate(next_ids.detach().cpu().tolist()):
    generated[row].append(int(token_id))
```

其中：

```python
next_ids.cpu()
```

要求 CPU 立即取得 GPU argmax 结果，因此会触发同步和 device-to-host copy。

若 `max_new_tokens=128`，一次 generate 至少发生约 128 次此类同步；Cache Stress 的 512-token case 会发生约 512 次。

batch 增大时，每次传输的数据量仍然很小，但同步次数不变。小而频繁的同步通常比一次性传输更低效。

## 5. 计划修改：GPU 上累计结果

生成开始时在 GPU 预分配：

```python
generated_ids = torch.empty(
    (batch_size, max_new_tokens),
    dtype=torch.long,
    device=batch.input_ids.device,
)
```

每步仅在 GPU 写入：

```python
generated_ids[:, index] = next_ids
```

整个 Decode 完成后一次性传回：

```python
generated_rows = generated_ids.cpu().tolist()
```

预期变化：

```text
修改前: 每 token 一次 D2H copy/synchronization
修改后: 每次 generate 一次 D2H copy/synchronization
```

最终仍转换为 benchmark 当前需要的：

```text
tuple[tuple[int, ...], ...]
```

不会改变输出接口。

## 6. 当前逐 Token Token-ID 范围检查

当前代码位于：

```text
toy_qwen/modeling.py
```

每次 model forward 都执行：

```python
if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
    raise ValueError("input token id is outside the vocabulary")
```

在 CUDA 上，`min()` 和 `max()` 结果是 GPU tensor。Python 的 `if` 必须读取比较结果，因此会等待 GPU 完成，形成同步点。

Prefill 时范围检查是必要的，因为 token IDs 来自外部 tokenizer/input。

Decode 时 `current_ids` 来自：

```python
argmax(logits)
```

argmax 的输出由词表维度产生，必然满足：

```text
0 <= token_id < vocab_size
```

因此对内部 Decode token 每步重复检查没有提供额外安全性，却增加同步。

## 7. 计划修改：只验证外部 Prefill 输入

第一版建议：

```python
if past_key_values is None:
    validate_input_id_range(input_ids)
```

含义：

- 无 cache：这是 Prefill 或外部模型调用，执行完整范围验证；
- 有 cache：这是由内部 greedy decode 产生的新 token，跳过重复范围检查。

必须保留以下行为：

- 初始输入包含负 token 时仍报错；
- 初始输入超过词表时仍报错；
- 空 tensor/错误 rank 仍报错；
- max position 检查仍每次有效；
- cache shape 验证仍有效。

如果后续出现允许外部用户直接传 cache + 任意 input IDs 的正式 API，需要改成显式内部 fast-path 参数，而不能单纯依赖 `past_key_values is not None`。当前 benchmark 只通过生成器内部调用 cache decode。

## 8. 当前逐 Token Cache Shape 元数据处理

当前 `batched_greedy_generate()` 每步执行：

```python
last_cache_shapes = _cache_shapes(past_key_values)
if first_cache_shapes is None:
    first_cache_shapes = last_cache_shapes
```

`_cache_shapes()` 会遍历模型的 24 层，为每层 K/V 构造 shape tuple。

它通常不会读取 tensor 数值，也不一定触发 CUDA 同步，但会带来：

- 每步遍历 24 层；
- 构造大量 Python tuple；
- 重复读取不会在中间使用的 shape；
- 增加 Decode hot path 的 Python 工作。

128-token Decode 会处理约：

```text
24 layers × 128 steps = 3072 layer cache shape records
```

这些 shape 只用于测试和教学 trace，不需要逐 token 保存。

## 9. 计划修改：只记录首步和末步 Shape

首步：

```python
if first_cache_shapes is None:
    first_cache_shapes = _cache_shapes(past_key_values)
```

循环结束后：

```python
last_cache_shapes = _cache_shapes(past_key_values)
```

这样仍能返回当前 dataclass 所需的：

```text
first_cache_shapes
last_cache_shapes
```

但不再记录无用的中间 shape。

## 10. 本阶段明确不修改的内容

Stage 1 不修改：

- 每层 K/V 的 `torch.cat`；
- attention mask 的 `torch.cat`；
- RoPE；
- left padding；
- Stage 0.5 的 `torch.where` 修复；
- SDPA/GQA backend；
- `repeat_kv` fallback；
- tokenizer；
- greedy argmax；
- fixed-step decode；
- OOM 二分恢复；
- `serve_requests()`；
- Prefix Cache；
- Paged KV；
- benchmark 脚本、数据和 baseline。

这些内容分别留给后续独立阶段。

## 11. 为什么 KV `torch.cat` 不在本阶段处理

当前 Decode 每层每步执行：

```python
key = torch.cat((past_key, key), dim=2)
value = torch.cat((past_value, value), dim=2)
```

这是更大的性能问题，但它需要改变 Cache 数据结构、写入方式和 Attention 读取方式，风险显著高于去同步。

Stage 1 只处理不改变模型数学与 Cache 语义的开销，Stage 2 再单独实现预分配连续 KV Cache。这样性能或正确性回退能够准确归因。

## 12. 为什么 Mask `torch.cat` 不在本阶段处理

当前每步扩展：

```python
full_attention_mask = torch.cat(...)
```

移除它需要预分配 mask capacity，并协调物理 cache slot、逻辑 position 和变长 batch。这属于 Stage 3。

Stage 1 不混入 mask 结构变化。

## 13. 回归测试计划

### 生成一致性

覆盖：

```text
batch=1/2/4
max_new_tokens=1/16/128
相同长度 prompt
不同长度 prompt
极端左 padding
```

要求 Stage 1 与 Stage 0.5 的原始 generated token IDs 完全一致。

### 输入范围验证

验证：

- Prefill 负 token 仍报错；
- Prefill 超词表 token 仍报错；
- 合法 cache decode 正常；
- 内部 argmax token 不执行 GPU 范围同步；
- context overflow 仍报错。

### Cache Shape

继续验证：

- `first_cache_shapes` 正确；
- `last_cache_shapes` 正确；
- max_new_tokens=1 时两者相同；
- max_new_tokens>1 时末步长度正确。

### CUDA Profiling

在 2080 Ti 上比较修改前后：

- D2H copy 次数；
- CUDA synchronize/event 等待；
- 每 token CPU launch gap；
- batch=1/2/4 latency；
- Decode TPS；
- GPU utilization。

## 14. Stage 1 验收标准

正确性要求：

```text
batch=1/2/4 greedy token IDs 与 Stage 0.5 一致
Decode valid 保持 1.000
Long partial/exact 不下降
Serving/Mixed/Cache valid 不下降
Runtime success 保持 1.000
```

性能要求：

```text
每次 generate 只在最终结果收集时进行一次 D2H copy
缓存 Decode 不再执行 token min/max GPU 同步
cache shape 不再逐 token 遍历 24 层
Decode TPS 不下降
```

若 TPS 提升不明显，但 profiler 确认同步点已移除，可以接受结构改动；必须记录实际收益，并把主要瓶颈继续归因到 KV `torch.cat`、权重带宽或 Attention backend。

## 15. 预计修改范围

主要代码：

```text
toy_qwen/generation.py
toy_qwen/modeling.py
```

测试：

```text
tests/test_batch_generation.py
tests/test_model.py
tests/test_config.py（仅在合适时）
```

不会修改 benchmark 脚本、公开数据、baseline 或 `student_release` 接口。

## 16. 远程验证流程

1. 本地完整单元测试；
2. 本地真实权重接口验证；
3. 增量同步 Stage 1 文件；
4. 远程 `validate_engine.py`；
5. 远程只跑 `decode_throughput` 诊断；
6. valid 与 token IDs 保持一致后，运行完整六套件；
7. 结果保存为新的阶段目录；
8. 与 Stage 0.5 正式结果比较，而不是与存在 correctness bug 的 Stage 0 比较。

## 17. 实际 A/B 结果与最终去向

在相同 RTX 2080 Ti、完整 Decode 数据、batch=1/2/4、128 tokens、3 次计时下：

| Batch | Stage 0.5 | Stage 1 | 变化 |
| ---: | ---: | ---: | ---: |
| 1 | 38.39 TPS | 39.54 TPS | +2.99% |
| 2 | 76.10 TPS | 77.01 TPS | +1.19% |
| 4 | 151.73 TPS | 150.80 TPS | -0.61% |
| Aggregate | 65.54 TPS | 66.81 TPS | +1.94% |

Stage 1 对小 batch 有小幅收益，对 batch=4 的差异在波动范围。因为它移除了确定的 D2H 同步点，且 valid=1.000、接口不变，最终作为后续 Stage 2/3 和算子融合的基础保留。

当前组合六套件总分为 87.16，Decode best TPS=172.5，但这是 Stage 1+2+3A+3B+Fused MLP 的组合收益，不应单独归因于 Stage 1。
