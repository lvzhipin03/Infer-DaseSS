# Stage 2：Preallocated Contiguous KV Cache

## 1. 阶段目标

Stage 2 的目标是移除 Decode 过程中每层、每个 token 扩展 K/V 时的 `torch.cat`，改为生成开始时一次分配连续 Cache，后续 token 原地写入固定 slot。

本阶段不改变 Attention 数学、RoPE 位置、left padding 语义、greedy argmax 和 Stage 0.5 的非有限值修复。

## 2. 当前实现

当前代码位于：

```text
toy_qwen/modeling.py
```

每个 Attention 层在 Decode 时执行：

```python
key = torch.cat((past_key_value[0], key), dim=2)
value = torch.cat((past_key_value[1], value), dim=2)
```

`torch.cat` 不能在原 tensor 末尾扩容。它必须：

1. 申请一块更大的新 tensor；
2. 把所有历史 K/V 复制到新 tensor；
3. 把当前 token 的 K/V 写到末尾；
4. 释放或回收上一步 tensor。

Qwen2.5-0.5B 共有 24 层，K/V 头数为 2，head dimension 为 64。对每个 Decode token，K 和 V 都会在 24 层上重复执行上述操作。

## 3. 为什么 `torch.cat` 的成本会增长

设 Prefill 后 Cache 长度为 `P`，生成 `T` 个 token。原实现需要复制的历史长度近似：

```text
P + (P + 1) + (P + 2) + ... + (P + T - 2)
```

即：

```text
O(P * T + T²)
```

生成越长，历史 K/V 被重复搬运的次数越多。Cache Stress 的 256/512-token case 会比 128-token Decode 更明显地暴露这个问题。

Decode 通常偏 memory-bound。这些历史 Cache 复制不产生新的模型信息，却消耗显存带宽、allocator 时间和 kernel launch。

## 4. 预分配 Cache 的设计

生成开始时，根据 batch、KV 头数、容量和 head dimension 一次分配每层 K/V：

```text
[batch, num_key_value_heads, capacity, head_dim]
```

其中：

```text
capacity = padded_prompt_width + max_new_tokens - 1
```

之所以减 1，是因为 Prefill forward 已经产生第一个 continuation token 的 logits；最后一个生成 token 不需要再进入模型。

Prefill 写入：

```python
key_buffer[:, :, 0:prompt_width].copy_(key)
value_buffer[:, :, 0:prompt_width].copy_(value)
```

Decode 第 `i` 步写入：

```python
key_buffer[:, :, cache_length:cache_length + 1].copy_(key)
value_buffer[:, :, cache_length:cache_length + 1].copy_(value)
```

Attention 只读取已填充的逻辑视图：

```python
key = key_buffer[:, :, :new_cache_length]
value = value_buffer[:, :, :new_cache_length]
```

这些 slice 是 view，不复制历史 K/V。

## 5. Cache 长度与物理容量

预分配后必须区分：

- `capacity`：底层 buffer 能容纳的最大 token 数；
- `length`：当前已写入、Attention 可见的 token 数。

Attention mask、causal query position、RoPE position 和上下文越界检查都必须使用逻辑 `length`，不能把整个 `capacity` 当作有效上下文。

## 6. 为什么保留 Legacy Tuple Cache

现有单元测试和外部教学接口允许直接调用：

```python
output = model(input_ids, use_cache=True)
```

并期望获得按层组织的 tuple K/V。Stage 2 不应强制所有调用者迁移到新 Cache 类型。

因此采用双路径：

- 未传入预分配 Cache：保持原 tuple Cache 语义；
- 生成器显式传入 `StaticKVCache`：使用原地写入 fast path。

这样可以将 Stage 2 的影响限制在 benchmark 的 generation hot path，同时保留已有 API 兼容性。

## 7. 正确性边界

Stage 2 必须保证：

1. Cache 每层 K/V 的逻辑 shape 与原实现相同；
2. Prefill 和每个 Decode step 的 logits/greedy token 与 Stage 1 一致；
3. 只有所有层完成当次 forward 后，全局 Cache length 才增加；
4. 容量不足时明确报错，不允许静默越界；
5. dtype、device、batch、KV heads 和 head dimension 必须与模型一致；
6. Stage 0.5 对 padding Query 的 `torch.where` 修复保持不变；
7. left padding 的物理 slot 与逻辑 RoPE position 仍按原路径处理。

## 8. 预期收益

修改前：

```text
每层 × 每 token：allocate + copy all past K + copy all past V
```

修改后：

```text
每次 generate：一次预分配
每层 × 每 token：只 copy 当前 token K/V 到固定 slot
```

预期效果：

- 长 Decode 吞吐提升；
- Cache Stress 256/512-token case 收益大于短 Decode；
- allocator 压力和临时显存波动降低；
- Cache tensor 地址稳定，为后续 CUDA Graph 和 continuous batching 打基础。

## 9. 本阶段不包含

Stage 2 不同时实现：

- Paged KV Cache；
- Prefix Cache；
- Cache eviction/reuse；
- Continuous Batching；
- Attention mask 预分配；
- 专用 GQA Decode kernel；
- CUDA Graph；
- 量化。

`full_attention_mask` 的逐 token `torch.cat` 仍然保留，由后续 Stage 3 单独处理。

## 10. 后续验证计划

本次按要求只实现，不运行测试。后续建议按以下顺序验证：

1. `StaticKVCache` 分配、写入、视图和越界单元测试；
2. Legacy tuple Cache 回归测试；
3. Stage 1/Stage 2 的 greedy token 逐 token 一致性；
4. batch=1/2/4 的快速 Decode smoke；
5. 组合 Stage 1+2+3 后再运行完整 Decode 和六套件 benchmark。

## 11. Stage 2.1 Profiler 结论

在 RTX 2080 Ti 上使用相同的 batch=2、32-token Decode 分别 profile Legacy tuple Cache 和 Static Cache。

```text
Legacy Cache:
  aten::cat Self CUDA:       9.189 ms
  aten::copy_ Self CUDA:    21.230 ms
  Self CUDA total:         194.895 ms

Static Cache:
  aten::cat Self CUDA:       4.749 ms
  aten::copy_ Self CUDA:    25.837 ms
  Self CUDA total:         194.496 ms
```

关键观察：

1. Static Cache 确实删除了 K/V 历史扩展的 `cat`，剩余 `cat` 主要来自 RoPE 等其他路径；
2. profiler 中没有发现可见的 SDPA `clone/contiguous` 热点，因此“非连续 view 导致 SDPA 整段隐式复制”不是当前主要证据；
3. Static Cache 每层、每 token 需要把新投影的 K/V `copy_` 到预分配 slot，32-token 工作负载新增约 1536 次 copy；
4. 删除历史 `cat` 节省的 CUDA 时间，被当前 token slot copy 基本抵消；
5. 两条路径的 CUDA 总时间几乎相同，与完整 benchmark 未见显著提升一致。

## 12. Stage 2.1 Validation Fast Path 实验（已撤回）

Static Cache 由 `allocate()` 创建时，层数、batch、KV heads、capacity、head dimension、dtype 和 device 已经确认。原实现仍在每个 Decode forward 遍历 24 层验证这些不变元数据，并在每层 `update()` 内重复检查 shape/device/dtype。

实验版为内部分配的 Cache 记录 validation signature：

- 首次分配时确认静态结构；
- 后续 Decode 只比较 signature 和逻辑 length 边界；
- 容量越界仍在每次 slot 写入前检查；
- 外部手工构造、尚未验证的 Cache 仍执行完整结构检查。

完整单测通过，但 profiler 中 CPU 总时间从 1.300s 增加到 1.369s，CUDA 差异仅属波动。该修改没有可确认收益，却会弱化对 Cache 被外部修改后的结构检查，因此已从正式代码撤回。

更大的后续收益需要消除“先生成 K/V 临时 tensor，再 copy 到 Cache slot”的双写入路径，或使用直接消费 Static Cache 的专用 Decode Attention。

## 13. 组合测试最终结论

Static KV Cache 单独没有产生显著吞吐收益，但它为后续 mask 复用、固定 buffer 和 CUDA Graph 提供了结构基础。

当 Stage 2 与 Stage 1、Stage 3A、Stage 3B、Decode `torch.where` 精简和 Fused MLP 组合后，完整六套件结果为：

```text
FINAL SCORE:       87.16 / 100
Decode best TPS:   172.5
Cache metric TPS:  172.1
Decode valid:      1.000
Aggregate valid:   0.958
```

相比 Stage 0.5：

```text
Decode: 149.0 -> 172.5 TPS  (+15.8%)
Cache:  149.6 -> 172.1 TPS  (+15.0%)
Score:   86.49 -> 87.16     (+0.67)
```

显存代价：

```text
Cache growth:       4.95 MB / 100 tokens
Peak extra:         1811 MB
Cache peak saving:  -287.0%
```

预分配和 Fused MLP 提高了初始/额外显存占用，但完整 benchmark 峰值仍显著低于 RTX 2080 Ti 11GB 限制，没有 OOM。

最终结论：Stage 2 作为后续优化的结构基础保留，但不应把完整组合收益单独归因于 Static KV Cache。
