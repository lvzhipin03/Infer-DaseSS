# Stage 3A：Preallocated Generation Attention Mask

## 1. 阶段目标

Stage 3A 的目标是移除 batch Decode 循环中每生成一个 token 都扩展 `full_attention_mask` 的 `torch.cat`，改为生成开始时一次预分配 mask buffer，后续原地写入新 token 的有效位。

本阶段只优化 generation 层的 mask 存储和扩展方式，不改变 Attention 内部的 causal/padding mask 语义。

## 2. 前置阶段

Stage 1 已移除逐 token 的输出 D2H copy 和部分 Python/同步开销。

Stage 2 已将 K/V Cache 从逐 token `torch.cat` 改为预分配 buffer 和原地 slot 写入。但当前 batch 生成路径仍然每步扩展 Attention Mask，所以 Decode hot path 中仍有一处长度递增的 `torch.cat`。

Stage 3A 与 Stage 2 使用相同的“物理容量 + 逻辑长度”设计，使 Cache 和 mask 的有效长度保持一致。

## 3. 当前实现

当前代码位于：

```text
toy_qwen/generation.py
```

每个 Decode step 结束后执行：

```python
full_attention_mask = torch.cat(
    (
        full_attention_mask,
        torch.ones(
            (batch_size, 1),
            dtype=full_attention_mask.dtype,
            device=full_attention_mask.device,
        ),
    ),
    dim=1,
)
```

这段代码每步都会：

1. 创建一个 `[batch_size, 1]` 的 `ones` tensor；
2. 分配一个比原 mask 长一列的新 tensor；
3. 复制所有历史 mask 元素；
4. 复制新的有效列；
5. 丢弃原 tensor，并改变 mask 地址。

Mask 比 K/V Cache 小很多，所以 Stage 3A 的单独收益预计小于 Stage 2。但它仍是不必要的逐 token allocation/copy，也会阻碍后续对稳定 tensor 地址的利用。

## 4. 左填充 Mask 语义

对于变长 batch，prompt mask 可能是：

```text
短 prompt: [0, 0, 0, 1, 1]
长 prompt: [1, 1, 1, 1, 1]
```

每个新生成 token 对所有请求都是有效 token，因此第一步 Decode 后：

```text
短 prompt: [0, 0, 0, 1, 1, 1]
长 prompt: [1, 1, 1, 1, 1, 1]
```

Stage 3A 必须完整保留 prompt 的左 padding 区域，只能把新增 Decode slot 写为 1，不能将整个 buffer 初始化为 1。

## 5. 预分配设计

与 Stage 2 相同，mask capacity 为：

```text
capacity = prompt_width + max_new_tokens - 1
```

生成开始时：

```python
attention_mask_buffer = torch.zeros(
    (batch_size, capacity),
    dtype=batch.attention_mask.dtype,
    device=batch.attention_mask.device,
)
attention_mask_buffer[:, :prompt_width].copy_(batch.attention_mask)
```

第 `index` 个 forward 只向模型传递当前逻辑视图：

```python
current_mask = attention_mask_buffer[:, :prompt_width + index]
```

如果仍需要下一步 Decode，则原地激活新 slot：

```python
attention_mask_buffer[:, prompt_width + index] = 1
```

slice 只创建 view，不复制 mask 内容。底层 buffer 的地址在整次 generation 中保持不变。

## 6. 与 Static KV Cache 的长度对齐

在第 `index` 个 forward 开始时：

```text
StaticKVCache.length = prompt_width + index - 1   # index > 0
current input length = 1                          # index > 0
attention mask len  = prompt_width + index
```

因此：

```text
attention mask len = past cache length + current input length
```

这与 `_allowed_attention_mask()` 的现有 shape 约束一致。

Prefill（`index=0`）时：

```text
cache length        = 0
current input width = prompt_width
attention mask len  = prompt_width
```

也保持同一不变式。

## 7. 为什么不在 Stage 3A 修改每层 Allowed Mask

Attention 内部当前每层调用：

```python
_allowed_attention_mask(...)
```

它会组合：

- causal mask；
- padding key mask；
- Prefill/Decode 的 `past_length`；
- batch 和 query/key length。

复用或简化这部分可以减少 24 层重复 mask 构造，但会直接影响 Attention 数学和 Stage 0.5 的 left-padding correctness。

因此将其留给 Stage 3B。Stage 3A 只替换 generation loop 中的存储扩展方式，使性能和正确性变化更容易归因。

## 8. 正确性边界

Stage 3A 必须保证：

1. Prefill 看到的 mask 与原 `batch.attention_mask` 完全相同；
2. 每个 Decode forward 看到的 mask 长度与当前 key length 相同；
3. prompt 的左 padding 列始终保持 0；
4. 每个新生成 token 的 slot 为 1；
5. batch=1/2/4 的 greedy token IDs 与 Stage 2 一致；
6. `max_new_tokens=1` 时 capacity 等于 prompt width，不写入额外 slot；
7. 上下文越界检查仍在分配前生效。

## 9. 预期收益

Stage 3A 的直接性能收益预计较小，主要价值是：

- 移除 generation loop 中最后一处递增长度的 `torch.cat`；
- 减少逐 token 的小 tensor allocation 和历史 mask copy；
- 使 Attention Mask 和 Static KV Cache 共享相同 capacity；
- 保持 mask buffer 地址稳定；
- 为 Stage 3B、CUDA Graph 和后续调度优化提供结构基础。

## 10. 本阶段不包含

Stage 3A 不包含：

- 跨层复用 allowed attention mask；
- Decode 专用 causal mask 快速路径；
- 无 padding batch 的 mask 省略；
- Paged KV Cache；
- Prefix Cache；
- Continuous Batching；
- CUDA Graph。

## 11. 后续验证计划

Stage 3A 实现后，建议将 Stage 1+2+3A 作为一个组合进行验证：

1. mask 容量、prompt copy 和逐步 slot 测试；
2. 极端 left-padding batch 的 token 一致性；
3. `max_new_tokens=1/2/64` 边界；
4. 本地完整单元测试；
5. 远程 batch=1/4 快速 Decode smoke；
6. 组合优化完成后再运行正式 benchmark。

## 12. 组合验收结果

Stage 3A 作为低风险结构优化保留。它单独收益较小，但删除了 generation loop 中 mask 递增 `torch.cat`，并为 Stage 3B 跨层 mask 复用提供了稳定 buffer。

当前最佳组合的标准六套件：

```text
FINAL SCORE:     87.16 / 100
Decode TPS:      172.5, valid=1.000
Serving TPS:     117.7, valid=0.917
Mixed TPS:       66.0, valid=0.917
Cache TPS:       172.1, valid=0.917
OOM / timeout:   0 / 0
```

结果不能单独归因于 Stage 3A；它是 Stage 1+2+3A+3B+Decode 热路径精简+Fused MLP 的组合结果。
