# Stage 3C：Fused MLP Gate/Up and Projection Experiments

## 1. 背景

Decode profiler 显示线性层 GEMM 占 CUDA 时间的一半以上。对 batch=1/2/4 的小 batch Decode，除了权重带宽，每层多次独立 GEMM kernel launch 也会产生开销。

Qwen MLP 使用 SwiGLU：

```python
down_proj(silu(gate_proj(x)) * up_proj(x))
```

`gate_proj` 和 `up_proj` 具有相同输入和输出尺寸，可以拼接权重后使用一次更宽的 GEMM，然后切分结果。

## 2. 实现

权重加载完成、模型移动到目标设备后，为每层构建非持久 fused weight：

```python
fused_gate_up_weight = torch.cat(
    (gate_proj.weight, up_proj.weight),
    dim=0,
)
```

Forward：

```python
gate_up = F.linear(x, fused_gate_up_weight, fused_gate_up_bias)
gate, up = gate_up.chunk(2, dim=-1)
output = down_proj(F.silu(gate) * up)
```

非持久 buffer 不进入 `state_dict`，因此：

- checkpoint 参数名不变；
- 加载校验不变；
- 教学模型未调用 `prepare_for_inference()` 时仍走原路径；
- 只有真实预训练模型的推理路径启用融合。

## 3. Profiler 结果

RTX 2080 Ti，batch=2、32-token Decode：

| 指标 | 融合前 | Fused Gate/Up | 变化 |
| --- | ---: | ---: | ---: |
| `mm` calls | 3104 | 2336 | -768 |
| `mm` CUDA | 76.337ms | 70.907ms | -7.1% |
| CUDA total | 180.570ms | 177.976ms | -1.4% |

768 次减少恰好对应：

```text
24 layers x 32 decode tokens = 768
```

## 4. Decode Smoke

单次计时的完整 Decode 样本：

| Batch | 融合前 | Fused Gate/Up |
| ---: | ---: | ---: |
| 1 | 43.99 TPS | 45.12 TPS |
| 2 | 87.78 TPS | 84.06 TPS |
| 4 | 174.43 TPS | 175.04 TPS |
| Aggregate | 75.28 TPS | 75.44 TPS |

batch=2 的单次差异属于需要通过正式多次计时消化的波动；完整六套件最终证明组合版 Decode 为 172.5 TPS，valid=1.000。

## 5. Fused QKV 实验（已撤回）

Q/K/V 也可以拼接为一次更宽的 projection。固定 batch=2、32-token profiler 表面上显示：

```text
addmm calls: 2304 -> 768
addmm CUDA:  18.109ms -> 8.189ms
CUDA total:  177.976ms -> 167.509ms
```

但真实完整 Decode smoke 显示 batch scaling 明显回退：

| Batch | Fused MLP | + Fused QKV | 变化 |
| ---: | ---: | ---: | ---: |
| 1 | 45.12 TPS | 46.94 TPS | +4.0% |
| 2 | 84.06 TPS | 76.76 TPS | -8.7% |
| 4 | 175.04 TPS | 139.35 TPS | -20.4% |
| Aggregate | 75.44 TPS | 72.28 TPS | -4.2% |

峰值显存也从 1432MB 增加到 1479MB。融合后的宽 QKV GEMM 在 Turing batch=2/4 上选择了更差的 kernel。

结论：Fused QKV 不能根据单一 profiler shape 验收，已从代码、暂存区和远程版本撤回。

## 6. Native RMSNorm 实验（已撤回）

`F.rms_norm` 在 profiler 中减少了显式 dtype copy：

```text
copy_ calls: 7073 -> 3937
CUDA total:  177.976ms -> 167.611ms
CPU total:   1.064s -> 0.975s
```

但真实权重 Decode 的：

```text
valid = 0.000
```

说明它与当前 Qwen FP32 accumulation/cast 路径不能保证长生成数值等价。该实验已撤回，保留原手写 RMSNorm。

## 7. 最终验收结果

当前最佳组合包含 Fused Gate/Up，不包含 Fused QKV 和 Native RMSNorm。

```text
FINAL SCORE:          87.16 / 100
Long:                 28.50 / 30
Decode:               23.17 / 25, 172.5 TPS, valid=1.000
TTFT:                 16.16 / 20, avg=0.125s, p95=0.409s
Serving:               9.32 / 15, 117.7 TPS, p95=8.867s
Runtime:              10.00 / 10
Mixed best TPS:       66.0
Cache metric TPS:     172.1
Aggregate valid:      0.958
Realism Guard:        OK
```

结果目录：

```text
/root/autodl-tmp/Infer-DaseSS/student_release/results/stage_best_full_20260719
```
