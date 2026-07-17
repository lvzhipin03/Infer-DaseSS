# Toy Qwen2.5 最小推理器设计

## 目标与验收

使用纯 PyTorch 在 CPU、float32 下实现 decoder-only causal LM。首版加载
`references/whiteboard_weights_summary.md` 中的白板权重，输入“`中国首都是`”后，
通过完整 forward 和 greedy argmax 输出“`北`”。实现不得按字符串分支、覆盖 logits
或训练权重。

命令行验收：

```bash
python whiteboard_llm_inference.py --prompt 中国首都是 --trace-shapes
```

最后一行必须是：

```text
next token: 北
```

## 架构决策

模型采用与 Hugging Face Qwen2 一致的模块边界和参数路径：

```text
QwenToyForCausalLM
├── model.embed_tokens
├── model.rotary_emb
├── model.layers[i]
│   ├── input_layernorm
│   ├── self_attn.{q_proj,k_proj,v_proj,o_proj}
│   ├── post_attention_layernorm
│   └── mlp.{gate_proj,up_proj,down_proj}
├── model.norm
└── lm_head
```

每层执行 pre-norm attention、第一条残差、pre-norm SwiGLU 和第二条残差；所有层后
执行 final RMSNorm，再由无 bias LM Head 得到完整词表 logits。Q/K 使用 RoPE，注意力
使用 causal mask，K/V 通过通用 `repeat_kv` 支持 GQA。

## 参数预设

首版 `whiteboard_toy`：

| 参数 | 值 |
|---|---:|
| vocab_size | 9 |
| hidden_size | 4 |
| intermediate_size | 8 |
| num_hidden_layers | 1 |
| num_attention_heads | 1 |
| num_key_value_heads | 1 |
| head_dim | 4（派生值） |
| max_position_embeddings | 64 |
| rope_theta | 1,000,000 |
| rms_norm_eps | 1e-6 |
| attention_dropout | 0.0 |
| attention_bias | false |
| mlp_bias | false |
| lm_head_bias | false |
| tie_word_embeddings | false |
| use_cache | true |
| use_sliding_window | false |

未来 `qwen2_5_0_5b` 预设保留官方结构值：`vocab_size=151936`、
`hidden_size=896`、`intermediate_size=4864`、`num_hidden_layers=24`、
`num_attention_heads=14`、`num_key_value_heads=2`、
`max_position_embeddings=32768`、`rope_theta=1000000`、
`rms_norm_eps=1e-6`、`attention_bias=true`、`tie_word_embeddings=true`。
完整预设只用于配置兼容和未来加载，不要求首版在普通 CPU 上运行 0.5B 权重。

配置必须检查 hidden/head 整除、偶数 head dimension、Q heads/KV heads 整除、正的
位置上限和合法 token ID。运行 dtype 不写死在架构中；白板预设固定使用 float32。

## Tokenizer 与 ID

toy 字符词表顺序固定为：

```text
中、国、首、都、是、北、京、上、海
```

模型使用连续 ID `0..8`。白板资料中的 `10,20,...,90` 只作为 `legacy_ids` trace
展示，不进入 `nn.Embedding`。未知字符立即抛出包含字符和位置的 `ValueError`。
未来完整 tokenizer 通过相同的 `encode/decode` 边界接入。

## 张量和数据流

```text
input_ids       [B,T]
embedding       [B,T,H]
query           [B,Nq,T,D]
key/value       [B,Nkv,T,D]
repeated K/V    [B,Nq,S,D]
attention score [B,Nq,T,S]
MLP gate/up     [B,T,I]
hidden          [B,T,H]
logits          [B,T,V]
```

`D=H/Nq`，`S=past_length+T`。attention softmax 在 float32 中计算后转回 value
dtype。RMSNorm 同样在 float32 中计算方差。

## KV Cache

`past_key_values` 是逐层 `(key,value)` tuple；每个张量 shape 为
`[B,Nkv,cached_length,D]`。forward 在提供 cache 时从 cached length 生成 position
IDs，拼接当前 K/V，并返回新 cache。必须验证一次完整 forward 与
prefill“`中国首都`”后 cached decode“`是`”的最后位置 logits 一致。

## 白板权重解释

原文数值均保持不变：embedding 为 `[9,4]`；Q/K/V/O 是单位矩阵；两组 RMSNorm
gamma 为全 1；FFN 由 `torch.manual_seed(0)` 后依次生成三个 `randn * 0.2` 矩阵。

原文采用 `x @ W` 和 `[in,out]` 记法，而 `nn.Linear.weight` 使用 `[out,in]`，加载器
必须显式转置，并逐项校验 shape。PPT 中存在但汇总表未列出的 final RMSNorm gamma
补为全 1。

LM Head 扩展为标准 `[9,4]`：dense ID 5..8 分别填入“北、京、上、海”的四行原始
权重，ID 0..4 行置零。这样不会改变已给权重，同时保证 logits 覆盖完整输入词表。

带 RoPE、causal attention 和 final RMSNorm 的参考候选 logits 为：

```text
北  6.199933
京  3.560251
上 -5.495592
海 -3.757640
```

## 接口

```python
@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    past_key_values: PastKeyValues | None
    hidden_states: tuple[torch.Tensor, ...] | None
    trace: dict[str, tuple[int, ...]] | None

@dataclass
class Prediction:
    token: str
    token_id: int
    logit: float
    runner_up_token: str
    runner_up_logit: float
    logits: torch.Tensor
```

模型 forward 接收 `input_ids`、可选 mask、position IDs、cache、`use_cache`、
`output_hidden_states` 和 `trace_shapes`。`predict_next_token` 负责 tokenizer、no-grad、
最后位置选择和 greedy argmax，不更改 logits。

## 错误处理

- 配置非法时在模块构建前抛出 `ValueError`。
- 未知字符报告字符与文本位置。
- 空 prompt、非二维 `input_ids`、越界 ID、超长序列报错。
- cache 层数或 B/Nkv/D 不匹配报错。
- 白板权重目标 shape 不匹配时报出参数名、期望和实际 shape。

## 文件边界

```text
toy_qwen/config.py      配置、预设和校验
toy_qwen/tokenizer.py   toy 字符 tokenizer 和 legacy trace
toy_qwen/cache.py       cache 类型、长度和校验
toy_qwen/modeling.py    RMSNorm、RoPE、GQA、MLP、decoder、LM
toy_qwen/weights.py     白板权重构造与加载
toy_qwen/inference.py   next-token 推理、排名和 shape trace
whiteboard_llm_inference.py  CLI
configs/*.json          两套配置快照
tests/                  分层和端到端测试
README.md               运行、数学链路和扩展说明
```

## 测试策略

测试覆盖配置验证、tokenizer、RMSNorm、RoPE、causal mask、14Q/2KV 的 repeat、
SwiGLU、两条残差、权重数值与转置、cache 等价性、完整 shape trace、端到端参考
logits 和 CLI。防硬编码测试交换“北/京”的 LM Head 行并确认预测随权重变化。

## 非目标

首版不实现训练、采样策略、batch padding、滑动窗口、量化、safetensors 分片读取、
Hugging Face tokenizer 或在普通 CPU 上执行完整 0.5B 权重。这些能力通过独立 adapter
和现有配置/参数命名边界扩展，不进入教学核心。
