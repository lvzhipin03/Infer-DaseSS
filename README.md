# Toy Qwen2.5 白板推理器

这是一个只依赖 PyTorch 的 Qwen2 风格最小 causal LM。它使用手写白板权重在 CPU
上执行完整的 Embedding、RMSNorm、RoPE、causal GQA、SwiGLU、残差、final
RMSNorm 和 LM Head 链路。

## 运行

```bash
python3 whiteboard_llm_inference.py --prompt 中国首都是 --trace-shapes
```

最后一行应为：

```text
next token: 北
```

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 白板配置

默认配置为 9 个字符、hidden size 4、SwiGLU intermediate size 8、1 层、1 个
query head 和 1 个 KV head。`1Q/1KV` 是通用 GQA 的退化形式；同一个实现另有
`14Q/2KV` 参数测试。

白板资料用 `[in,out]` 记录线性矩阵，加载器会转置为 PyTorch 的
`nn.Linear.weight=[out,in]`。LM Head 扩成完整 `[9,4]`：输入字符对应行置零，
“北、京、上、海”保留参考权重。

## 张量流

```text
input_ids [B,T]
→ embedding [B,T,H]
→ Q [B,Nq,T,D], K/V [B,Nkv,T,D]
→ attention scores [B,Nq,T,S]
→ decoder hidden [B,T,H]
→ logits [B,T,V]
```

KV Cache 保存每层 `[B,Nkv,cached_length,D]` 的 K/V。测试验证完整输入与
prefill“`中国首都`”再 cached decode“`是`”的最终 logits 一致。

## 扩展为完整 Qwen2.5-0.5B

`configs/qwen2_5_0_5b.json` 保留官方结构参数。模块路径使用 Qwen/Hugging Face
风格，例如 `model.layers.0.self_attn.q_proj.weight`。QKV bias、tied embedding、
层数、head 数和维度均由配置控制。

首版不包含 Hugging Face tokenizer、safetensors 分片加载、量化、滑动窗口或完整
0.5B CPU 内存优化；这些应作为独立 adapter 添加，不需要修改当前 forward 数学链路。
