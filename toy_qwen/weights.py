import torch

from .config import whiteboard_config
from .modeling import QwenToyForCausalLM

EMBEDDING = torch.tensor([
    [2., 0., 0., 0.], [0., 2., 0., 0.], [0., 0., 2., 0.],
    [0., 0., 0., 2.], [1., 1., -1., -1.], [1., 1., 0., 0.],
    [.8, .8, 0., 0.], [0., 0., 1., 1.], [0., 0., .8, .8],
])
LM_CANDIDATES = torch.tensor([
    [2., 2., -1., -1.], [1.2, 1.2, -.5, -.5],
    [-1., -1., 2., 2.], [-.8, -.8, 1.2, 1.2],
])


def load_whiteboard_weights(model: QwenToyForCausalLM) -> None:
    config = model.config
    expected = (config.vocab_size, config.hidden_size, config.intermediate_size,
                config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads)
    if expected != (9, 4, 8, 1, 1, 1):
        raise ValueError("whiteboard weights require vocab_size=9, hidden_size=4, intermediate_size=8, one layer and 1/1 heads")
    generator = torch.Generator(device="cpu").manual_seed(0)
    w_gate = torch.randn(4, 8, generator=generator) * .2
    w_up = torch.randn(4, 8, generator=generator) * .2
    w_down = torch.randn(8, 4, generator=generator) * .2
    lm_head = torch.zeros(9, 4)
    lm_head[5:9] = LM_CANDIDATES
    layer = model.model.layers[0]
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(EMBEDDING)
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                           layer.self_attn.v_proj, layer.self_attn.o_proj):
            projection.weight.copy_(torch.eye(4))
        layer.input_layernorm.weight.fill_(1.)
        layer.post_attention_layernorm.weight.fill_(1.)
        model.model.norm.weight.fill_(1.)
        layer.mlp.gate_proj.weight.copy_(w_gate.T)
        layer.mlp.up_proj.weight.copy_(w_up.T)
        layer.mlp.down_proj.weight.copy_(w_down.T)
        model.lm_head.weight.copy_(lm_head)


def build_whiteboard_model() -> QwenToyForCausalLM:
    model = QwenToyForCausalLM(whiteboard_config())
    load_whiteboard_weights(model)
    return model
