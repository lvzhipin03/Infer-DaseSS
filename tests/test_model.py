import unittest
from dataclasses import replace
import torch
import torch.nn.functional as F

from toy_qwen.config import QwenToyConfig, whiteboard_config
from toy_qwen.modeling import QwenToyForCausalLM, QwenToyMLP


class ModelTest(unittest.TestCase):
    def test_mlp_matches_swiglu(self):
        mlp = QwenToyMLP(whiteboard_config())
        x = torch.randn(1, 2, 4)
        torch.testing.assert_close(mlp(x), mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x)))

    def test_model_shapes_cache_and_paths(self):
        model = QwenToyForCausalLM(whiteboard_config())
        output = model(torch.tensor([[0, 1, 2, 3, 4]]), use_cache=True, output_hidden_states=True, trace_shapes=True)
        self.assertEqual(output.logits.shape, (1, 5, 9))
        self.assertEqual(output.past_key_values[0][0].shape, (1, 1, 5, 4))
        self.assertEqual(len(output.hidden_states), 3)
        names = set(model.state_dict())
        self.assertIn("model.layers.0.self_attn.q_proj.weight", names)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", names)

    def test_tied_weights_share_parameter(self):
        model = QwenToyForCausalLM(replace(whiteboard_config(), tie_word_embeddings=True))
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    def test_14q_2kv_forward(self):
        config = QwenToyConfig(9, 28, 152, 2, 14, 2)
        self.assertEqual(QwenToyForCausalLM(config)(torch.tensor([[0, 1, 2]])).logits.shape, (1, 3, 9))
