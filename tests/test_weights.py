import unittest
import torch

from toy_qwen.config import QwenToyConfig
from toy_qwen.modeling import QwenToyForCausalLM
from toy_qwen.weights import build_whiteboard_model, load_whiteboard_weights


class WeightTest(unittest.TestCase):
    def test_embedding_lm_projection_and_norms(self):
        model = build_whiteboard_model()
        torch.testing.assert_close(model.model.embed_tokens.weight[0], torch.tensor([2., 0., 0., 0.]))
        torch.testing.assert_close(model.model.embed_tokens.weight[5], torch.tensor([1., 1., 0., 0.]))
        torch.testing.assert_close(model.lm_head.weight[5], torch.tensor([2., 2., -1., -1.]))
        torch.testing.assert_close(model.lm_head.weight[:5], torch.zeros(5, 4))
        layer = model.model.layers[0]
        torch.testing.assert_close(layer.self_attn.q_proj.weight, torch.eye(4))
        torch.testing.assert_close(model.model.norm.weight, torch.ones(4))

    def test_ffn_is_transposed_for_linear(self):
        weight = build_whiteboard_model().model.layers[0].mlp.gate_proj.weight
        self.assertAlmostEqual(weight[0, 0].item(), -0.2252, places=4)
        self.assertAlmostEqual(weight[1, 0].item(), -0.2305, places=4)

    def test_rejects_wrong_shape_before_loading(self):
        other = QwenToyForCausalLM(QwenToyConfig(9, 8, 8, 1, 2, 1))
        with self.assertRaisesRegex(ValueError, "whiteboard.*hidden_size"):
            load_whiteboard_weights(other)
