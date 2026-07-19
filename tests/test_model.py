import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
import torch.nn.functional as F

from toy_qwen.config import QwenToyConfig, whiteboard_config
from toy_qwen.modeling import QwenToyForCausalLM, QwenToyMLP
from toy_qwen.weights import build_whiteboard_model


class ModelTest(unittest.TestCase):
    def test_mlp_matches_swiglu(self):
        mlp = QwenToyMLP(whiteboard_config())
        x = torch.randn(1, 2, 4)
        torch.testing.assert_close(mlp(x), mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x)))

    def test_fused_mlp_matches_unfused_and_preserves_checkpoint_keys(self):
        mlp = QwenToyMLP(whiteboard_config()).eval()
        x = torch.randn(2, 3, 4)
        expected = mlp(x)
        checkpoint_keys = set(mlp.state_dict())

        mlp.prepare_for_inference()

        torch.testing.assert_close(mlp(x), expected)
        self.assertEqual(set(mlp.state_dict()), checkpoint_keys)

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

    def test_num_logits_to_keep_projects_only_last_positions(self):
        model = build_whiteboard_model().eval()
        ids = torch.tensor([[0, 1, 2, 3, 4]])
        full = model(ids, use_cache=False).logits
        last = model(ids, use_cache=False, num_logits_to_keep=1).logits
        last_two = model(ids, use_cache=False, num_logits_to_keep=2).logits

        self.assertEqual(last.shape, (1, 1, 9))
        self.assertEqual(last_two.shape, (1, 2, 9))
        torch.testing.assert_close(last, full[:, -1:])
        torch.testing.assert_close(last_two, full[:, -2:])

    def test_num_logits_to_keep_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "num_logits_to_keep"):
            build_whiteboard_model()(torch.tensor([[0]]), num_logits_to_keep=0)

    def test_prefill_rejects_token_ids_outside_vocabulary(self):
        model = build_whiteboard_model()
        for invalid_id in (-1, model.config.vocab_size):
            with self.subTest(token_id=invalid_id):
                with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
                    model(torch.tensor([[invalid_id]]), use_cache=True)

    def test_cached_decode_skips_redundant_input_id_range_validation(self):
        model = build_whiteboard_model().eval()
        with torch.no_grad():
            prefill = model(torch.tensor([[0, 1]]), use_cache=True)
            with patch("toy_qwen.modeling._validate_input_ids") as validate:
                model(
                    torch.tensor([[2]]),
                    past_key_values=prefill.past_key_values,
                    use_cache=True,
                )

        validate.assert_not_called()
