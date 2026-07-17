import unittest
import torch

from toy_qwen.cache import cache_length, validate_past_key_values
from toy_qwen.config import whiteboard_config
from toy_qwen.modeling import QwenToyRMSNorm, apply_rotary_pos_emb, rotate_half


class PrimitiveTest(unittest.TestCase):
    def test_rmsnorm_matches_formula(self):
        x = torch.tensor([[[1., 2., 3., 4.]]])
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(QwenToyRMSNorm(4)(x), expected)

    def test_qwen_rotate_half(self):
        torch.testing.assert_close(rotate_half(torch.tensor([1., 2., 3., 4.])), torch.tensor([-3., -4., 1., 2.]))

    def test_rope_preserves_norm(self):
        q = torch.tensor([[[[1., 2., 3., 4.]]]])
        rotated, _ = apply_rotary_pos_emb(q, q, torch.zeros_like(q), torch.ones_like(q))
        torch.testing.assert_close(rotated.square().sum(-1), q.square().sum(-1))

    def test_cache_validation(self):
        valid = ((torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),)
        self.assertEqual(cache_length(valid), 3)
        validate_past_key_values(valid, whiteboard_config(), 1)
        invalid = ((torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),)
        with self.assertRaisesRegex(ValueError, "KV heads"):
            validate_past_key_values(invalid, whiteboard_config(), 1)
