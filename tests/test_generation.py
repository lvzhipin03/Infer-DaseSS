import unittest

import torch

from toy_qwen.generation import greedy_generate
from toy_qwen.weights import build_whiteboard_model


class GreedyGenerationTest(unittest.TestCase):
    def setUp(self):
        self.model = build_whiteboard_model().eval()
        self.input_ids = torch.tensor([[0, 1, 2, 3, 4]])

    def test_prefill_generates_expected_token_and_cache(self):
        result = greedy_generate(self.model, self.input_ids, eos_token_id=None, max_new_tokens=1)

        self.assertEqual(result.generated_ids, (5,))
        self.assertEqual(result.prefill_logits_shape, (1, 5, 9))
        self.assertEqual(result.first_cache_shapes[0], ((1, 1, 5, 4), (1, 1, 5, 4)))
        self.assertEqual(result.last_cache_shapes, result.first_cache_shapes)

    def test_top_k_is_sorted_and_contains_selected_logit(self):
        result = greedy_generate(self.model, self.input_ids, eos_token_id=None, max_new_tokens=1, top_k=4)
        step = result.steps[0]

        self.assertEqual(step.token_id, step.top_ids[0])
        self.assertAlmostEqual(step.selected_logit, step.top_logits[0], places=6)
        self.assertEqual(tuple(sorted(step.top_logits, reverse=True)), step.top_logits)

    def test_eos_stops_after_first_token(self):
        result = greedy_generate(self.model, self.input_ids, eos_token_id=5, max_new_tokens=3)

        self.assertEqual(result.generated_ids, (5,))
        self.assertEqual(len(result.steps), 1)

    def test_cached_second_step_matches_full_forward_top_k(self):
        result = greedy_generate(self.model, self.input_ids, eos_token_id=None, max_new_tokens=2, top_k=4)
        full_ids = torch.cat((self.input_ids, torch.tensor([[result.generated_ids[0]]])), dim=1)

        with torch.no_grad():
            full_logits = self.model(full_ids, use_cache=False).logits[:, -1, :]
        values, ids = torch.topk(full_logits, k=4, dim=-1)

        self.assertEqual(result.steps[1].top_ids, tuple(ids[0].tolist()))
        torch.testing.assert_close(torch.tensor(result.steps[1].top_logits), values[0])
        self.assertEqual(result.last_cache_shapes[0][0][2], 6)

    def test_rejects_zero_tokens(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            greedy_generate(self.model, self.input_ids, eos_token_id=None, max_new_tokens=0)


if __name__ == "__main__":
    unittest.main()
