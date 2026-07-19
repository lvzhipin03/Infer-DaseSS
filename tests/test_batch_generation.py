import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

import toy_qwen.generation as generation_module
from toy_qwen.config import whiteboard_config
from toy_qwen.generation import PaddedBatch, batched_greedy_generate, left_pad_token_ids
from toy_qwen.modeling import QwenToyForCausalLM
from toy_qwen.weights import build_whiteboard_model, load_whiteboard_weights


class LeftPaddingTest(unittest.TestCase):
    def test_left_padding_builds_ids_mask_positions_and_lengths(self):
        batch = left_pad_token_ids(
            [[1, 2], [3, 4, 5]], pad_token_id=0, device="cpu"
        )

        self.assertEqual(batch.input_ids.tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(batch.attention_mask.tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(batch.position_ids.tolist(), [[0, 0, 1], [0, 1, 2]])
        self.assertEqual(batch.lengths.tolist(), [2, 3])

    def test_rejects_no_sequences(self):
        with self.assertRaisesRegex(ValueError, "sequence"):
            left_pad_token_ids([], pad_token_id=0, device="cpu")

    def test_rejects_empty_row(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            left_pad_token_ids([[1], []], pad_token_id=0, device="cpu")

    def test_rejects_missing_pad_token(self):
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            left_pad_token_ids([[1]], pad_token_id=None, device="cpu")

    def test_rejects_non_integer_token(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            left_pad_token_ids([[1, "2"]], pad_token_id=0, device="cpu")


class BatchedGenerationTest(unittest.TestCase):
    def setUp(self):
        self.model = build_whiteboard_model().eval().set_attention_implementation("sdpa")

    def test_identical_prompts_match_batch_one_and_keep_requested_length(self):
        prompt = [0, 1, 2, 3, 4]
        paired = batched_greedy_generate(
            self.model,
            left_pad_token_ids([prompt, prompt], pad_token_id=8, device="cpu"),
            max_new_tokens=2,
        )
        single = batched_greedy_generate(
            self.model,
            left_pad_token_ids([prompt], pad_token_id=8, device="cpu"),
            max_new_tokens=2,
        )

        self.assertEqual(paired.generated_ids, (single.generated_ids[0],) * 2)
        self.assertEqual(tuple(map(len, paired.generated_ids)), (2, 2))
        self.assertEqual(paired.prefill_logits_shape, (2, 1, 9))
        self.assertEqual(paired.first_cache_shapes[0][0], (2, 1, 5, 4))
        self.assertEqual(paired.last_cache_shapes[0][0], (2, 1, 6, 4))

    def test_variable_length_rows_match_independent_unpadded_runs(self):
        prompts = [[0, 1], [2, 3, 4]]
        combined = batched_greedy_generate(
            self.model,
            left_pad_token_ids(prompts, pad_token_id=8, device="cpu"),
            max_new_tokens=2,
        )
        independent = tuple(
            batched_greedy_generate(
                self.model,
                left_pad_token_ids([prompt], pad_token_id=8, device="cpu"),
                max_new_tokens=2,
            ).generated_ids[0]
            for prompt in prompts
        )

        self.assertEqual(combined.generated_ids, independent)

    def test_prefill_projects_only_the_final_hidden_position(self):
        lm_head_lengths = []
        handle = self.model.lm_head.register_forward_pre_hook(
            lambda _module, args: lm_head_lengths.append(args[0].shape[1])
        )
        try:
            batched_greedy_generate(
                self.model,
                left_pad_token_ids([[0, 1], [2, 3, 4]], pad_token_id=8, device="cpu"),
                max_new_tokens=3,
            )
        finally:
            handle.remove()

        self.assertEqual(lm_head_lengths, [1, 1, 1])

    def test_cache_shapes_are_collected_only_for_first_and_last_steps(self):
        batch = left_pad_token_ids(
            [[0, 1], [2, 3, 4]], pad_token_id=8, device="cpu"
        )

        with patch(
            "toy_qwen.generation._cache_shapes",
            wraps=generation_module._cache_shapes,
        ) as cache_shapes:
            result = batched_greedy_generate(
                self.model,
                batch,
                max_new_tokens=4,
            )

        self.assertEqual(cache_shapes.call_count, 2)
        self.assertEqual(result.first_cache_shapes[0][0], (2, 1, 3, 4))
        self.assertEqual(result.last_cache_shapes[0][0], (2, 1, 6, 4))

    def test_cached_second_token_matches_uncached_full_forward(self):
        prompts = [[0, 1], [2, 3, 4]]
        result = batched_greedy_generate(
            self.model,
            left_pad_token_ids(prompts, pad_token_id=8, device="cpu"),
            max_new_tokens=2,
        )

        expected = []
        with torch.no_grad():
            for prompt, generated in zip(prompts, result.generated_ids):
                full_ids = torch.tensor([prompt + [generated[0]]])
                logits = self.model(full_ids, use_cache=False).logits[:, -1]
                expected.append(int(logits.argmax(dim=-1).item()))
        self.assertEqual(tuple(row[1] for row in result.generated_ids), tuple(expected))

    def test_rejects_non_positive_token_count(self):
        batch = left_pad_token_ids([[0]], pad_token_id=8, device="cpu")
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            batched_greedy_generate(self.model, batch, max_new_tokens=0)

    def test_rejects_malformed_batch_shapes(self):
        batch = left_pad_token_ids([[0], [1, 2]], pad_token_id=8, device="cpu")
        malformed = PaddedBatch(
            batch.input_ids,
            batch.attention_mask[:, :1],
            batch.position_ids,
            batch.lengths,
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            batched_greedy_generate(self.model, malformed, max_new_tokens=1)

    def test_rejects_right_padding_and_empty_valid_rows(self):
        right_padded = PaddedBatch(
            torch.tensor([[0, 8]]),
            torch.tensor([[1, 0]]),
            torch.tensor([[0, 0]]),
            torch.tensor([1]),
        )
        with self.assertRaisesRegex(ValueError, "left-padded"):
            batched_greedy_generate(self.model, right_padded, max_new_tokens=1)

        empty = PaddedBatch(
            torch.tensor([[8]]),
            torch.tensor([[0]]),
            torch.tensor([[0]]),
            torch.tensor([0]),
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            batched_greedy_generate(self.model, empty, max_new_tokens=1)

    def test_rejects_context_overflow_before_forward(self):
        model = QwenToyForCausalLM(
            replace(whiteboard_config(), max_position_embeddings=3)
        ).set_attention_implementation("sdpa")
        load_whiteboard_weights(model)
        batch = left_pad_token_ids([[0, 1, 2]], pad_token_id=8, device="cpu")

        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            batched_greedy_generate(model, batch, max_new_tokens=2)


if __name__ == "__main__":
    unittest.main()
