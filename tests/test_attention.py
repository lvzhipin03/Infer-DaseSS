import copy
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F

from toy_qwen.config import QwenToyConfig, whiteboard_config
from toy_qwen.modeling import QwenToyAttention, QwenToyForCausalLM, QwenToyRotaryEmbedding, repeat_kv
from toy_qwen.weights import build_whiteboard_model


class AttentionTest(unittest.TestCase):
    def test_repeat_kv_14_query_2_kv(self):
        source = torch.arange(12.).reshape(1, 2, 3, 2)
        repeated = repeat_kv(source, 7)
        self.assertEqual(repeated.shape, (1, 14, 3, 2))
        torch.testing.assert_close(repeated[:, 0], source[:, 0])
        torch.testing.assert_close(repeated[:, 7], source[:, 1])

    def test_future_token_does_not_change_first_output(self):
        config = whiteboard_config()
        attention = QwenToyAttention(config, 0).eval()
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
            projection.weight.data.copy_(torch.eye(4))
        x = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.]]])
        changed = x.clone(); changed[:, 1] = 9.
        rope = QwenToyRotaryEmbedding(config)
        position = torch.tensor([[0, 1]])
        embeddings = rope(position, x.dtype)
        original, _, _ = attention(x, embeddings)
        modified, _, _ = attention(changed, embeddings)
        torch.testing.assert_close(original[:, 0], modified[:, 0])

    def test_trace_and_cache_shapes(self):
        config = whiteboard_config()
        attention = QwenToyAttention(config, 0)
        x = torch.zeros(1, 5, 4)
        rope = QwenToyRotaryEmbedding(config)
        output, cache, trace = attention(x, rope(torch.arange(5).unsqueeze(0), x.dtype), use_cache=True)
        self.assertEqual(output.shape, (1, 5, 4))
        self.assertEqual(cache[0].shape, (1, 1, 5, 4))
        self.assertEqual(trace["attention_scores"], (1, 1, 5, 5))

    def test_rejects_unknown_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "eager.*sdpa"):
            build_whiteboard_model().set_attention_implementation("flash_magic")

    def test_sdpa_matches_eager(self):
        eager = build_whiteboard_model().eval()
        sdpa = copy.deepcopy(eager).set_attention_implementation("sdpa")
        ids = torch.tensor([[0, 1, 2, 3, 4]])
        with torch.no_grad():
            expected = eager(ids, use_cache=False).logits
            actual = sdpa(ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_sdpa_matches_eager_with_14_query_and_2_kv_heads(self):
        torch.manual_seed(7)
        config = QwenToyConfig(17, 28, 56, 2, 14, 2)
        eager = QwenToyForCausalLM(config).eval()
        sdpa = copy.deepcopy(eager).set_attention_implementation("sdpa")
        ids = torch.tensor([[0, 1, 2, 3]])

        with torch.no_grad():
            expected = eager(ids, use_cache=True, trace_shapes=True)
            actual = sdpa(ids, use_cache=True, trace_shapes=True)

        torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=1e-5)
        self.assertEqual(actual.trace["layer_0.attention_scores"], (1, 14, 4, 4))
        self.assertEqual(actual.past_key_values[0][0].shape, (1, 2, 4, 2))

    def test_sdpa_capability_errors_repeat_gqa_keys_and_values(self):
        config = QwenToyConfig(17, 28, 56, 1, 14, 2)
        ids = torch.tensor([[0, 1, 2, 3]])
        native_sdpa = F.scaled_dot_product_attention

        for capability_error in (
            TypeError("unexpected keyword argument 'enable_gqa'"),
            RuntimeError("No available kernel. Aborting execution."),
        ):
            with self.subTest(error=type(capability_error).__name__):
                torch.manual_seed(11)
                eager = QwenToyForCausalLM(config).eval()
                sdpa = copy.deepcopy(eager).set_attention_implementation("sdpa")
                calls = []

                def simulated_backend(query, key, value, **kwargs):
                    calls.append((query, key, value, kwargs))
                    if len(calls) == 1:
                        raise capability_error
                    return native_sdpa(query, key, value, **kwargs)

                with torch.no_grad():
                    expected = eager(ids, use_cache=False).logits
                    with patch(
                        "toy_qwen.modeling.F.scaled_dot_product_attention",
                        side_effect=simulated_backend,
                    ):
                        actual = sdpa(ids, use_cache=False).logits

                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][1].shape[1], 2)
                self.assertTrue(calls[0][3]["enable_gqa"])
                self.assertEqual(calls[1][1].shape[1], 14)
                self.assertEqual(calls[1][2].shape[1], 14)
                self.assertNotIn("enable_gqa", calls[1][3])
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_sdpa_future_token_does_not_change_first_output(self):
        model = build_whiteboard_model().eval().set_attention_implementation("sdpa")
        original = torch.tensor([[0, 1]])
        changed = torch.tensor([[0, 8]])
        with torch.no_grad():
            original_logits = model(original, use_cache=False).logits
            changed_logits = model(changed, use_cache=False).logits
        torch.testing.assert_close(original_logits[:, 0], changed_logits[:, 0], rtol=1e-5, atol=1e-5)

    def test_sdpa_left_padding_matches_unpadded_final_token(self):
        eager = build_whiteboard_model().eval()
        sdpa = copy.deepcopy(eager).set_attention_implementation("sdpa")
        padded_ids = torch.tensor([[8, 0, 1], [2, 3, 4]])
        attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        position_ids = torch.tensor([[0, 0, 1], [0, 1, 2]])

        with torch.no_grad():
            padded = sdpa(
                padded_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            ).logits
            first = eager(torch.tensor([[0, 1]]), use_cache=False).logits[:, -1]
            second = eager(torch.tensor([[2, 3, 4]]), use_cache=False).logits[:, -1]

        self.assertTrue(torch.isfinite(padded).all())
        torch.testing.assert_close(padded[0, -1], first[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(padded[1, -1], second[0], rtol=1e-5, atol=1e-5)

    def test_sdpa_clears_non_finite_outputs_for_fully_masked_padding_queries(self):
        model = build_whiteboard_model().eval().set_attention_implementation("sdpa")
        padded_ids = torch.tensor([[8, 8, 0], [2, 3, 4]])
        attention_mask = torch.tensor([[0, 0, 1], [1, 1, 1]])
        position_ids = torch.tensor([[0, 0, 0], [0, 1, 2]])
        native_sdpa = F.scaled_dot_product_attention

        def backend_with_non_finite_masked_rows(query, key, value, **kwargs):
            output = native_sdpa(query, key, value, **kwargs)
            allowed = kwargs["attn_mask"]
            fully_masked = ~allowed.any(dim=-1, keepdim=True)
            return output.masked_fill(fully_masked, float("nan"))

        with torch.no_grad(), patch(
            "toy_qwen.modeling.F.scaled_dot_product_attention",
            side_effect=backend_with_non_finite_masked_rows,
        ):
            output = model(
                padded_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        self.assertTrue(torch.isfinite(output.logits).all())
        for key, value in output.past_key_values:
            self.assertTrue(torch.isfinite(key).all())
            self.assertTrue(torch.isfinite(value).all())

    def test_attention_mask_must_cover_full_key_length(self):
        model = build_whiteboard_model().eval().set_attention_implementation("sdpa")
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            model(torch.tensor([[0, 1]]), attention_mask=torch.ones(1, 1, dtype=torch.long))
