import unittest
import torch

from toy_qwen.config import whiteboard_config
from toy_qwen.modeling import QwenToyAttention, QwenToyRotaryEmbedding, repeat_kv


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
