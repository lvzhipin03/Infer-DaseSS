import subprocess
import sys
import unittest
import torch

from toy_qwen.inference import predict_next_token
from toy_qwen.tokenizer import ToyTokenizer
from toy_qwen.weights import build_whiteboard_model


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.model = build_whiteboard_model().eval()
        self.tokenizer = ToyTokenizer()

    def test_reference_logits_and_next_token(self):
        prediction = predict_next_token("中国首都是", self.model, self.tokenizer)
        self.assertEqual((prediction.token, prediction.token_id), ("北", 5))
        expected = torch.tensor([6.199933, 3.560251, -5.495592, -3.757640])
        torch.testing.assert_close(prediction.logits[[5, 6, 7, 8]], expected, atol=1e-5, rtol=1e-5)

    def test_cached_decode_matches_full_forward(self):
        ids = torch.tensor([self.tokenizer.encode("中国首都是")])
        with torch.no_grad():
            full = self.model(ids).logits[:, -1]
            prefill = self.model(ids[:, :-1], use_cache=True)
            cached = self.model(ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True).logits[:, -1]
        torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)

    def test_prediction_follows_weights(self):
        original = self.model.lm_head.weight.detach().clone()
        self.model.lm_head.weight.data[[5, 6]] = original[[6, 5]]
        self.assertEqual(predict_next_token("中国首都是", self.model, self.tokenizer).token, "京")

    def test_empty_prompt_fails(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            predict_next_token("", self.model, self.tokenizer)

    def test_cli(self):
        completed = subprocess.run([sys.executable, "whiteboard_llm_inference.py", "--prompt", "中国首都是", "--trace-shapes"],
                                   text=True, capture_output=True, check=True)
        self.assertIn("dense ids: [0, 1, 2, 3, 4]", completed.stdout)
        self.assertTrue(completed.stdout.rstrip().endswith("next token: 北"))
