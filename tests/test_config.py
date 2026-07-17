import json
import tempfile
import unittest
from pathlib import Path

from toy_qwen.config import QwenToyConfig, qwen25_05b_config, whiteboard_config


class ConfigTest(unittest.TestCase):
    def test_whiteboard_values(self):
        config = whiteboard_config()
        self.assertEqual((config.vocab_size, config.hidden_size, config.intermediate_size), (9, 4, 8))
        self.assertEqual((config.num_attention_heads, config.num_key_value_heads, config.head_dim), (1, 1, 4))
        self.assertFalse(config.attention_bias)

    def test_full_preset_values(self):
        config = qwen25_05b_config()
        self.assertEqual((config.hidden_size, config.intermediate_size, config.num_hidden_layers), (896, 4864, 24))
        self.assertEqual((config.num_attention_heads, config.num_key_value_heads), (14, 2))
        self.assertTrue(config.attention_bias)
        self.assertTrue(config.tie_word_embeddings)

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(whiteboard_config().to_dict()), encoding="utf-8")
            self.assertEqual(QwenToyConfig.from_json(path), whiteboard_config())

    def test_official_qwen_config_implies_qkv_bias(self):
        payload = qwen25_05b_config().to_dict()
        del payload["attention_bias"]
        self.assertTrue(QwenToyConfig.from_dict(payload).attention_bias)

    def test_invalid_head_combinations_fail(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            QwenToyConfig(9, 5, 8, 1, 2, 1)
        with self.assertRaisesRegex(ValueError, "even"):
            QwenToyConfig(9, 6, 8, 1, 2, 1)
        with self.assertRaisesRegex(ValueError, "key_value"):
            QwenToyConfig(9, 8, 8, 1, 4, 3)


if __name__ == "__main__":
    unittest.main()
