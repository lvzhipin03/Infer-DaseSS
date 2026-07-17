import unittest

from toy_qwen.tokenizer import ToyTokenizer


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ToyTokenizer()

    def test_required_prompt_ids(self):
        self.assertEqual(self.tokenizer.encode("中国首都是"), [0, 1, 2, 3, 4])
        self.assertEqual(self.tokenizer.legacy_ids("中国首都是"), [10, 20, 30, 40, 50])
        self.assertEqual(self.tokenizer.decode([5]), "北")

    def test_round_trip(self):
        text = "中国首都是北京上海"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_unknown_character_reports_position(self):
        with self.assertRaisesRegex(ValueError, "位置 1"):
            self.tokenizer.encode("中法")

    def test_invalid_decode_id_fails(self):
        with self.assertRaisesRegex(ValueError, "token id 9"):
            self.tokenizer.decode([9])
