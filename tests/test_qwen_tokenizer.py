import unittest

from toy_qwen.qwen_tokenizer import render_qwen_chat


class QwenTokenizerTest(unittest.TestCase):
    def test_default_system_chat_template(self):
        rendered = render_qwen_chat([{"role": "user", "content": "中国的首都是哪里？"}])
        self.assertEqual(
            rendered,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n中国的首都是哪里？<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_explicit_system_and_history(self):
        messages = [
            {"role": "system", "content": "简洁回答。"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "继续"},
        ]
        rendered = render_qwen_chat(messages)
        self.assertTrue(rendered.startswith("<|im_start|>system\n简洁回答。<|im_end|>\n"))
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_without_generation_prompt(self):
        rendered = render_qwen_chat([{"role": "user", "content": "你好"}], False)
        self.assertFalse(rendered.endswith("<|im_start|>assistant\n"))

    def test_rejects_empty_and_unsupported_roles(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            render_qwen_chat([])
        with self.assertRaisesRegex(ValueError, "tool"):
            render_qwen_chat([{"role": "tool", "content": "x"}])
