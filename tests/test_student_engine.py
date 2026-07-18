import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

import student_release.student_engine as student_engine_module


class StudentEngineTest(unittest.TestCase):
    def setUp(self):
        tokenizer_class_patch = patch.object(
            student_engine_module, "QwenTokenizerAdapter", create=True
        )
        loader_patch = patch.object(
            student_engine_module, "load_pretrained_qwen", create=True
        )
        generator_patch = patch.object(
            student_engine_module, "greedy_generate", create=True
        )
        self.tokenizer_class = tokenizer_class_patch.start()
        self.load_model = loader_patch.start()
        self.generate_tokens = generator_patch.start()
        self.addCleanup(tokenizer_class_patch.stop)
        self.addCleanup(loader_patch.stop)
        self.addCleanup(generator_patch.stop)

        self.tokenizer = self.tokenizer_class.from_model_dir.return_value
        self.model = MagicMock()
        self.report = MagicMock()
        self.load_model.return_value = (self.model, self.report)

    def build_engine(self):
        return student_engine_module.StudentEngine(
            "/model",
            device="cpu",
            dtype="float16",
            attn_implementation="sdpa",
            local_files_only=True,
            seed=7,
        )

    def test_initialization_loads_exact_local_model_configuration(self):
        engine = self.build_engine()

        self.tokenizer_class.from_model_dir.assert_called_once_with("/model")
        self.load_model.assert_called_once_with(
            "/model", device=torch.device("cpu"), dtype="float16"
        )
        self.assertIs(engine.model, self.model)
        self.assertIs(engine.checkpoint_report, self.report)
        self.assertEqual(engine.attn_implementation, "sdpa")
        self.assertEqual(engine.seed, 7)
        self.assertTrue(engine.local_files_only)

    def test_generate_preserves_order_and_returns_only_continuations(self):
        self.tokenizer.encode_chat.side_effect = [
            ("chat-a", [10, 11]),
            ("chat-b", [20, 21, 22]),
        ]
        self.tokenizer.decode.side_effect = ["answer-a", "answer-b"]
        self.generate_tokens.side_effect = [
            SimpleNamespace(generated_ids=(101, 102)),
            SimpleNamespace(generated_ids=(201, 202)),
        ]
        engine = self.build_engine()

        outputs = engine.generate(
            ["prompt-a", "prompt-b"],
            max_new_tokens=2,
            batch_size=2,
            suite_name="ignored",
        )

        self.assertEqual(outputs, ["answer-a", "answer-b"])
        self.assertEqual(self.generate_tokens.call_count, 2)
        expected_ids = ((10, 11), (20, 21, 22))
        for invocation, expected in zip(self.generate_tokens.call_args_list, expected_ids):
            input_ids = invocation.args[1]
            self.assertEqual(tuple(input_ids.shape), (1, len(expected)))
            self.assertEqual(input_ids.dtype, torch.long)
            self.assertEqual(tuple(input_ids[0].tolist()), expected)
            self.assertIsNone(invocation.kwargs["eos_token_id"])
            self.assertEqual(invocation.kwargs["max_new_tokens"], 2)
            self.assertEqual(invocation.kwargs["top_k"], 5)
        self.tokenizer.decode.assert_has_calls([
            call((101, 102), skip_special_tokens=True),
            call((201, 202), skip_special_tokens=True),
        ])

    def test_rejects_empty_prompt_list(self):
        engine = self.build_engine()
        with self.assertRaisesRegex(ValueError, "prompts"):
            engine.generate([], 1)

    def test_rejects_non_string_prompt(self):
        engine = self.build_engine()
        with self.assertRaisesRegex(TypeError, "prompt"):
            engine.generate(["valid", 3], 1)

    def test_rejects_non_positive_token_budget(self):
        engine = self.build_engine()
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            engine.generate(["prompt"], 0)

    def test_rejects_non_positive_batch_size(self):
        engine = self.build_engine()
        with self.assertRaisesRegex(ValueError, "batch_size"):
            engine.generate(["prompt"], 1, batch_size=0)


if __name__ == "__main__":
    unittest.main()
