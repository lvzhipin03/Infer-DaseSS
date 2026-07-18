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
        padder_patch = patch.object(
            student_engine_module, "left_pad_token_ids", create=True
        )
        generator_patch = patch.object(
            student_engine_module, "batched_greedy_generate", create=True
        )
        self.tokenizer_class = tokenizer_class_patch.start()
        self.load_model = loader_patch.start()
        self.pad_token_ids = padder_patch.start()
        self.generate_tokens = generator_patch.start()
        self.addCleanup(tokenizer_class_patch.stop)
        self.addCleanup(loader_patch.stop)
        self.addCleanup(padder_patch.stop)
        self.addCleanup(generator_patch.stop)

        self.tokenizer = self.tokenizer_class.from_model_dir.return_value
        self.tokenizer.pad_token_id = 151643
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
            "/model",
            device=torch.device("cpu"),
            dtype="float16",
            attn_implementation="sdpa",
        )
        self.assertIs(engine.model, self.model)
        self.assertIs(engine.checkpoint_report, self.report)
        self.assertEqual(engine.attn_implementation, "sdpa")
        self.assertEqual(engine.seed, 7)
        self.assertTrue(engine.local_files_only)

    def test_generate_batches_chunks_and_preserves_output_order(self):
        self.tokenizer.encode_chat.side_effect = [
            ("chat-a", [10]),
            ("chat-b", [20, 21]),
            ("chat-c", [30, 31, 32]),
            ("chat-d", [40]),
            ("chat-e", [50, 51]),
        ]
        self.tokenizer.decode.side_effect = [f"answer-{letter}" for letter in "abcde"]
        padded_batches = [SimpleNamespace(name=name) for name in ("ab", "cd", "e")]
        self.pad_token_ids.side_effect = padded_batches
        self.generate_tokens.side_effect = [
            SimpleNamespace(generated_ids=((101, 102), (201, 202))),
            SimpleNamespace(generated_ids=((301, 302), (401, 402))),
            SimpleNamespace(generated_ids=((501, 502),)),
        ]
        engine = self.build_engine()

        outputs = engine.generate(
            [f"prompt-{letter}" for letter in "abcde"],
            max_new_tokens=2,
            batch_size=2,
            suite_name="ignored",
        )

        self.assertEqual(outputs, [f"answer-{letter}" for letter in "abcde"])
        self.pad_token_ids.assert_has_calls([
            call([[10], [20, 21]], pad_token_id=151643, device=torch.device("cpu")),
            call([[30, 31, 32], [40]], pad_token_id=151643, device=torch.device("cpu")),
            call([[50, 51]], pad_token_id=151643, device=torch.device("cpu")),
        ])
        self.generate_tokens.assert_has_calls([
            call(self.model, padded_batches[0], max_new_tokens=2),
            call(self.model, padded_batches[1], max_new_tokens=2),
            call(self.model, padded_batches[2], max_new_tokens=2),
        ])
        self.tokenizer.decode.assert_has_calls([
            call((101, 102), skip_special_tokens=True),
            call((201, 202), skip_special_tokens=True),
            call((301, 302), skip_special_tokens=True),
            call((401, 402), skip_special_tokens=True),
            call((501, 502), skip_special_tokens=True),
        ])

    def test_batch_size_one_uses_batch_generator(self):
        self.tokenizer.encode_chat.return_value = ("chat", [10, 11])
        padded = SimpleNamespace(name="one")
        self.pad_token_ids.return_value = padded
        self.generate_tokens.return_value = SimpleNamespace(generated_ids=((101,),))
        self.tokenizer.decode.return_value = "answer"
        engine = self.build_engine()

        self.assertEqual(engine.generate(["prompt"], 1, batch_size=1), ["answer"])
        self.generate_tokens.assert_called_once_with(
            self.model, padded, max_new_tokens=1
        )

    def test_cuda_oom_splits_chunk_and_preserves_order(self):
        self.tokenizer.encode_chat.side_effect = [
            ("chat-a", [10]),
            ("chat-b", [20, 21]),
        ]
        padded_pair, padded_a, padded_b = [
            SimpleNamespace(name=name) for name in ("pair", "a", "b")
        ]
        self.pad_token_ids.side_effect = [padded_pair, padded_a, padded_b]
        self.generate_tokens.side_effect = [
            torch.cuda.OutOfMemoryError("simulated allocation failure"),
            SimpleNamespace(generated_ids=((101,),)),
            SimpleNamespace(generated_ids=((201,),)),
        ]
        self.tokenizer.decode.side_effect = ["answer-a", "answer-b"]
        engine = self.build_engine()

        outputs = engine.generate(["prompt-a", "prompt-b"], 1, batch_size=2)

        self.assertEqual(outputs, ["answer-a", "answer-b"])
        self.pad_token_ids.assert_has_calls([
            call([[10], [20, 21]], pad_token_id=151643, device=torch.device("cpu")),
            call([[10]], pad_token_id=151643, device=torch.device("cpu")),
            call([[20, 21]], pad_token_id=151643, device=torch.device("cpu")),
        ])

    def test_cuda_oom_from_single_request_is_not_hidden(self):
        self.tokenizer.encode_chat.return_value = ("chat", [10])
        self.pad_token_ids.return_value = SimpleNamespace(name="one")
        self.generate_tokens.side_effect = torch.cuda.OutOfMemoryError("still too large")
        engine = self.build_engine()

        with self.assertRaisesRegex(torch.cuda.OutOfMemoryError, "still too large"):
            engine.generate(["prompt"], 1, batch_size=1)

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

    def test_rejects_tokenizer_without_pad_token(self):
        self.tokenizer.pad_token_id = None
        engine = self.build_engine()
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            engine.generate(["prompt"], 1)


if __name__ == "__main__":
    unittest.main()
