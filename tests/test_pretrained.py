import unittest
from dataclasses import replace

import torch

from toy_qwen.config import whiteboard_config
from toy_qwen.modeling import QwenToyForCausalLM
from toy_qwen.pretrained import _resolve_dtype, validate_checkpoint


class CheckpointValidationTest(unittest.TestCase):
    def setUp(self):
        self.config = replace(whiteboard_config(), tie_word_embeddings=True)
        self.model = QwenToyForCausalLM(self.config)
        self.state = {key: value.clone() for key, value in self.model.state_dict().items()}

    def test_tied_checkpoint_may_omit_lm_head(self):
        self.state.pop("lm_head.weight")

        report = validate_checkpoint(self.model, self.state)

        self.assertEqual(report.tensor_count, len(self.state))
        self.assertEqual(report.expected_tied_missing, ("lm_head.weight",))

    def test_tied_checkpoint_rejects_conflicting_alias_values(self):
        self.state["model.embed_tokens.weight"].zero_()
        self.state["lm_head.weight"].fill_(7)

        with self.assertRaisesRegex(ValueError, "tied.*lm_head.weight"):
            validate_checkpoint(self.model, self.state)

    def test_other_missing_tensor_fails(self):
        self.state.pop("model.norm.weight")

        with self.assertRaisesRegex(ValueError, "model.norm.weight"):
            validate_checkpoint(self.model, self.state)

    def test_unexpected_tensor_fails(self):
        self.state["bad.weight"] = torch.ones(1)

        with self.assertRaisesRegex(ValueError, "bad.weight"):
            validate_checkpoint(self.model, self.state)

    def test_shape_mismatch_names_tensor(self):
        key = "model.embed_tokens.weight"
        self.state[key] = self.state[key][:-1]

        with self.assertRaisesRegex(ValueError, key):
            validate_checkpoint(self.model, self.state)

    def test_untied_checkpoint_requires_lm_head(self):
        model = QwenToyForCausalLM(whiteboard_config())
        state = {key: value.clone() for key, value in model.state_dict().items()}
        state.pop("lm_head.weight")

        with self.assertRaisesRegex(ValueError, "lm_head.weight"):
            validate_checkpoint(model, state)

    def test_float16_dtype_is_supported(self):
        self.assertIs(_resolve_dtype("float16"), torch.float16)
        self.assertIs(_resolve_dtype(torch.float16), torch.float16)

    def test_unsupported_dtype_message_lists_all_supported_values(self):
        with self.assertRaisesRegex(ValueError, "float16.*bfloat16.*float32"):
            _resolve_dtype("float64")


if __name__ == "__main__":
    unittest.main()
