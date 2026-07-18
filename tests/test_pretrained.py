import unittest
from dataclasses import replace

import torch

from toy_qwen.config import whiteboard_config
from toy_qwen.modeling import QwenToyForCausalLM
from toy_qwen.pretrained import validate_checkpoint


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


if __name__ == "__main__":
    unittest.main()
