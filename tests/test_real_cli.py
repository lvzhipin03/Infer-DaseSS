import subprocess
import sys
import unittest

from real_qwen_inference import DEFAULT_MODEL_PATH, build_parser


class RealQwenCliTest(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.model_path, DEFAULT_MODEL_PATH)
        self.assertEqual(args.prompt, "中国的首都是哪里？")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.max_new_tokens, 32)
        self.assertFalse(args.trace_shapes)

    def test_help_does_not_require_optional_packages(self):
        completed = subprocess.run(
            [sys.executable, "real_qwen_inference.py", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--model-path", completed.stdout)
        self.assertIn("--trace-shapes", completed.stdout)


if __name__ == "__main__":
    unittest.main()
