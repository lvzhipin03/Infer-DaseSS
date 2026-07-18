import subprocess
import sys
import unittest


class VerificationCliTest(unittest.TestCase):
    def test_direct_script_help_finds_project_package(self):
        completed = subprocess.run(
            [sys.executable, "verification/compare_transformers.py", "--help"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--tolerance", completed.stdout)
        self.assertIn("--attn-implementation", completed.stdout)
        self.assertIn("{eager,sdpa}", completed.stdout)


if __name__ == "__main__":
    unittest.main()
