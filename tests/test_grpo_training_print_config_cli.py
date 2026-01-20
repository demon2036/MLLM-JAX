from __future__ import annotations

import os
import subprocess
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class TestGrpoTrainingPrintConfigCli(unittest.TestCase):
    def _run(self, args: list[str]) -> str:
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts", "run_grpo_gsm8k_training.py"), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(
                "print-config failed.\n"
                f"exit={proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )
        return proc.stdout

    def test_print_config_default(self) -> None:
        out = self._run(["--print-config"])
        self.assertIn("rollout:", out)
        self.assertIn("prompt_batch_size: 32", out)

    def test_print_config_bs128(self) -> None:
        out = self._run(["--print-config", "--config", "plugins/training/configs/grpo_gsm8k_bs128_steps100.yaml"])
        self.assertIn("batch_size: 128", out)


if __name__ == "__main__":
    unittest.main()

