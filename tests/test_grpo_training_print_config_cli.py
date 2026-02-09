from __future__ import annotations

import os
import subprocess
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class TestGrpoTrainingPrintConfigCli(unittest.TestCase):
    def _run(self, args: list[str]) -> str:
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "projects", "gsm8k_grpo", "scripts", "run_train.py"), *args],
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
        self.assertIn("algo:", out)
        self.assertIn("name: grpo", out)
        self.assertIn("model_path: Qwen/Qwen2.5-3B-Instruct", out)
        self.assertIn("batch_size: 128", out)
        self.assertIn("n: 8", out)
        self.assertIn("sequences_global_per_step: 1024", out)
        self.assertIn("mesh_shape: 1,-1,1", out)
        self.assertIn("wandb_mode: online", out)
        self.assertIn("gradient_checkpointing: true", out)

    def test_print_config_explicit_config(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
            ]
        )
        self.assertIn("algo:", out)
        self.assertIn("name: grpo", out)
        self.assertIn("model_path: Qwen/Qwen2.5-3B-Instruct", out)
        self.assertIn("batch_size: 128", out)
        self.assertIn("n: 8", out)
        self.assertIn("sequences_global_per_step: 1024", out)
        self.assertIn("mesh_shape: 1,-1,1", out)
        self.assertIn("gradient_checkpointing: true", out)

    def test_print_config_maxrl_config(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml",
            ]
        )
        self.assertIn("algo:", out)
        self.assertIn("name: max-rl", out)
        self.assertIn("estimator:", out)
        self.assertIn("name: max-rl", out)
        self.assertIn("batch_size: 128", out)
        self.assertIn("n: 8", out)
        self.assertIn("sequences_global_per_step: 1024", out)
        self.assertIn("gradient_checkpointing: true", out)



if __name__ == "__main__":
    unittest.main()
