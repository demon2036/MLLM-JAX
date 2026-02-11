from __future__ import annotations

import os
import subprocess
import sys
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class TestGrpoTrainingPrintConfigCli(unittest.TestCase):
    def _run(self, args: list[str], *, expect_success: bool = True) -> str:
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "projects", "gsm8k_grpo", "scripts", "run_train.py"), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if expect_success and proc.returncode != 0:
            raise AssertionError(
                "print-config failed.\n"
                f"exit={proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )
        if (not expect_success) and proc.returncode == 0:
            raise AssertionError(
                "print-config unexpectedly succeeded.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )
        return proc.stdout + proc.stderr

    def test_print_config_default_grpo_is_explicit(self) -> None:
        out = self._run(["--print-config"])
        self.assertIn("algo:", out)
        self.assertIn("estimator:", out)
        self.assertIn("update:", out)
        self.assertIn("name: grpo", out)
        self.assertIn("kwargs:", out)
        self.assertIn("eps: 0.0001", out)
        self.assertIn("clip_range: null", out)
        self.assertIn("policy_gradient", out)
        self.assertNotIn("value_coef", out)
        self.assertIn("dynamic_sampling:", out)
        self.assertIn("enabled: false", out)
        self.assertIn("metric: acc", out)
        self.assertIn("eval_rollout_n: 1", out)
        self.assertIn("eval_full_sweep: false", out)

    def test_print_config_explicit_maxrl_hides_ppo_fields(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/rl_gsm8k_qwen25_3b_batch128_roll8_literal_maxrl_v6e8.yaml",
            ]
        )
        self.assertIn("name: maxrl", out)
        self.assertNotIn("value_coef", out)

    def test_print_config_ppo_shows_value_head_fields(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
                "--set",
                "algo.estimator.name=gae",
                "--set",
                "algo.estimator.kwargs={gamma: 1.0, gae_lambda: 0.95, normalize: true}",
                "--set",
                "algo.update.name=ppo",
                "--set",
                "algo.update.kwargs={value_coef: 0.5, value_clip_range: 0.2, entropy_coef: 0.0}",
            ]
        )
        self.assertIn("name: ppo", out)
        self.assertIn("gamma: 1.0", out)
        self.assertIn("gae_lambda: 0.95", out)
        self.assertIn("normalize: true", out)
        self.assertIn("value_coef: 0.5", out)
        self.assertIn("value_clip_range: 0.2", out)
        self.assertIn("entropy_coef: 0.0", out)

    def test_old_flat_update_keys_are_rejected(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
                "--set",
                "algo.update.value_coef=0.5",
            ],
            expect_success=False,
        )
        self.assertIn("Detected deprecated RL config keys", out)
        self.assertIn("algo.update.value_coef", out)

    def test_legacy_optimizer_flat_fields_are_rejected(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
                "--set",
                "train.optimizer.clip_norm=1.0",
            ],
            expect_success=False,
        )
        self.assertIn("Detected deprecated RL config keys", out)
        self.assertIn("train.optimizer.clip_norm", out)

    def test_optimizer_kwargs_shape_prints_cleanly(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
                "--set",
                "train.optimizer.name=lion",
                "--set",
                "train.optimizer.kwargs={clip_norm: 1.0, weight_decay: 1.0e-08}",
                "--set",
                "train.optimizer.lr_schedule.name=warmup_cosine",
                "--set",
                "train.optimizer.lr_schedule.kwargs={init_value: 0.0, peak_value: 1.0e-06, end_value: 0.0, warmup_ratio: 0.05, warmup_steps: null}",
            ]
        )
        self.assertIn("optimizer:", out)
        self.assertIn("name: lion", out)
        self.assertIn("kwargs:", out)
        self.assertIn("clip_norm: 1.0", out)
        self.assertIn("weight_decay: 1.0e-08", out)
        self.assertNotIn("muon_aux_lr", out)

    def test_dynamic_sampling_override_prints(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
                "--set",
                "rollout.dynamic_sampling.enabled=true",
                "--set",
                "rollout.dynamic_sampling.metric=seq_reward",
                "--set",
                "rollout.dynamic_sampling.max_extra_roll_rounds=5",
            ]
        )
        self.assertIn("dynamic_sampling:", out)
        self.assertIn("enabled: true", out)
        self.assertIn("metric: seq_reward", out)
        self.assertIn("max_extra_roll_rounds: 5", out)

    def test_dynamic_sampling_invalid_metric_rejected(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
                "--set",
                "rollout.dynamic_sampling.metric=foo",
            ],
            expect_success=False,
        )
        self.assertIn("rollout.dynamic_sampling.metric must be one of: acc, seq_reward", out)

    def test_eval_rollout_n_and_full_sweep_overrides_print(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
                "--set",
                "eval_rollout_n=1",
                "--set",
                "eval_full_sweep=true",
            ]
        )
        self.assertIn("eval_rollout_n: 1", out)
        self.assertIn("eval_full_sweep: true", out)

    def test_eval_rollout_n_must_be_positive(self) -> None:
        out = self._run(
            [
                "--print-config",
                "--config",
                "projects/gsm8k_grpo/configs/grpo_gsm8k_qwen25_3b_batch128_roll8_literal_v6e8.yaml",
                "--set",
                "eval_rollout_n=0",
            ],
            expect_success=False,
        )
        self.assertIn("eval_rollout_n must be >= 1", out)


if __name__ == "__main__":
    unittest.main()
