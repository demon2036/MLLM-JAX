"""Sampling / generation abstractions.

This package hosts code related to "sampling" (prompt -> completion) that can be
shared across multiple pipelines without merging project code with reusable
training plugins.

Canonical subpackages:
- `plugins.sample.backends`: generation engines/backends
- `plugins.sample.workflows`: domain workflows (e.g. GRPO rollout -> training batch)
"""
