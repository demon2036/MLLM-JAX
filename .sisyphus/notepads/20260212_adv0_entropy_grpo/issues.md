## 2026-02-12T15:57:58Z Task: adv0-entropy-grpo
- Risk: adding entropy term to loss backprop may increase memory/compute (entropy uses softmax over vocab per token). Mitigation: keep coef small; reuse existing entropy computation; normalize by `total_valid_token_count` to scale with adv0 token fraction.
- Risk: “adv==0” is exact float comparison; in GRPO-by-group-id it is often exactly 0 for homogeneous reward groups, so should be fine. Expose `adv0_entropy_eps` for tolerance.
- Risk: running full-split eval every 50 steps is expensive; ensure no duplicate final full-sweep run when last step already evaluated.
