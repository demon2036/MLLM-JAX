from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


def parse_dtype(name: str) -> jnp.dtype:
    name_norm = str(name).strip().lower()
    if name_norm in {"float32", "fp32"}:
        return jnp.float32
    if name_norm in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if name_norm in {"float16", "fp16"}:
        return jnp.float16
    raise ValueError(f"Unknown dtype: {name!r}")


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool
    param_dtype: jnp.dtype
    compute_dtype: jnp.dtype


class CausalSelfAttention(nn.Module):
    cfg: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        bsz, seqlen, n_embd = x.shape
        if n_embd % self.cfg.n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={self.cfg.n_head}")
        head_dim = n_embd // self.cfg.n_head

        qkv = nn.Dense(
            3 * n_embd,
            use_bias=self.cfg.bias,
            dtype=self.cfg.compute_dtype,
            param_dtype=self.cfg.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(bsz, seqlen, self.cfg.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seqlen, self.cfg.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seqlen, self.cfg.n_head, head_dim).transpose(0, 2, 1, 3)

        qf = q.astype(jnp.float32)
        kf = k.astype(jnp.float32)
        att = jnp.matmul(qf, jnp.swapaxes(kf, -1, -2)) / jnp.sqrt(jnp.asarray(head_dim, dtype=jnp.float32))

        causal = jnp.tril(jnp.ones((seqlen, seqlen), dtype=jnp.bool_))
        att = jnp.where(causal, att, jnp.full_like(att, -1e10))
        att = jax.nn.softmax(att, axis=-1).astype(self.cfg.compute_dtype)
        att = nn.Dropout(rate=self.cfg.dropout)(att, deterministic=deterministic)

        y = jnp.matmul(att, v.astype(self.cfg.compute_dtype))
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, n_embd)

        y = nn.Dense(
            n_embd,
            use_bias=self.cfg.bias,
            dtype=self.cfg.compute_dtype,
            param_dtype=self.cfg.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(y)
        y = nn.Dropout(rate=self.cfg.dropout)(y, deterministic=deterministic)
        return y


class MLP(nn.Module):
    cfg: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        hidden = 4 * self.cfg.n_embd
        x = nn.Dense(
            hidden,
            use_bias=self.cfg.bias,
            dtype=self.cfg.compute_dtype,
            param_dtype=self.cfg.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(
            self.cfg.n_embd,
            use_bias=self.cfg.bias,
            dtype=self.cfg.compute_dtype,
            param_dtype=self.cfg.param_dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
        )(x)
        x = nn.Dropout(rate=self.cfg.dropout)(x, deterministic=deterministic)
        return x


class Block(nn.Module):
    cfg: GPTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        x = x + CausalSelfAttention(self.cfg)(nn.LayerNorm(dtype=self.cfg.compute_dtype, param_dtype=self.cfg.param_dtype)(x), deterministic=deterministic)
        x = x + MLP(self.cfg)(nn.LayerNorm(dtype=self.cfg.compute_dtype, param_dtype=self.cfg.param_dtype)(x), deterministic=deterministic)
        return x


class GPT(nn.Module):
    cfg: GPTConfig

    @nn.compact
    def __call__(self, idx: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        if idx.ndim != 2:
            raise ValueError(f"idx must be (batch, time), got shape={idx.shape}")
        bsz, seqlen = idx.shape
        if seqlen > self.cfg.block_size:
            raise ValueError(f"seq_len={seqlen} exceeds block_size={self.cfg.block_size}")

        wte = self.param(
            "wte",
            nn.initializers.normal(stddev=0.02),
            (self.cfg.vocab_size, self.cfg.n_embd),
            self.cfg.param_dtype,
        )
        wpe = self.param(
            "wpe",
            nn.initializers.normal(stddev=0.02),
            (self.cfg.block_size, self.cfg.n_embd),
            self.cfg.param_dtype,
        )

        tok_emb = wte[idx]
        pos = jnp.arange(seqlen, dtype=jnp.int32)
        pos_emb = wpe[pos][None, :, :]

        x = (tok_emb + pos_emb).astype(self.cfg.compute_dtype)
        x = nn.Dropout(rate=self.cfg.dropout)(x, deterministic=deterministic)

        for _ in range(self.cfg.n_layer):
            x = Block(self.cfg)(x, deterministic=deterministic)

        x = nn.LayerNorm(dtype=self.cfg.compute_dtype, param_dtype=self.cfg.param_dtype)(x)

        logits = jnp.einsum("bte,ve->btv", x.astype(self.cfg.compute_dtype), wte.astype(self.cfg.compute_dtype))
        return logits


__all__ = ["GPTConfig", "GPT", "parse_dtype"]

