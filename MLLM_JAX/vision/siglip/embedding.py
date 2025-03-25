#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   embedding.py
@Time    :   2025/03/20 21:35:52
@Author  :   biabuluo
@Version :   1.0
@Desc    :   None
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   modeling_siglip.py
@Time    :   2025/03/19 23:28:02
@Author  :   biabuluo
@Version :   1.0
@Desc    :   None
"""
import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any
from configuration_siglip import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Module):
    config: SiglipVisionConfig

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size

        self.patch_embedding = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            name="patch_embedding",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embed(
            num_embeddings=self.num_positions,
            features=self.embed_dim,
            name="position_embedding",
        )

        # Create position_ids - in Flax, we handle this in __call__
        self.position_ids = jnp.expand_dims(jnp.arange(self.num_positions), axis=0)

    def interpolate_pos_encoding(
        self, embeddings: jnp.ndarray, height: int, width: int
    ) -> jnp.ndarray:
        num_patches = embeddings.shape[1]
        num_positions = self.num_positions

        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = jnp.expand_dims(self.position_embedding.embedding, axis=0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(math.sqrt(num_positions))
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))

        patch_pos_embed = jax.image.resize(
            patch_pos_embed,
            shape=(1, dim, new_height, new_width),
            method="bicubic",
            antialias=True,
        )

        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 2, 3, 1))
        patch_pos_embed = patch_pos_embed.reshape(1, -1, dim)
        return patch_pos_embed

    def __call__(
        self, pixel_values: jnp.ndarray, interpolate_pos_encoding: bool = False
    ) -> jnp.ndarray:
        batch_size, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = jnp.reshape(patch_embeds, (batch_size, -1, self.embed_dim))

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


if __name__ == "__main__":
    config = SiglipVisionConfig(
        hidden_size=128,  # Transformer 隐藏层维度
        image_size=224,  # 输入图像尺寸 224x224
        patch_size=16,  # Patch 大小 16x16
        num_channels=3,  # 输入通道（RGB）
    )

    model = SiglipVisionEmbeddings(config=config)
    rng = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(rng, (2, 224, 224, 3))

    variables = model.init(rng, dummy_input)
    output = model.apply(variables, dummy_input)

    num_patches = (config.image_size // config.patch_size) ** 2
    expected_shape = (2, num_patches, config.hidden_size)
    assert (
        output.shape == expected_shape
    ), f"输出形状错误: {output.shape}, 期望: {expected_shape}"

    print("测试通过，模型初始化并前向传播成功！")
    print("输入形状:", dummy_input.shape)
    print("输出形状:", output.shape)
