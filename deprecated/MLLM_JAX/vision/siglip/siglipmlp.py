#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   modeling_siglip.py
@Time    :   2025/03/19 23:28:02
@Author  :   biabuluo
@Version :   1.0
@Desc    :   None
"""
import flax.linen as nn
from configuration_siglip import SiglipVisionConfig
from act import gelu_pytorch_tanh


class SiglipMLP(nn.Module):
    config: SiglipVisionConfig

    def setup(self):
        self.fc1 = nn.Dense(features=self.config.intermediate_size)
        self.fc2 = nn.Dense(features=self.config.hidden_size)

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = gelu_pytorch_tanh(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


if __name__ == "__main__":
    import jax

    config = SiglipVisionConfig(hidden_size=128, intermediate_size=256)
    model = SiglipMLP(config=config)
    # 生成随机输入 (batch_size=1, hidden_size=128)
    rng = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(rng, (1, config.hidden_size))

    # 初始化模型参数
    variables = model.init(rng, dummy_input)
    output = model.apply(variables, dummy_input)

    assert output.shape == (1, config.hidden_size), f"输出形状错误: {output.shape}"

    print("测试通过，输出形状正确:", output.shape)
