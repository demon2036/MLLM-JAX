# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""An example showing how to load a checkpoint and sample from it.

Getting Started with Gemma Sampling:

Prerequisites:

1. Download your Gemma checkpoint: Choose the desired checkpoint and download it.
2. Get the Gemma tokenizer: Download the tokenizer file required for your model.
3. Install Gemma: Follow the straightforward instructions in the README to install the Gemma repository.

Ready to Sample!

Here's how to run the sampling.py script:

python sampling.py --path_checkpoint=${PATH_TO_THE_GEMMA_CHECKPOINT} \
    --path_tokenizer=${PATH_TO_THE_GEMMA_TOKENIZER} \
    --string_to_sample="Where is Paris?"
"""

from typing import Sequence

import jax
from absl import app
from absl import flags

# from gemma import params as params_lib
from language.gemma import sampler as sampler_lib
from language.gemma import transformer as transformer_lib


import sentencepiece as spm

from language.gemma.transformer import Transformer, TransformerConfig
from load_weight import get_pretrain_params

_PATH_CHECKPOINT = flags.DEFINE_string(
    "path_checkpoint", None, required=False, help="Path to checkpoint."
)
_PATH_TOKENIZER = flags.DEFINE_string(
    "path_tokenizer", None, required=False, help="Path to tokenizer."
)
_TOTAL_GENERATION_STEPS = flags.DEFINE_integer(
    "total_sampling_steps",
    128,
    help="Maximum number of step to run when decoding.",
)
_STRING_TO_SAMPLE = flags.DEFINE_string(
    "string_to_sample",
    "Where is Paris ?",#Where is Paris ?
    help="Input string to sample.",
)

_CACHE_SIZE = 1024


def _load_and_sample(
    *,
    path_checkpoint: str,
    path_tokenizer: str,
    input_string: str,
    cache_size: int,
    total_generation_steps: int,
) -> None:
  jax.config.update('jax_platform_name', 'cpu')

  """Loads and samples a string from a checkpoint."""
  print(f"Loading the parameters from {path_checkpoint}")
  # parameters = params_lib.load_and_format_params(path_checkpoint)
  path_tokenizer="tokenizer.model"
  parameters = get_pretrain_params()

  parameters=jax.tree_util.tree_map(jax.numpy.asarray,parameters)

  print("Parameters loaded.")
  # Create a sampler with the right param shapes.
  vocab = spm.SentencePieceProcessor()
  vocab.Load(path_tokenizer)
  # transformer_config = transformer_lib.TransformerConfig.from_params(
  #     parameters,
  #     cache_size=cache_size,
  # )
  # transformer = transformer_lib.Transformer(transformer_config)

  gemma_2b_config = TransformerConfig.gemma_2b_pali(1024)
  transformer = Transformer(gemma_2b_config)



  sampler = sampler_lib.Sampler(
      transformer=transformer,
      vocab=vocab,
      params=parameters['params'],#parameters["transformer"],
  )


  sampled_str = sampler(
      input_strings=[input_string],
      total_generation_steps=total_generation_steps,
  ).text

  print(f"Input string: {input_string}")
  print(f"Sampled string: {sampled_str}")


def main(argv: Sequence[str]) -> None:

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  _load_and_sample(
      path_checkpoint=_PATH_CHECKPOINT.value,
      path_tokenizer=_PATH_TOKENIZER.value,
      input_string=_STRING_TO_SAMPLE.value,
      cache_size=_CACHE_SIZE,
      total_generation_steps=_TOTAL_GENERATION_STEPS.value,
  )


if __name__ == "__main__":
  app.run(main)
