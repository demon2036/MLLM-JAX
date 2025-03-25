import flax.linen as nn
import jax

from ..language.gemma.transformer import Transformer as Gemma
from ..vision.siglip.vit_pali import Model as ViT
import jax.numpy as jnp

def make_attn_mask(input_mask, mask_ar):
  """Returns attention mask bool[B, N, N] to use in transformer.

  Tokens can attend to valid inputs tokens which have a cumulative mask_ar
  smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
  setup several types of attention, for example:

    [[1 1 1 1 1 1]]: pure causal attention.

    [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
        themselves and the last 3 tokens have a causal attention. The first
        entry could also be a 1 without changing behaviour.

    [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
        block can attend all previous blocks and all tokens on the same block.

  Args:
    input_mask: bool[B, N] true if its part of the input, false if padding.
    mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
      it and 0 where it shares the same attention mask as the previous token.
  """
  cumsum = jnp.cumsum(mask_ar, axis=1)
  attn_mask = (cumsum[:, None, :] <= cumsum[:, :, None])
  valid_mask = (input_mask[:, None, :] * input_mask[:, :, None])
  return jnp.logical_and(attn_mask, valid_mask)


class PaliGemma(nn.Module):

    img:ViT
    llm:Gemma

    def embed_image(self,pixel_values):
        x, out=self.img(pixel_values)
        return x

    def embed_text(self,tokens):
        return self.llm.embedder.encode(tokens)

    def decode_embeding(
            self,
            prefix_embed: jax.Array,  # [B, L]
            positions: jax.Array,  # [B, L]
            cache: None,  # (sequence length L')
            attention_mask: jax.Array,  # [B, L, L']
    ):

        x=prefix_embed
        for i, block in enumerate(self.llm.blocks):
            layer_name = f'layer_{i}'
            layer_cache = cache[layer_name] if cache else None
            layer_cache, x = block(
                x,
                positions,
                layer_cache,
                attention_mask,
            )
            if cache is not None:
                cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

        x = self.llm.final_norm(x)
        logits = self.llm.embedder.decode(x)

        # if self.config.final_logit_softcap is not None:
        #     logits /= self.config.final_logit_softcap
        #     logits = jnp.tanh(logits) * self.config.final_logit_softcap

        return logits, cache  # pytype: disable=bad-return-type

    def embed_image_and_text(self, image, text, *,
                             input_mask=None, mask_ar=None, train=False):
        """Concats image/text into a sequence of embeded tokens to pass to `llm`.

        Args:
          image: float[B, H, W, 3] image to be embedded by the `img` model and used
            as prefix to the sequence passed to the `llm` model.
          text: int32[B, T] token sequence to embedded by the `llm`.
          input_mask: bool[B, T] true if the text token is a valid token and false
            if its a token to pad the sequence. Defaults to all being input tokens.
          mask_ar: int32[B, T] mask that's 1 where `text` should be attended to
            causally, and 0 where it can be attended to with full self-attention.
            Defaults to all text tokens being auto-regressive.
          train: bool whether we're in train or test mode (dropout etc).

        Returns:
          Tuple (x: float[B, N, E], input_mask: bool[B, N], mask_ar: int[B, N]) and
          auxiliary outputs.
        """
        out_img = self.embed_image(image,)
        out_txt = self.embed_text(text, )

        if input_mask is None:
            input_mask = jnp.full(text.shape, True)
        if mask_ar is None:
            mask_ar = jnp.full(text.shape, 1)

        # Concatenate embeded image and text into a single token sequence.
        x = jnp.concatenate([out_img, out_txt], axis=1)
        _, img_len, _ = out_img.shape
        pad_width = ((0, 0), (img_len, 0))
        mask_ar = jnp.pad(mask_ar, pad_width, constant_values=0)
        input_mask = jnp.pad(input_mask, pad_width, constant_values=True)

        return (x, input_mask, mask_ar),out_img

    def __call__(self, image, text, mask_ar, train=False):
        """Concats image/text and returns text logits.

        Args:
          image: float32[B, H, W, 3] image that can be passed to the `img` model.
          text: int32[B, T] token sequence that can be embedded by the `txt` model.
          mask_ar: int32[B, T] mask that's 1 where `text` should be attended to
            causally, and 0 where it can be attended to with full self-attention.
          train: bool whether we're in train or test mode (dropout etc).

        Returns:
          float32[B, T, V] logits for the `text` input, and an out-dict of named
          intermediates.
        """
        # Embed the image and text.
        (x, input_mask, mask_ar),out_img = self.embed_image_and_text(
            image, text, mask_ar=mask_ar, train=train)
        positions=jnp.arange(0,x.shape[1])[None,...]
        # Call transformer on the embedded token sequence.
        attn_mask = make_attn_mask(input_mask, mask_ar)

        for i, block in enumerate(self.llm.blocks):
            layer_cache, x = block(
                x,
                positions,
                None,
                attn_mask,
            )

        x = self.llm.final_norm(x)
        print(out_img.shape)
        x=x[:,out_img.shape[1]:]
        logits = self.llm.embedder.decode(x)

        return logits,None




        """
        
        
        _, out_llm = self.llm(x, mask=attn_mask, train=train)
        for k, v in out_llm.items():
            out[f"llm/{k}"] = v

        # Extract the logits for the text tokens.
        zimg = out["img/zimg"]
        text_pre_logits = out["llm/pre_logits"][:, zimg.shape[1]:, :]
        text_logits = self._llm.compute_logits(text_pre_logits, train=train)
        out["text_logits"] = text_logits
        out["text_tokens"] = jnp.argmax(text_logits, axis=-1)
        return text_logits, out
        """


