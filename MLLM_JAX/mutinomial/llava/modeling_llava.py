import flax.linen as nn

from ...activations import ACT2FN
from ...vision.clip.clip import CLIPVisionTransformer, CLIPVisionModel
from .configuration_llava import LlavaConfig








class LlavaMultiModalProjector(nn.Module):
    config: LlavaConfig
    def setup(self) -> None:
        print(self.config)
        self.linear_1 = nn.Dense(
             self.config.text_config.hidden_size, use_bias=True
        )
        self.act = ACT2FN[self.config.projector_hidden_act]
        self.linear_2 = nn.Dense(
             self.config.text_config.hidden_size, use_bias=True
        )

    def __call__(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states




class LlavaForConditionalGeneration(nn.Module):
    config:LlavaConfig


    def setup(self) -> None:
        self.vision_tower=CLIPVisionModel(self.config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(self.config)



    def get_image_features(self,pixel_values,vision_feature_layer:int,vision_feature_select_strategy:str):
        image_outputs=self.vision_tower(pixel_values,output_hidden_states=True)
        selected_image_feature=image_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

        # return selected_image_feature



    def __call__(self, pixel_values,vision_feature_layer=None,vision_feature_select_strategy=None):
        """
        hidden_states=self.vision_tower.vision_model.embeddings(pixel_values)
        out = self.vision_tower.vision_model.pre_layrnorm(hidden_states)

        # layer_0 = self.vision_tower.vision_model.encoder.layers[0]
        #
        #
        # out=layer_0(out,attention_mask=None,causal_attention_mask=None)[0]

        layer_0 = self.vision_tower.vision_model.encoder

        out = layer_0(out, attention_mask=None, causal_attention_mask=None).last_hidden_state
        """

        # residual = out
        # out = layer_0.layer_norm1(out)
        # out = layer_0.self_attn(out)[0]
        # out += residual
        # residual=out
        # out = layer_0.layer_norm2(out)
        # out = layer_0.mlp(out)
        # out+=residual

        # out = self.vision_tower.vision_model.encoder.layers[0](out,attention_mask=None,causal_attention_mask=None)


        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        return self.get_image_features(pixel_values,vision_feature_layer,vision_feature_select_strategy)





def convert_hf_to_flax():
    pass


