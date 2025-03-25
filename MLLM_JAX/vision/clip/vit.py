from transformers.models.clip.modeling_clip import CLIPVisionModel
import flax.linen as nn









class CLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig
    def setup(self) -> None:
        config = self.config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = self.param(
            "class_embedding", init.truncated_normal(0.02), ( self.embed_dim,)
        )


        self.patch_embedding = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size,self.patch_size),
            strides=(self.patch_size,self.patch_size),
            use_bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embed(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)


    def forward(self, pixel_values: torch.FloatTensor,) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings



# class CLIPVisionEmbeddings(nn.Module):
#     def __init__(self, config: CLIPVisionConfig):
#         super().__init__()
#         self.config = config
#         self.embed_dim = config.hidden_size
#         self.image_size = config.image_size
#         self.patch_size = config.patch_size
#
#         self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
#
#         self.patch_embedding = nn.Conv2d(
#             in_channels=config.num_channels,
#             out_channels=self.embed_dim,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#             bias=False,
#         )
#
#         self.num_patches = (self.image_size // self.patch_size) ** 2
#         self.num_positions = self.num_patches + 1
#         self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
#         self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)
#
#
#     def forward(self, pixel_values: torch.FloatTensor,) -> torch.Tensor:
#         batch_size, _, height, width = pixel_values.shape
#
#         target_dtype = self.patch_embedding.weight.dtype
#         patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
#         patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
#
#         class_embeds = self.class_embedding.expand(batch_size, 1, -1)
#         embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
#         embeddings = embeddings + self.position_embedding(self.position_ids)
#         return embeddings