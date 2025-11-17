
from torch import nn
import torch.nn.functional as F
import torch

from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


from utils.registry import ARCHITECTURE_REGISTRY

# Examlpe de pretrained_encoder_path: facebook/dinov3-vitl16-pretrain-sat493m 
@ARCHITECTURE_REGISTRY.register("AligNetViT")
class AligNetViT(nn.Module):
    def __init__(self, input_dim, num_heads, window_size, img_size, num_transform_parameters, pretrained_encoder_path, project_embed=False, embed_dim=None):
        super().__init__()
        self.window_size = window_size
        self.project_embed = project_embed
        self.num_patches = None

        if project_embed and embed_dim is None:
            print(f"[Warning] 'project_embed' is True but 'embed_dim' is not defined. Using 'embed_dim=1024' by default.")
            embed_dim = 1024

        # pretrained encoder
        self.processor = AutoImageProcessor.from_pretrained(pretrained_encoder_path)
        self.encoder = AutoModel.from_pretrained(
                pretrained_encoder_path, 
                device_map="auto", 
            )
        self.num_patches = ((self.encoder.config.image_size // self.encoder.config.patch_size) ** 2) + self.encoder.config.num_register_tokens + 1
        if self.project_embed:
            # self.proj_embed = nn.Linear(self.encoder.config.hidden_size, embed_dim)
            self.proj_embed = nn.Sequential(
                nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size*2),
                nn.ReLU(),
                nn.Linear(self.encoder.config.hidden_size*2, embed_dim)
            )
        else:
            embed_dim = self.encoder.config.hidden_size
        
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, norm_first=True, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # mlp head
        self.head = nn.Sequential(
            nn.Linear(self.num_patches*embed_dim, 2046),
            nn.ReLU(),
            nn.Linear(2046, num_transform_parameters)
        )

    def decode_pair(self, x1_emb, x2_emb):
        out_forward = self.transformer_decoder(x1_emb, x2_emb)
        out_inverse = self.transformer_decoder(x2_emb, x1_emb)
        B = x1_emb.size(0)
        pred_forward = self.head(out_forward.view(B, -1))
        pred_inverse = self.head(out_inverse.view(B, -1))
        return pred_forward, pred_inverse

    def forward_encoder(self, x):
        x = self.processor(images=x, return_tensors="pt", do_rescale=False)
        x_emb = self.encoder(**x).last_hidden_state
        if self.project_embed:
            x_emb = self.proj_embed(x_emb)
        return x_emb

    def forward(self, x1, x2):
        # x1 : [B, Image]
        # x2 : [B, Image]
        x1 = self.processor(images=x1, return_tensors="pt", do_rescale=False)
        x2 = self.processor(images=x2, return_tensors="pt", do_rescale=False)
        x1_emb = self.encoder(**x1).last_hidden_state
        x2_emb = self.encoder(**x2).last_hidden_state
        if self.project_embed:
            x1_emb = self.proj_embed(x1_emb)
            x2_emb = self.proj_embed(x2_emb)

        return self.decode_pair(x1_emb, x2_emb)
