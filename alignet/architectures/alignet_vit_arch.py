
from torch import nn
import torch.nn.functional as F
import torch

from transformers import AutoModel
from transformers.image_utils import load_image

from utils.registry import ARCHITECTURE_REGISTRY


# Examlpe de pretrained_encoder_path: facebook/dinov3-vitl16-pretrain-sat493m 
@ARCHITECTURE_REGISTRY.register("AligNetViT")
class AligNetViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg = cfg

        input_dim=self.model_cfg["input_channels"]
        num_heads=self.model_cfg.get("attention_heads", 4)
        self.window_size=self.model_cfg.get("window_size", 16)
        img_size=self.model_cfg["img_size"]
        num_transform_parameters=self.model_cfg["num_transform_parameters"]
        pretrained_encoder_path=self.model_cfg.get("pretrained_encoder_path", "facebook/dinov3-vitl16-pretrain-sat493m")
        self.project_embed=self.model_cfg.get("project_embed", False)
        embed_dim=self.model_cfg.get("embed_dim", None)
        self.use_lora=self.model_cfg.get("use_lora", False)
        self.num_patches = None

        if project_embed and embed_dim is None:
            print(f"[Warning] 'project_embed' is True but 'embed_dim' is not defined. Using 'embed_dim=1024' by default.")
            embed_dim = 1024

        # pretrained encoder
        self.encoder = AutoModel.from_pretrained(pretrained_encoder_path, device_map="auto")

        self.num_patches = ((self.encoder.config.image_size // self.encoder.config.patch_size) ** 2) + getattr(self.encoder.config, "num_register_tokens", 0) + 1
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
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_transform_parameters)
        )

    def decode_pair(self, x1_emb, x2_emb):
        out_forward = self.transformer_decoder(x1_emb, x2_emb)
        out_inverse = self.transformer_decoder(x2_emb, x1_emb)
        out_forward_cls = out_forward[:, 0]      # (B, D)
        out_inverse_cls = out_inverse[:, 0]      # (B, D)
        pred_forward = self.head(out_forward_cls)
        pred_inverse = self.head(out_inverse_cls)
        return pred_forward, pred_inverse

    def forward(self, x1, x2):
        # x1 : [B, Image]
        # x2 : [B, Image]
        x1_emb = self.encoder(x1.repeat(1, 3, 1, 1)).last_hidden_state
        x2_emb = self.encoder(x2.repeat(1, 3, 1, 1)).last_hidden_state
        if self.project_embed:
            x1_emb = self.proj_embed(x1_emb)
            x2_emb = self.proj_embed(x2_emb)

        return self.decode_pair(x1_emb, x2_emb)
