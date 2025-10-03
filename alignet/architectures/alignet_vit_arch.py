
from torch import nn
import torch.nn.functional as F
import torch

from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


from utils.registry import ARCHITECTURE_REGISTRY

# Examlpe de pretrained_encoder_path: facebook/dinov3-vitl16-pretrain-sat493m 
@ARCHITECTURE_REGISTRY.register("AligNetViT")
class AligNetViT(nn.Module):
    def __init__(self, input_dim, num_heads, window_size, embed_dim, encoder_dim_feedforward, img_size, num_transform_parameters, shared_encoder=True, shared_decoder=True, pretrained_encoder_path=None, project_embed=False):
        super().__init__()
        self.window_size = window_size
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        self.project_embed = project_embed
        self.num_patches = None
        pretrained_encoder_path = "facebook/dinov3-vitl16-pretrain-sat493m"

        if pretrained_encoder_path is None:
            # projection
            self.proj = nn.Linear(input_dim * window_size * window_size, embed_dim)
            if shared_encoder == False:
                self.proj2 = nn.Linear(input_dim * window_size * window_size, embed_dim)

            # positional embeddings
            num_patches = (img_size//window_size) * (img_size//window_size)
            self.pos_embedding = nn.Parameter(torch.empty(1, num_patches, embed_dim).normal_(std=0.02))
            if shared_encoder == False:
                self.pos_embedding2 = nn.Parameter(torch.empty(1, num_patches, embed_dim).normal_(std=0.02))

            # encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=encoder_dim_feedforward, norm_first=True, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
            if shared_encoder == False:
                encoder_layer2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=encoder_dim_feedforward, norm_first=True, batch_first=True)
                self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=8)
            self.encoder = None
        else:
            self.processor = AutoImageProcessor.from_pretrained(pretrained_encoder_path) #pretrained_model_name
            self.encoder = AutoModel.from_pretrained(
                    pretrained_encoder_path, 
                    device_map="auto", 
                )
            self.num_patches = num_patches = ((self.encoder.config.image_size // self.encoder.config.patch_size) ** 2) + self.encoder.config.num_register_tokens + 1
            if self.project_embed:
                self.proj_embed = nn.Linear(self.encoder.config.hidden_size, embed_dim)
            else:
                embed_dim = self.encoder.config.hidden_size
        
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, norm_first=True, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        if shared_decoder == False:
            decoder_layer2 = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, norm_first=True, batch_first=True)
            self.transformer_decoder2 = nn.TransformerDecoder(decoder_layer2, num_layers=6)

        # mlp head
        num_patches = (img_size//window_size) * (img_size//window_size) if self.num_patches is None else self.num_patches
        self.head = nn.Sequential(
            nn.Linear(num_patches*embed_dim, 2046),
            nn.ReLU(),
            nn.Linear(2046, num_transform_parameters)
        )
        if shared_decoder == False:
            self.head2 = nn.Sequential(
                nn.Linear(num_patches*embed_dim, 2046),
                nn.ReLU(),
                nn.Linear(2046, num_transform_parameters)
            )

    def encode(self, x, stream=1):  # (B,C,H,W) -> (B, num_patches, embed_dim)
        B, C, H, W = x.shape
        P = self.window_size
        x_patches = x.unfold(2, P, P).unfold(3, P, P)\
                     .permute(0,2,3,1,4,5).reshape(B, -1, C*P*P)
        if stream == 1:
            x_patches = self.proj(x_patches) + self.pos_embedding
            x_emb = self.transformer_encoder(x_patches)
        else:
            x_patches = self.proj2(x_patches) + self.pos_embedding2
            x_emb = self.transformer_encoder2(x_patches)
        return x_emb

    def decode_pair(self, x1_emb, x2_emb, num_streams=1):
        out_forward = self.transformer_decoder(x1_emb, x2_emb)
        if num_streams == 1:
            out_inverse = self.transformer_decoder(x2_emb, x1_emb)
        else:
            out_inverse = self.transformer_decoder2(x2_emb, x1_emb)
        B = x1_emb.size(0)
        pred_forward = self.head(out_forward.view(B, -1))
        if num_streams == 1:
            pred_inverse = self.head(out_inverse.view(B, -1))
        else:
            pred_inverse = self.head2(out_inverse.view(B, -1))
        return pred_forward, pred_inverse

    def forward_encoder(self, x):
        if self.encoder is not None:
            x = self.processor(images=x, return_tensors="pt", do_rescale=False)
            x_emb = self.encoder(**x).last_hidden_state
            if self.project_embed:
                x_emb = self.proj_embed(x_emb)
        else:
            x_emb = self.encode(x)
        return x_emb

    def forward(self, x1, x2):
        # x1 : [B, Image]
        # x2 : [B, Image]
        if self.encoder is not None:
            x1 = self.processor(images=x1, return_tensors="pt", do_rescale=False)
            x2 = self.processor(images=x2, return_tensors="pt", do_rescale=False)
            x1_emb = self.encoder(**x1).last_hidden_state
            x2_emb = self.encoder(**x2).last_hidden_state
            if self.project_embed:
                x1_emb = self.proj_embed(x1_emb)
                x2_emb = self.proj_embed(x2_emb)
        else:
            x1_emb = self.encode(x1)
            x2_emb = self.encode(x2, stream=1 if self.shared_encoder else 2)

        return self.decode_pair(x1_emb, x2_emb, num_streams=1 if self.shared_decoder else 2)
