
from torch import nn
import torch.nn.functional as F
import torch

from transformers import AutoModel
from transformers.image_utils import load_image

from peft import LoraConfig, get_peft_model

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
        self.img_size=self.model_cfg["img_size"]
        num_transform_parameters=self.model_cfg["num_transform_parameters"]
        pretrained_encoder_path=self.model_cfg.get("pretrained_encoder_path", "facebook/dinov3-vitl16-pretrain-sat493m")
        self.project_embed=self.model_cfg.get("project_embed", False)
        self.embed_dim=self.model_cfg.get("embed_dim", None)
        self.use_lora=self.model_cfg.get("use_lora", False)
        self.num_patches = None

        # pretrained encoder
        self.encoder = AutoModel.from_pretrained(pretrained_encoder_path) #, device_map="auto"
        if self.use_lora:
            peft_config = LoraConfig(
                r=self.model_cfg.get("lora_r", 8), 
                lora_alpha=self.model_cfg.get("lora_alpha", 32), 
                lora_dropout=self.model_cfg.get("lora_dropout", 0.05), 
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                task_type=None, 
                bias="none",
                inference_mode=False, 
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            self.encoder.gradient_checkpointing_enable()
            self.encoder.print_trainable_parameters()

        if self.embed_dim is None:
            if self.project_embed:
                print(f"[WARNING] 'project_embed' is True but 'embed_dim' is not defined. Using 'embed_dim=encoder.config.hidden_size' by default.")
            self.embed_dim = getattr(self.encoder.config, "hidden_size", None)

        if self.embed_dim is None:
            raise ValueError("Could not determine 'embed_dim' from encoder config, and none was provided in cfg. Please set 'embed_dim' manually.")

        self.num_patches = ((self.encoder.config.image_size // self.encoder.config.patch_size) ** 2) + getattr(self.encoder.config, "num_register_tokens", 0) + 1
        if self.project_embed:
            # self.proj_embed = nn.Linear(self.encoder.config.hidden_size, self.embed_dim)
            self.proj_embed = nn.Sequential(
                nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size*2),
                nn.ReLU(),
                nn.Linear(self.encoder.config.hidden_size*2, self.embed_dim)
            )
        
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=num_heads, norm_first=True, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # mlp head
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
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
        x1 = x1.repeat(1, 3, 1, 1).to(self.encoder.device)
        x2 = x2.repeat(1, 3, 1, 1).to(self.encoder.device)
        x1_emb = self.encoder(x1).last_hidden_state
        x2_emb = self.encoder(x2).last_hidden_state
        if self.project_embed:
            x1_emb = self.proj_embed(x1_emb)
            x2_emb = self.proj_embed(x2_emb)

        return self.decode_pair(x1_emb, x2_emb)
