from torch import nn
import torch.nn.functional as F
import torch

from utils.registry import ARCHITECTURE_REGISTRY

@ARCHITECTURE_REGISTRY.register("ModalityDiscriminator")
class ModalityDiscriminator(nn.Module):
    def __init__(self, img_size, window_size, embed_dim, hidden=2046, num_modalities=2):
        super().__init__()
        num_patches = (img_size//window_size) * (img_size//window_size)
        self.net = nn.Sequential(
            nn.Linear(num_patches*embed_dim, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, num_modalities)
        )
    def forward(self, feat):  # feat: (B, num_patches, embed_dim)
        B = feat.shape[0]
        feat_flat = feat.view(B, -1) # feat_flat: (B, num_patches*embed_dim)
        return self.net(feat_flat)