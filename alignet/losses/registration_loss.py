import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

from utils.utils import inverse_affine_matrix
from utils.transforms import STN
from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("RegistrationLoss")
class RegistrationLoss(nn.Module):
    def __init__(self, alpha_1=10, alpha_2=1, alpha_3=1):
        """
        Initializes the RegistrationLoss class.

        Args:
            alpha_1 (float): is the weight for the loss_tp component. 
            alpha_2 (float): is the weight for the loss_nmi component. 
            alpha_3 (float): is the weight for the loss_sym component. 
        """
        super(RegistrationLoss, self).__init__()
        self.nmi_loss = MILoss(sample_ratio=0.8, normalised=False)
        self.tp_loss = nn.L1Loss()
        self.att_loss = nn.L1Loss()
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3

    def generate_mask(self, img):
        mask = torch.gt(img, 1)
        mask = torch.tensor(mask, dtype=torch.float32)
        return mask
    
    def symaffi(self, pred_inverse_tp, pred_forward_tp, device):
        b = pred_inverse_tp.shape[0]
        unit = torch.Tensor([[[1,0,0],[0,1,0],[0,0,1]]]).to(device).repeat(b, 1, 1)

        if pred_inverse_tp.shape[-1] == 6:
            idx3 = torch.Tensor([[[0,0,1]]]).to(device).repeat(b, 1, 1)
            pred_inverse_tp = pred_inverse_tp.reshape(-1, 2, 3)
            pred_inverse_tp_mat = torch.cat((pred_inverse_tp, idx3), dim=1)
            pred_forward_tp = pred_forward_tp.reshape(-1, 2, 3)
            pred_forward_tp_mat = torch.cat((pred_forward_tp, idx3), dim=1)  
        elif pred_inverse_tp.shape[-1] == 8:
            ones = torch.ones(b, 1, device=device, dtype=pred_inverse_tp.dtype)
            pred_inverse_tp = torch.cat([pred_inverse_tp, ones], dim=1)
            pred_inverse_tp_mat = pred_inverse_tp.reshape(-1, 3, 3)
            pred_forward_tp = torch.cat([pred_forward_tp, ones], dim=1)
            pred_forward_tp_mat = pred_forward_tp.reshape(-1, 3, 3)
        else:
            raise ValueError(f"Expected 6 (Affine) or 8 (Projective) prediction parameters, but got {pred_inverse_tp.shape[-1]}.")

        e = torch.matmul(pred_inverse_tp_mat, pred_forward_tp_mat)

        return torch.sum(abs(e-unit), dim=[-2,-1]).mean()

    def forward(self, gt_inverse_tp, gt_forward_tp, pred_inverse_tp, pred_forward_tp, 
                samples_r, samples_s, mask_r, mask_s, device='cpu'):
        """
        Computes the combined registration loss.

        Args:
            y_true (torch.Tensor): Ground truth tensor.
            y_pred (torch.Tensor): Predicted tensor.

        Returns:
            torch.Tensor: Combined loss value.
        """

        # Apply predicted transformation to the sensed image
        samples_s_registered = STN(samples_s, inverse_affine_matrix(pred_inverse_tp, device))
        samples_r_registered = STN(samples_r, inverse_affine_matrix(pred_forward_tp, device))

        loss_nmi = self.nmi_loss(samples_r*mask_r, samples_s_registered) + self.nmi_loss(samples_s*mask_s, samples_r_registered)
        loss_tp1 = self.tp_loss(gt_inverse_tp, pred_inverse_tp) 
        loss_tp2 = self.tp_loss(gt_forward_tp, pred_forward_tp)
        loss_tp = loss_tp1 + loss_tp2
        loss_sym = self.symaffi(pred_inverse_tp, pred_forward_tp, device)

        combined_loss = self.alpha_1*loss_tp + self.alpha_2*loss_nmi + self.alpha_3*loss_sym

        return combined_loss, loss_tp, loss_nmi, loss_sym


class MILoss(nn.Module):
    """
    Mutual information loss using Gaussian kernel in KDE
    From: https://github.com/Ahuer-Lei/ADRNet/blob/9ca0f8c65d9de390ae22d281e86e2e9d058f940c/modules/losses.py
    """
    def __init__(self,
                 vmin=0.0,
                 vmax=1.0,
                 num_bins=64,
                 sample_ratio=0.1,
                 normalised=True
                 ):
        super(MILoss, self).__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.sample_ratio = sample_ratio
        self.normalised = normalised

        # set the std of Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / num_bins
        self.sigma = bin_width * (1/(2 * math.sqrt(2 * math.log(2))))

        # set bin edges
        self.num_bins = num_bins
        self.bins = torch.linspace(self.vmin, self.vmax, self.num_bins, requires_grad=False).unsqueeze(1)

    def _compute_joint_prob(self, x, y):
        """
        Compute joint distribution and entropy
        Input shapes (N, 1, prod(sizes))
        """
        # cast bins
        self.bins = self.bins.type_as(x)

        # calculate Parzen window function response (N, #bins, H*W*D)
        win_x = torch.exp(-(x - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_x = win_x / (math.sqrt(2 * math.pi) * self.sigma)
        win_y = torch.exp(-(y - self.bins) ** 2 / (2 * self.sigma ** 2))
        win_y = win_y / (math.sqrt(2 * math.pi) * self.sigma)

        # calculate joint histogram batch
        hist_joint = win_x.bmm(win_y.transpose(1, 2))  # (N, #bins, #bins)

        # normalise joint histogram to get joint distribution
        hist_norm = hist_joint.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        p_joint = hist_joint / hist_norm.view(-1, 1, 1)  # (N, #bins, #bins) / (N, 1, 1)

        return p_joint

    def forward(self, x, y):
        """
        Calculate (Normalised) Mutual Information Loss.

        Args:
            x: (torch.Tensor, size (N, 1, *sizes))
            y: (torch.Tensor, size (N, 1, *sizes))

        Returns:
            (Normalise)MI: (scalar)
        """
        if self.sample_ratio < 1.:
            # random spatial sampling with the same number of pixels/voxels
            # chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions
        # batch size in dim0, x bins in dim1, y bins in dim2
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -torch.mean((ent_x + ent_y) / ent_joint)
        else:
            return -torch.mean(ent_x + ent_y - ent_joint)