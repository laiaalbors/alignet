import torch
import numpy as np

from utils.transforms import STN
from utils.utils import inverse_affine_matrix, apply_normalized_affine_to_polygon
from utils.registry import METRIC_REGISTRY

def calculate_metrics(img_r, img_s, forward_tp_gt, inverse_tp_gt, forward_tp_pred, inverse_tp_pred, mask_r, mask_s, cfg, metric_results, h=256, w=256):
    device = img_r.device

    assert not torch.isnan(forward_tp_pred).any(), f"forward_tp_pred contains NaNs: \n{forward_tp_pred}"
    assert not torch.isnan(inverse_tp_pred).any(), f"inverse_tp_pred contains NaNs: \n{inverse_tp_pred}"

    # Apply predicted registrations
    img_s_registered = STN(img_r, inverse_affine_matrix(forward_tp_pred, device))
    img_r_registered = STN(img_s, inverse_affine_matrix(inverse_tp_pred, device))

    # Get bbox polygons
    bbox_r = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    bbox_s = apply_normalized_affine_to_polygon(bbox_r, forward_tp_gt.cpu().squeeze().numpy(), h, w)
    bbox_r_registered = apply_normalized_affine_to_polygon(bbox_s, inverse_tp_pred.cpu().squeeze().numpy(), h, w)
    bbox_s_registered = apply_normalized_affine_to_polygon(bbox_r, forward_tp_pred.cpu().squeeze().numpy(), h, w)

    if forward_tp_gt.shape[-1] == 6:
        f = 2
    elif forward_tp_gt.shape[-1] == 8:
        one = torch.ones(1, 1, device=forward_tp_gt.device)
        forward_tp_gt = torch.cat([forward_tp_gt, one], dim=1)
        forward_tp_pred = torch.cat([forward_tp_pred, one], dim=1)
        inverse_tp_gt = torch.cat([inverse_tp_gt, one], dim=1)
        inverse_tp_pred = torch.cat([inverse_tp_pred, one], dim=1)
        f = 3
    else:
        raise ValueError(f"Number of transformation parameters not recognized, expected 6 or 8, got {forward_tp_gt.shape[-1]}")

    for name, options in cfg['metrics'].items():
        if name == "auc":
            metric_results[name] = []
        elif name not in metric_results:
            metric_results[name] = 0
        
        if name in ('rmse', 'mpd', 'iou'):
            metric_results[name] += METRIC_REGISTRY.get(options['type'])(bbox_r, bbox_r_registered)
            metric_results[name] += METRIC_REGISTRY.get(options['type'])(bbox_s, bbox_s_registered)
        elif name == 'h_err':
            metric_results[name] += METRIC_REGISTRY.get(options['type'])(inverse_tp_gt.cpu().squeeze().numpy().reshape(f,3), inverse_tp_pred.cpu().squeeze().numpy().reshape(f,3))
            metric_results[name] += METRIC_REGISTRY.get(options['type'])(forward_tp_gt.cpu().squeeze().numpy().reshape(f,3), forward_tp_pred.cpu().squeeze().numpy().reshape(f,3))
        elif name in ('mi', 're'):
            metric_results[name] += METRIC_REGISTRY.get(options['type'])(img_r*mask_r, img_r_registered)
            metric_results[name] += METRIC_REGISTRY.get(options['type'])(img_s*mask_s, img_s_registered)
        elif name == 'auc':
            metric_results[name].append(METRIC_REGISTRY.get('calculate_mpd')(bbox_r, bbox_r_registered))
            metric_results[name].append(METRIC_REGISTRY.get('calculate_mpd')(bbox_s, bbox_s_registered))

    return metric_results


def auc_corner_error(errors, max_threshold):
    xs = np.linspace(0, max_threshold, 1000)
    ys = [(errors < x).mean() for x in xs]
    auc = np.trapz(ys, xs) / max_threshold
    return auc * 100