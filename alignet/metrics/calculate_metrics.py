import os
import torch
import numpy as np

import torchvision.utils as vutils

from utils.transforms import STN
from utils.utils import inverse_affine_matrix, apply_normalized_affine_to_polygon
from utils.registry import METRIC_REGISTRY

def calculate_metrics(img_r, img_s, forward_tp_gt, inverse_tp_gt, forward_tp_pred, inverse_tp_pred, mask_r, mask_s, cfg, metric_results, h=256, w=256):
    device = img_r.device

    path_save = cfg.get('path_save_images', None)
    direction = cfg.get('direction', 'both')
    if direction not in ('both', 'forward', 'inverse'):
        print(f"[WARNING] Direction of evaluation not recognized. It should be one of these: 'both', 'forward', 'inverse'. Setting it to the default value: 'both'.", flush=True)
        direction = 'both'
    forward_dir = inverse_dic = True
    if direction == 'forward':
        inverse_dic = False
    elif direction == "inverse":
        forward_dir = False

    assert not torch.isnan(forward_tp_pred).any(), f"forward_tp_pred contains NaNs: \n{forward_tp_pred}"
    assert not torch.isnan(inverse_tp_pred).any(), f"inverse_tp_pred contains NaNs: \n{inverse_tp_pred}"

    # Apply predicted registrations
    img_s_registered = STN(img_r, inverse_affine_matrix(forward_tp_pred, device))
    img_r_registered = STN(img_s, inverse_affine_matrix(inverse_tp_pred, device))

    # Apply GT registrations
    img_s_registered_gt = STN(img_r, inverse_affine_matrix(forward_tp_gt, device))
    img_r_registered_gt = STN(img_s, inverse_affine_matrix(inverse_tp_gt, device))

    # Get bbox polygons
    bbox_r = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    bbox_s = apply_normalized_affine_to_polygon(bbox_r, forward_tp_gt.cpu().squeeze().numpy(), h, w)
    bbox_r_registered = apply_normalized_affine_to_polygon(bbox_s, inverse_tp_pred.cpu().squeeze().numpy(), h, w)
    bbox_s_registered = apply_normalized_affine_to_polygon(bbox_r, forward_tp_pred.cpu().squeeze().numpy(), h, w)

    # -- Qualitative results --
    if path_save is not None:
        num_files = len(os.listdir(path_save))
        print(f"img_r_registered: {img_r_registered.shape} - {type(img_r_registered)}", flush=True)
        print(f"    inverse_tp_pred: {inverse_tp_pred}", flush=True)
        vutils.save_image(img_r_registered, path_save+f'output_s_{num_files+1}.png', normalize=True)
        vutils.save_image(img_s_registered, path_save+f'output_r_{num_files+1}.png', normalize=True)
        vutils.save_image(img_r, path_save+f'input_r_{num_files+1}.png', normalize=True)
        vutils.save_image(img_s, path_save+f'input_s_{num_files+1}.png', normalize=True)
        img_s_registered_gt = STN(img_r, inverse_affine_matrix(forward_tp_gt, device))
        img_r_registered_gt = STN(img_s, inverse_affine_matrix(inverse_tp_gt, device))
        vutils.save_image(img_r_registered_gt, path_save+f'output_s_{num_files+1}_gt.png', normalize=True)
        vutils.save_image(img_s_registered_gt, path_save+f'output_r_{num_files+1}_gt.png', normalize=True)
        bbox_r_registered_ = apply_normalized_affine_to_polygon(bbox_r, inverse_tp_pred.cpu().squeeze().numpy(), h, w)
        bbox_s_registered_ = apply_normalized_affine_to_polygon(bbox_r, forward_tp_pred.cpu().squeeze().numpy(), h, w)
        bbox_r_registered_gt_ = apply_normalized_affine_to_polygon(bbox_r, inverse_tp_gt.cpu().squeeze().numpy(), h, w)
        bbox_s_registered_gt_ = apply_normalized_affine_to_polygon(bbox_r, forward_tp_gt.cpu().squeeze().numpy(), h, w)
        if cfg.get('metrics', False):
            print(f"[PRED] bbox_r_registered file: {num_files+1}:")
            print(bbox_r_registered_)
            print(f"\n[PRED] bbox_s_registered file: {num_files+1}:")
            print(bbox_s_registered_, flush=True)
            print(f"\n[GT] bbox_r_registered file: {num_files+1}:")
            print(bbox_r_registered_gt_)
            print(f"\n[GT] bbox_s_registered file: {num_files+1}:")
            print(bbox_s_registered_gt_, flush=True)
    # -------------------------

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

    if cfg.get('metrics', False):
        for name, options in cfg['metrics'].items():
            if name not in metric_results:
                metric_results[name] = [] if name in ("auc", "acc") else 0
            if "rmse_for" not in metric_results:
                metric_results['rmse_for'] = 0
            if "rmse_inv" not in metric_results:
                metric_results['rmse_inv'] = 0
            
            if name in ('rmse', 'mpd', 'iou'):
                if inverse_dic:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(bbox_r, bbox_r_registered)
                if forward_dir:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(bbox_s, bbox_s_registered)
            elif name in ('acc5', 'acc10', 'acc20'):
                if inverse_dic:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(bbox_r, bbox_r_registered, thr=int(name[3:]))
                if forward_dir:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(bbox_s, bbox_s_registered, thr=int(name[3:]))
            elif name == 'h_err':
                if inverse_dic:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(inverse_tp_gt.cpu().squeeze().numpy().reshape(f,3), inverse_tp_pred.cpu().squeeze().numpy().reshape(f,3))
                if forward_dir:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(forward_tp_gt.cpu().squeeze().numpy().reshape(f,3), forward_tp_pred.cpu().squeeze().numpy().reshape(f,3))
            elif name in ('mi', 're'):
                if inverse_dic:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(img_r*mask_r, img_r_registered)
                if forward_dir:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(img_s*mask_s, img_s_registered)
            elif name in ('auc', 'acc'):
                if inverse_dic:
                    metric_results[name].append(METRIC_REGISTRY.get('calculate_mpd')(bbox_r, bbox_r_registered))
                if forward_dir:
                    metric_results[name].append(METRIC_REGISTRY.get('calculate_mpd')(bbox_s, bbox_s_registered))
            elif name == 'ssim':
                if inverse_dic:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(img_r_registered_gt, img_r_registered)
                if forward_dir:
                    metric_results[name] += METRIC_REGISTRY.get(options['type'])(img_s_registered_gt, img_s_registered)
            elif name == 'diff_rmse':
                metric_results[name] += abs(METRIC_REGISTRY.get('calculate_rmse')(bbox_r, bbox_r_registered) - METRIC_REGISTRY.get('calculate_rmse')(bbox_s, bbox_s_registered))

            # if name == "mpd":
            #     print("\n[INVERSE]", METRIC_REGISTRY.get('calculate_mpd')(bbox_r, bbox_r_registered), "- [FORWARD]", METRIC_REGISTRY.get('calculate_mpd')(bbox_s, bbox_s_registered))

            # if name == "mpd":
            #     print(f"\n[forward] MI: {METRIC_REGISTRY.get('calculate_mi')(img_s*mask_s, img_s_registered)} - MPD: {METRIC_REGISTRY.get('calculate_mpd')(bbox_s, bbox_s_registered)}", flush=True)
            #     print(f"\n[inverse] MI: {METRIC_REGISTRY.get('calculate_mi')(img_r*mask_r, img_r_registered)} - MPD: {METRIC_REGISTRY.get('calculate_mpd')(bbox_r, bbox_r_registered)}", flush=True)

    return metric_results


def auc_corner_error(errors, max_threshold):
    xs = np.linspace(0, max_threshold, 1000)
    ys = [(errors < x).mean() for x in xs]
    auc = np.trapz(ys, xs) / max_threshold
    return auc * 100

def acc_corner_error(errors, max_threshold):
    errors = np.asanyarray(errors)
    acc = (errors <= max_threshold).mean()
    return acc * 100