import numpy as np
from shapely.geometry import Polygon
from shapely.errors import GEOSException
from shapely.validation import make_valid

from torch.nn.functional import l1_loss

from losses.registration_loss import MILoss
from utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register("calculate_rmse")
def calculate_rmse(gt, pred):
    return np.sqrt(np.mean(np.sum((gt - pred) ** 2, axis=1)))

@METRIC_REGISTRY.register("calculate_mpd")
def calculate_mpd(gt, pred):
    return np.mean(np.linalg.norm(gt - pred, axis=1))

@METRIC_REGISTRY.register("calculate_iou")
def calculate_iou(gt, pred):
    try:
        poly1 = Polygon(gt)
        poly2 = Polygon(pred)
    except GEOSException as e:
        print(f"[Error] Invalid polygon construction: {e}")
        return 0.0

    # Try to repair invalid geometries
    try:
        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)
    except Exception:
        # Fallback for older Shapely versions (<2.0)
        poly1 = poly1.buffer(0)
        poly2 = poly2.buffer(0)

    # In case they are still invalid after repair
    if not poly1.is_valid or not poly2.is_valid:
        print("[Warning] Could not repair invalid polygons, returning IoU = 0")
        return 0.0

    try:
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        if union_area == 0:
            return 0.0
        return intersection_area / union_area
    except GEOSException as e:
        print(f"[Error] Intersection/Union failed: {e}")
        return 0.0

@METRIC_REGISTRY.register("calculate_h_err")
def calculate_h_err(gt, pred):
    return np.linalg.norm(pred - gt, ord='fro')

@METRIC_REGISTRY.register("calculate_mi")
def calculate_mi(gt, pred):
    mi = MILoss(sample_ratio=0.8, normalised=False)
    return -1 * mi(pred, gt).cpu().numpy()

@METRIC_REGISTRY.register("calculate_re")
def calculate_re(gt, pred):
    return l1_loss(pred, gt).cpu().numpy()