import numpy as np
from shapely.geometry import Polygon
from shapely.errors import GEOSException

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
    except GEOSException as e:
        print(f"Invalid gt polygon coordinates: {gt}")
        raise
    try:
        poly2 = Polygon(pred)
    except GEOSException as e:
        print(f"Invalid pred polygon coordinates: {pred}")
        raise
    return poly1.intersection(poly2).area / poly1.union(poly2).area

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