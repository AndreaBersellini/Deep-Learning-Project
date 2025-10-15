from .constants import BINARY_THRESHOLD
from .utils import otsu_threshold
import torch

def compute_IoU(preds : torch.Tensor, targets : torch.Tensor, classes : torch.Tensor) -> list:
    preds = otsu_threshold(preds) if BINARY_THRESHOLD == 0 else (preds > BINARY_THRESHOLD).float() # binary thresholding (0 = Otsu's method) (!0 = constant value)
    not_zero = 1e-6 # prevent dizision by 0
    results = []
    for p, t in zip(preds, targets):
        pred = p.view(-1)
        target = t.view(-1)
        iou = (abs((pred * target).sum()) + not_zero) / (abs(pred.sum() + target.sum() - (pred * target).sum()) + not_zero)
        results.append(iou)
    return list(zip(results, classes))

def compute_Dice(preds : torch.Tensor, targets : torch.Tensor, classes : torch.Tensor) -> list:
    preds = otsu_threshold(preds) if BINARY_THRESHOLD == 0 else (preds > BINARY_THRESHOLD).float() # binary thresholding (0 = Otsu's method) (!0 = constant value)
    not_zero = 1e-6 # prevent dizision by 0
    results = []
    for p, t in zip(preds, targets):
        pred = p.view(-1)
        target = t.view(-1)
        dice = (2.0 * abs((pred * target).sum()) + not_zero) / (abs(pred.sum()) + abs(target.sum()) + not_zero)
        results.append(dice)
    return list(zip(results, classes))

def compute_PixelError(preds : torch.Tensor, targets : torch.Tensor, classes : torch.Tensor) -> list:
    preds = otsu_threshold(preds) if BINARY_THRESHOLD == 0 else (preds > BINARY_THRESHOLD).float() # binary thresholding (0 = Otsu's method) (!0 = constant value)
    results = []
    for p, t in zip(preds, targets):
        pred = p.view(-1)
        target = t.view(-1)
        error = 1.0 - ((pred == target).sum().float() / target.numel())
        results.append(error)
    return list(zip(results, classes))