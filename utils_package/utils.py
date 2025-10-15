from .constants import EARLY_STOP_COEFF, EARLY_STOP_TAIL
import numpy as np
import cv2
import torch
from torch.optim import Optimizer
from torch.nn import Module
from collections import OrderedDict

def inter_quad(xs : list, ys : list) -> tuple:
    xs = np.asarray(xs[-EARLY_STOP_TAIL:], dtype=float)
    ys = np.asarray(ys[-EARLY_STOP_TAIL:], dtype=float)

    # ----- FIRST AND LAST POINT ON WHICH THE INTERPOLATION WILL PASS -----
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]

    if np.isclose(x1, x2):
        return 0, 1, y1, -np.pi

    m = (y2 - y1) / (x2 - x1)
    def L(x): return m * (x - x1) + y1

    phi = (xs - x1) * (xs - x2)

    r_line = ys - L(xs)

    denom = np.sum(phi**2)
    
    if np.isclose(denom, 0.0):
        a = 0.0
    else:
        a = np.sum(phi * r_line) / denom # parameter of X^0

    b = m - a * (x1 + x2) # parameter of X^1
    c = y1 - a * x1**2 - b * x1 # parameter of X^0

    # ----- DERIVATIVE OF THE INTERPOLATION IN THE FINAL POINT -----
    deriv_slope = np.arctan(2 * a * x2 + b)
    deriv_slope = 2 * a * x2 + b

    return (a, b, c, deriv_slope)

def early_stopping(losses : list) -> bool:
    iterations = range(len(losses))

    _, _, _, m = inter_quad(iterations, losses) # angular coefficient of the interpolation of the loss trend

    if m > EARLY_STOP_COEFF:
        return True
    else:
        return False
    
def dc_loss(pred : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
    smooth = 1.

    predf = pred.view(-1)
    targetf = target.view(-1)
    intersection = (predf * targetf).sum()
    
    return 1 - ((2. * intersection + smooth) / (predf.sum() + targetf.sum() + smooth))

def save_model(parameters : OrderedDict, optimizer : Optimizer, loss : Module, path : str) -> None:
    torch.save({
    'parameters': parameters,
    'optimizer': optimizer,
    'loss': loss,
    }, f"{path}")

def load_model(model : Module, optimizer : Optimizer, path : str) -> None:
    checkpoint = torch.load(f"{path}", weights_only=False)
    model.load_state_dict(checkpoint['parameters'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    return model, optimizer, loss

def otsu_threshold(images : torch.Tensor) -> torch.Tensor:
    if images.dim() == 4 and images.size(1) == 1:
        images = images.squeeze(1)

    binary_images = []
    for img in images:
        img = (img.cpu().numpy() * 255).astype(np.uint8)
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_images.append(torch.from_numpy(th.astype(np.float32) / 255.0))

    binary_images = torch.stack(binary_images, dim=0)

    if len(images.shape) == 4:
        binary_images = binary_images.unsqueeze(1)

    return binary_images