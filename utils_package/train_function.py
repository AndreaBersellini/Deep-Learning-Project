from .constants import DEVICE, CLASS_NAMES
from .plot import plot_batch, plot_pred
from .metrics import compute_IoU, compute_Dice, compute_PixelError
from .net_UNET import Unet
from tqdm import tqdm
import numpy as np
import torch

from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

def train_loop(loader : DataLoader, model : Module, optimizer : Optimizer, loss_fn : Module, curr_epoch : int, hpc : bool) -> float:
    loop = tqdm(enumerate(loader), desc=f"Training epoch {curr_epoch}", total = len(loader)) if not hpc else enumerate(loader)
    #oop = tqdm(enumerate(loader), desc=f"Training epoch {curr_epoch}", total = len(loader))

    train_losses = []

    model.train()

    for batch_idx, (images, targets, _) in loop:
        images : torch.Tensor = images.to(DEVICE)
        targets : torch.Tensor = targets.to(DEVICE)
        
        model.zero_grad()
        pred = model(images)
        loss : torch.Tensor = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        model.eval()
        
        #if batch_idx == 3 : break
    avg_loss = np.mean(train_losses)

    return avg_loss
    
def test_loop(loader : DataLoader, model : Module, loss_fn : Module, hpc : bool) -> tuple:
    loop = tqdm(enumerate(loader), desc=f"Testing the model... ", total = len(loader)) if not hpc else enumerate(loader)
    #loop = tqdm(enumerate(loader), desc=f"Testing the model... ", total = len(loader))

    test_losses = []
    test_iou_scores = []
    test_dice_scores = []
    test_pixel_error = []

    with torch.no_grad():
        for _, (images, targets, classes) in loop:
            images : torch.Tensor = images.to(DEVICE)
            targets : torch.Tensor = targets.to(DEVICE)

            pred = model(images.detach())

            loss : torch.Tensor = loss_fn(pred, targets)

            # ----- SAVE BATCH SCORE -----
            test_losses.append(loss.item())
            test_iou_scores.extend(compute_IoU(pred, targets, classes))
            test_dice_scores.extend(compute_Dice(pred, targets, classes))
            test_pixel_error.extend(compute_PixelError(pred, targets, classes))

    # ----- PLOT VISUAL PREDICTIONS -----
    if not hpc:
        plot_batch(images[10::], pred[10::], classes[10::], 10, (30, 6))
        plot_pred(images[10::], targets[10::], pred[10::], classes[10::], 10, path = "plots/predictions")

    avg_loss = np.mean(test_losses)
    avg_iou = [[] for _ in CLASS_NAMES]
    avg_dice = [[] for _ in CLASS_NAMES]
    avg_pxl_err = [[] for _ in CLASS_NAMES]

    for iou, cls in test_iou_scores:
        avg_iou[cls].append(iou)
    for dice, cls in test_dice_scores:
        avg_dice[cls].append(dice)
    for err, cls in test_pixel_error:
        avg_pxl_err[cls].append(err)
    
    avg_iou = [np.mean(iou) for iou in avg_iou]
    avg_dice = [np.mean(dice) for dice in avg_dice]
    avg_pxl_err = [np.mean(err) for err in avg_pxl_err]

    return avg_loss, avg_iou, avg_dice, avg_pxl_err
