from .constants import BATCH_SIZE, CLASS_NAMES, EARLY_STOP_TAIL, BINARY_THRESHOLD
from .utils import inter_quad
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
    
def plot_batch(images : torch.Tensor, targets : torch.Tensor, classes : torch.Tensor, batch_size : int = BATCH_SIZE, plot_size : tuple = (0, 0)) -> None:
    
    size = (batch_size * 3, 6) if plot_size == (0, 0) else plot_size

    fig, ax = plt.subplots(2, batch_size, figsize=size)

    for i in range(batch_size):
        img = images[i].permute(1, 2, 0)
        gt = targets[i].permute(1, 2, 0)
        cls = classes[i]

        # ----- IMAGES -----
        ax[0][i].imshow(img, cmap='gray')
        ax[0][i].set_title(f"Image - Class: {cls}")
        ax[0][i].axis('off')

        # ----- TARGETS -----
        ax[1][i].imshow(gt, cmap='binary')
        ax[1][i].set_title("Target")
        ax[1][i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_loss(losses : list, title : str, ylabel : str, path : str = "", color : str = "blue") -> None:
    iterations = range(len(losses))
    
    figure, ax = plt.subplots()

    # ----- PLOT LOSS TREND -----
    ax.plot(iterations, losses, color=color)
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ----- PLOT QUADRATIC INTERPOLATION OF THE LOSS -----
    a, b, c, m = inter_quad(iterations, losses)
    x = np.array(iterations[-EARLY_STOP_TAIL:])
    y = a * x**2 + b * x + c
    ax.plot(x, y, color="red")

    # ----- PLOT DERIVATIVE OF THE QUADRATIC INTERPOLATION -----
    #xi = x[-1]
    #yi = a * xi**2 + b * xi + c
    #y_tangent = m * (x - xi) + yi
    #ax.plot(x, y_tangent, color="green")

    match path:
        case "":
            plt.show()
        case _:
            plt.savefig(path + ".pdf", bbox_inches='tight', format='pdf')
            plt.close(figure)

def plot_avg_metric(avg_scores : list, title : str, path : str = "") -> None:
    iterations = range(len(avg_scores))
    
    figure, ax = plt.subplots()

    avg_scores = [list(x) for x in zip(*avg_scores)]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, scores in enumerate(avg_scores):
        ax.plot(iterations, scores, label=CLASS_NAMES[i], color=colors[i])

    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Score")
    ax.grid()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc="upper left")
    
    match path:
        case "":
            plt.show()
        case _:
            plt.savefig(path + ".pdf", bbox_inches='tight', format='pdf')
            plt.close(figure)

def plot_pred(images : torch.Tensor, targets : torch.Tensor, predictions : torch.Tensor, classes : torch.Tensor, batch_size : int = BATCH_SIZE, plot_size : tuple = (0, 0), path : str = "") -> None:

    binary_seg = (predictions > BINARY_THRESHOLD).float()

    size = (batch_size * 3, 18) if plot_size == (0, 0) else plot_size

    fig, ax = plt.subplots(4, batch_size, figsize=size)

    for i in range(batch_size):
        
        img = images[i].permute(1, 2, 0)
        gt = targets[i].permute(1, 2, 0)
        pred = predictions[i].permute(1, 2, 0)
        seg = binary_seg[i].permute(1, 2, 0)
        cls = classes[i]

        ax[0][i].imshow(img, cmap='gray')
        ax[0][i].set_title(f"Image - Class: {cls}")
        ax[0][i].axis('off')

        ax[1][i].imshow(pred, cmap='gray')
        ax[1][i].set_title("Prediction")
        ax[1][i].axis('off')

        ax[2][i].imshow(seg, cmap='binary')
        ax[2][i].set_title(f"Thresholded prediction")
        ax[2][i].axis('off')
        
        ax[3][i].imshow(gt, cmap='binary')
        ax[3][i].set_title("Target")
        ax[3][i].axis('off')
    
    plt.tight_layout()

    match path:
        case "":
            plt.show()
        case _:
            plt.savefig(path + ".pdf", bbox_inches='tight', format='pdf')
            plt.close(fig)