import torch

"""HYPERPARAMETERS AND DIRECTORIES"""

DATASET_DIR = "EBHI-SEG"
DATASET_AUG_DIR = "EBHI-SEG-AUG"
CLASS_NAMES = ["Adenocarcinoma", "High-grade_IN", "Low-grade_IN", "Normal", "Polyp", "Serrated_adenoma"]

MODEL_DIR = "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
MIN_EPOCHS = 10 # minimum number of epochs to train before early stopping
MAX_EPOCHS = 50 # maximum number of epochs
EARLY_STOP_TAIL = 10 # number of element (validation losses) on which calculate the quadratic interpolation and the angular coefficient
EARLY_STOP_COUNT = 5 # number of time the angular coefficient doesn't reach the threshold after which the model will stop train
EARLY_STOP_COEFF = -1e-4 # angular coefficient threshold of the validation loss trend

IMAGE_SIZE = (64, 64) # (224, 224) original
IN_CHANNELS = 1
OUT_CHANNELS = 1
BINARY_THRESHOLD = 0.5 # threshold for binarization of gray scale images (set to 0 to binarize with Otsu's method from cv2)