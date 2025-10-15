from .constants import BATCH_SIZE
from .transforms import train_img_transform, train_gt_transform, test_img_transform, test_gt_transform
from .datasets import EBHI_SEG_Dataset, splitter, splitter_aug
from torch.utils.data import DataLoader

def build_loaders():
    train_images, train_ground_truths, train_class_names, test_images, test_ground_truths, test_class_names = splitter() # split the dataset

    train_set = EBHI_SEG_Dataset(train_images, train_ground_truths, train_class_names, train_img_transform, train_gt_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    test_set = EBHI_SEG_Dataset(test_images, test_ground_truths, test_class_names, test_img_transform, test_gt_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader

def build_loaders_aug():
    train_images, train_ground_truths, train_class_names, test_images, test_ground_truths, test_class_names = splitter_aug() # split and augment the dataset

    train_set = EBHI_SEG_Dataset(train_images, train_ground_truths, train_class_names, train_img_transform, train_gt_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    test_set = EBHI_SEG_Dataset(test_images, test_ground_truths, test_class_names, test_img_transform, test_gt_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader