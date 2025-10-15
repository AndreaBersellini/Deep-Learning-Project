from .constants import IMAGE_SIZE
import torch
import torchvision.transforms as transforms

train_img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

train_gt_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Lambda(lambda x: (x > 0.5).float()) # image binarization with 0.5 threshold
    #transforms.Normalize(mean=[0.5], std=[0.5])
])

test_img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

test_gt_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Lambda(lambda x: (x > 0.5).float()) # image binarization with 0.5 threshold
])