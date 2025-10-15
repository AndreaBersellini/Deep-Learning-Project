from .constants import DATASET_DIR, DATASET_AUG_DIR, CLASS_NAMES, IN_CHANNELS
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def image_transform(image: Image.Image, ground_truth: Image.Image) -> tuple[Image.Image, Image.Image]:
    # ----- RANDOM FLIP -----
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        ground_truth = ground_truth.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        ground_truth = ground_truth.transpose(Image.FLIP_TOP_BOTTOM)

    # ----- RANDOM ROTATION -----
    if random.random() < 0.5:
        angle = random.uniform(-45, 45)
        image = image.rotate(angle)
        ground_truth = ground_truth.rotate(angle)

    # ----- RANDOM SCALING -----
    if random.random() < 0.2:
        scale = random.uniform(1.1, 1.4)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        ground_truth = ground_truth.resize((int(ground_truth.width * scale), int(ground_truth.height * scale)), Image.Resampling.LANCZOS)

    return image, ground_truth

def augment_dataset(data_count : list):
    for idx, cls in enumerate(CLASS_NAMES):
        img_dir = os.path.join(DATASET_DIR, cls, 'image')
        gt_dir = os.path.join(DATASET_DIR, cls, 'label')

        imgs = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
        gts = [os.path.join(gt_dir, gt) for gt in sorted(os.listdir(gt_dir))]

        # ----- CLEAR THE CURRENT AUGMENTED DATASET -----
        for filename in os.listdir(os.path.join(DATASET_AUG_DIR, cls, 'image')):
            image = os.path.join(os.path.join(DATASET_AUG_DIR, cls, 'image'), filename)
            os.unlink(image)
        for filename in os.listdir(os.path.join(DATASET_AUG_DIR, cls, 'label')):
            image = os.path.join(os.path.join(DATASET_AUG_DIR, cls, 'label'), filename)
            os.unlink(image)

        # ----- GENERATE SAMPLES TO BALANCE THE CLASSES ----
        for x in range(max(data_count) - data_count[idx]):
            rnd_idx = random.randint(0, data_count[idx] - 1) # select a random image and graund truth to augment
            img_path = imgs[rnd_idx]
            gt_path = gts[rnd_idx]

            base_img = Image.open(imgs[rnd_idx])
            base_gt = Image.open(gts[rnd_idx])

            aug_img, aug_gt = image_transform(base_img, base_gt) # apply transformations to the image and ground truth

            img_name, img_ext = os.path.splitext(os.path.basename(img_path))
            gt_name, gt_ext = os.path.splitext(os.path.basename(gt_path))

            aug_img_path = os.path.join(DATASET_AUG_DIR, cls, 'image', f"{img_name}-AUG-{x}{img_ext}")
            aug_gt_path = os.path.join(DATASET_AUG_DIR, cls, 'label', f"{gt_name}-AUG-{x}{gt_ext}")

            aug_img.save(aug_img_path)
            aug_gt.save(aug_gt_path)

def splitter():
    train_images, test_images = [], []
    train_ground_truths, test_ground_truths = [], []
    train_class_names, test_class_names = [], []
    
    # ----- DIVIDE THE SAMPLES IN TRAIN/TEST SET -----
    for idx, cls in enumerate(CLASS_NAMES):
        img_dir = os.path.join(DATASET_DIR, cls, 'image')
        gt_dir = os.path.join(DATASET_DIR, cls, 'label')

        imgs = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
        gts = [os.path.join(gt_dir, gt) for gt in sorted(os.listdir(gt_dir))]

        train_imgs, test_imgs, train_gts, test_gts = train_test_split(imgs, gts, train_size = 0.8, shuffle = True)

        train_images.extend(train_imgs)
        train_ground_truths.extend(train_gts)
        train_class_names.extend([idx] * len(train_imgs))

        test_images.extend(test_imgs)
        test_ground_truths.extend(test_gts)
        test_class_names.extend([idx] * len(test_imgs))

    return train_images, train_ground_truths, train_class_names, test_images, test_ground_truths, test_class_names

def splitter_aug():
    train_images, test_images = [], []
    train_ground_truths, test_ground_truths = [], []
    train_class_names, test_class_names = [], []
    data_count = []
    
    # ----- DIVIDE THE SAMPLES IN TRAIN/TEST SET -----
    for idx, cls in enumerate(CLASS_NAMES):
        img_dir = os.path.join(DATASET_DIR, cls, 'image')
        gt_dir = os.path.join(DATASET_DIR, cls, 'label')

        imgs = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
        gts = [os.path.join(gt_dir, gt) for gt in sorted(os.listdir(gt_dir))]

        train_imgs, test_imgs, train_gts, test_gts = train_test_split(imgs, gts, train_size = 0.8, shuffle = True)

        data_count.append(len(train_imgs)) # track the number of samples per class in the train set to perform data augmentation

        train_images.extend(train_imgs)
        train_ground_truths.extend(train_gts)
        train_class_names.extend([idx] * len(train_imgs))

        test_images.extend(test_imgs)
        test_ground_truths.extend(test_gts)
        test_class_names.extend([idx] * len(test_imgs))

    augment_dataset(data_count) # generate augmented data 
    
    # ----- INCLUDE AUGMENTED DATA -----
    for idx, cls in enumerate(CLASS_NAMES):
        aug_img_dir = os.path.join(DATASET_AUG_DIR, cls, 'image')
        aug_gt_dir = os.path.join(DATASET_AUG_DIR, cls, 'label')

        aug_imgs = [os.path.join(aug_img_dir, img) for img in sorted(os.listdir(aug_img_dir))]
        aug_gts = [os.path.join(aug_gt_dir, gt) for gt in sorted(os.listdir(aug_gt_dir))]

        train_images.extend(aug_imgs)
        train_ground_truths.extend(aug_gts)
        train_class_names.extend([idx] * len(aug_imgs))

    return train_images, train_ground_truths, train_class_names, test_images, test_ground_truths, test_class_names

class EBHI_SEG_Dataset(Dataset):
    def __init__(self, images, ground_truths, class_names, img_transform = None, gt_transform = None):
        self._images = images
        self._gts = ground_truths
        self._class_names = class_names
        self._img_transform = img_transform
        self._gt_transform = gt_transform

    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, index):
        img_path = self._images[index]
        gt_path = self._gts[index]
        class_name = self._class_names[index]

        image = Image.open(img_path)
        ground_truth = Image.open(gt_path)

        match IN_CHANNELS:
            case 1:
                image = image.convert("L") #convert to gray-scale

        # ----- APPLY IMAGE TRANSFORMATION -----
        if self._img_transform is not None : image = self._img_transform(image)
        if self._gt_transform is not None : ground_truth = self._gt_transform(ground_truth)

        return image, ground_truth, class_name