import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, PROCESSED_DATA_DIR, BATCH_SIZE


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])


class SkinDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.albu_transform = transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(self.loader(path))

        if self.albu_transform:
            image = self.albu_transform(image=image)["image"]

        return image, label


def get_dataloaders():
    train_ds = SkinDataset(PROCESSED_DATA_DIR / "train", get_train_transforms())
    val_ds = SkinDataset(PROCESSED_DATA_DIR / "val", get_val_transforms())
    test_ds = SkinDataset(PROCESSED_DATA_DIR / "test", get_val_transforms())
    targets = [label for _, label in train_ds.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"✅ Dataloaders ready (with oversampling):")
    print(f"   Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"   Val:   {len(val_ds)} images, {len(val_loader)} batches")
    print(f"   Test:  {len(test_ds)} images, {len(test_loader)} batches")
    print(f"   Classes: {train_ds.classes}")

    return train_loader, val_loader, test_loader