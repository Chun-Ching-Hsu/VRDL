import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tifffile import imread
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class CellInstanceDataset(Dataset):
    """
    PyTorch Dataset for instance segmentation of cells.

    Each sample directory under `data_dir` should contain:
      - image.tif: RGB image
      - class{1..4}.tif: mask files with instance IDs per cell class
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.sample_dirs = [p for p in self.data_dir.iterdir() if p.is_dir()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int):
        sample_path = self.sample_dirs[index]
        image = Image.open(sample_path / 'image.tif').convert('RGB')
        masks, labels = self._parse_masks(sample_path)
        targets = self._build_target(masks, labels, index, image.size)

        if self.transform:
            image, targets = self.transform(image, targets)
        return image, targets

    def _parse_masks(self, sample_path: Path):
        masks = []
        labels = []
        # Iterate over four classes
        for cls in range(1, 5):
            mask_path = sample_path / f'class{cls}.tif'
            if not mask_path.exists():
                continue
            raw = imread(mask_path)
            # Extract each instance by unique ID
            for inst_id in np.unique(raw):
                if inst_id == 0:
                    continue
                bin_mask = (raw == inst_id)
                masks.append(bin_mask)
                labels.append(cls)
        return masks, labels

    def _build_target(self, masks: list, labels: list, idx: int, img_size: tuple):
        if masks:
            mask_tensor = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        else:
            # Create an empty mask tensor if no instances
            w, h = img_size
            mask_tensor = torch.zeros((0, h, w), dtype=torch.uint8)

        boxes = []
        # Compute bounding box for each instance mask
        for m in mask_tensor:
            ys, xs = torch.nonzero(m, as_tuple=True)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            boxes.append([xmin, ymin, xmax, ymax])

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': mask_tensor,
            'image_id': torch.tensor([idx])
        }
        return target

class AugmentationPipeline:
    """
    Data augmentation and preprocessing for instance segmentation.
    Applies random flips and color jitter during training.
    """
    def __init__(self, train: bool = True):
        self.train = train
        self.color_jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.02)

    def __call__(self, image: Image.Image, target: dict):
        if self.train:
            # Random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                target['masks'] = target['masks'].flip(-1)
                width = image.width
                target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
            # Random vertical flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                target['masks'] = target['masks'].flip(-2)
                height = image.height
                target['boxes'][:, [1, 3]] = height - target['boxes'][:, [3, 1]]
            # Color jitter
            image = self.color_jitter(image)
        # Convert PIL image to tensor
        image_tensor = TF.to_tensor(image)
        return image_tensor, target

def get_transform(train: bool = True) -> AugmentationPipeline:
    """
    Returns the augmentation pipeline.
    """
    return AugmentationPipeline(train)
