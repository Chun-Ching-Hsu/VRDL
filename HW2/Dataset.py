import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CustomTrainSet(Dataset):
    def __init__(self, img_dir: str = "data/train", ann_path: str = "data/train.json", tfms=None):
        self.img_dir = img_dir
        self.tfms = tfms

        with open(ann_path, "r") as f:
            data = json.load(f)

        self.id_to_file = {item["id"]: item["file_name"] for item in data["images"]}
        self.id_to_ann = {k: [] for k in self.id_to_file}
        for ann in data["annotations"]:
            self.id_to_ann[ann["image_id"]].append(ann)

        self.ids = list(self.id_to_file.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        path = os.path.join(self.img_dir, self.id_to_file[img_id])
        img = Image.open(path).convert("RGB")

        boxes, classes = [], []
        for ann in self.id_to_ann[img_id]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            classes.append(ann["category_id"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(classes, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        if self.tfms:
            img = self.tfms(img)

        return img, target


class CustomTestSet(Dataset):
    def __init__(self, img_dir: str = "data/test", tfms=None):
        self.img_dir = img_dir
        self.tfms = tfms
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")

        if self.tfms:
            img = self.tfms(img)

        file_id = os.path.splitext(fname)[0]
        return img, file_id
