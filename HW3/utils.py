import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.amp import GradScaler, autocast
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on cell dataset")
    parser.add_argument("--data_dir", type=str, default="data/train", help="Training data directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for SGD")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")
    return parser.parse_args()


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Mask R-CNN on cell dataset")
    parser.add_argument("--data_dir", type=str, default="data/test", help="Directory for test images")
    parser.add_argument(
        "--model_path", type=str, default="checkpoints/model_last.pth",
        help="Path to saved model weights",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument(
        "--score_thresh", type=float, default=0.5,
        help="Threshold for predicted mask scores",
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mask_rcnn(num_classes: int = 5, pretrained: bool = True) -> torch.nn.Module:
    """Load Mask R-CNN model with custom number of classes."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    mask_in_feats = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in_feats, 256, num_classes)
    return model


def encode_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to COCO-style RLE."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    scaler: GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    progress = Progress(
        TextColumn("[green]Training:"), BarColumn(), TextColumn("{task.percentage:>3.1f}%"),
        TimeElapsedColumn(), TimeRemainingColumn(), transient=True
    )
    task = progress.add_task("train", total=len(loader))
    for images, targets in progress.track(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with autocast():
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        progress.update(task, advance=1)
        if task.completed % 10 == 0:
            torch.cuda.empty_cache()
    return total_loss / len(loader)


def evaluate_map(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    score_thresh: float = 0.5,
) -> float:
    model.eval()
    coco_gt = {"images": [], "annotations": [], "categories": []}
    for i in range(1, 5):
        coco_gt['categories'].append({"id": i, "name": f"class_{i}"})
    ann_id = 1
    predictions = []
    progress = Progress(
        TextColumn("[cyan]Evaluating:"), BarColumn(), TextColumn("{task.percentage:>3.1f}%"),
        TimeElapsedColumn(), TimeRemainingColumn(), transient=True
    )
    task = progress.add_task("eval", total=len(loader))
    for images, targets in progress.track(loader):
        imgs = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(imgs)
        for tgt, out in zip(targets, outputs):
            img_id = int(tgt['image_id'].item())
            h, w = imgs[0].shape[-2:]
            coco_gt['images'].append({
                'id': img_id, 'height': h, 'width': w, 'file_name': f"{img_id}.png"
            })
            # ground truth
            for m, lbl in zip(tgt['masks'], tgt['labels']):
                mask_np = m.cpu().numpy()
                coco_gt['annotations'].append({
                    'id': ann_id, 'image_id': img_id,
                    'category_id': int(lbl), 'segmentation': encode_rle(mask_np),
                    'area': int(mask_np.sum()), 'bbox': list(mask_utils.toBbox(encode_rle(mask_np))),
                    'iscrowd': 0
                })
                ann_id += 1
            # predictions
            for mask, lbl, score in zip(out['masks'], out['labels'], out['scores']):
                if score < score_thresh:
                    continue
                bin_mask = (mask[0].cpu().numpy() > 0.5)
                predictions.append({
                    'image_id': img_id, 'category_id': int(lbl),
                    'segmentation': encode_rle(bin_mask), 'score': float(score)
                })
        progress.update(task, advance=1)
    # Evaluate with temporary files
    with tempfile.NamedTemporaryFile('w+', suffix='.json') as gt_f, \
         tempfile.NamedTemporaryFile('w+', suffix='.json') as pred_f:
        json.dump(coco_gt, gt_f); gt_f.flush()
        json.dump(predictions, pred_f); pred_f.flush()
        coco_gt_obj = COCO(gt_f.name)
        coco_dt_obj = coco_gt_obj.loadRes(pred_f.name)
        evaluator = COCOeval(coco_gt_obj, coco_dt_obj, iouType='segm')
        evaluator.params.iouThrs = np.array([0.5])
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return evaluator.stats[0]
