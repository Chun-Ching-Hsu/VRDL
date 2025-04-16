import os
import json
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from rich import print
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from Dataset import TestDataset
from num_pred import num_pred
from utils import prediction_draw


def launch_prediction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images", type=str, default="data/test", help="Test image directory")
    parser.add_argument("--checkpoint_path", type=str, default="save_model/epoch10.pth", help="Trained model weight path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="save_result", help="Where to store prediction outputs")
    parser.add_argument("--score_threshold", type=float, default=0.7, help="Only boxes above this score will be kept")
    cfg = parser.parse_args()

    # prepare test dataset and loader
    test_set = TestDataset(root=cfg.test_images, transform=ToTensor())
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # load model and custom anchors
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=11)

    # load checkpoint
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=cfg.device))
    model.to(cfg.device).eval()

    os.makedirs(cfg.output_dir, exist_ok=True)
    all_preds = []

    with torch.no_grad():
        with Progress(
            TextColumn("[bold green]Predicting"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.1f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as bar:
            task = bar.add_task("Processing", total=len(test_loader))

            for imgs, img_ids in test_loader:
                imgs = [img.to(cfg.device) for img in imgs]
                results = model(imgs)

                for img_id, res in zip(img_ids, results):
                    boxes = res["boxes"].cpu().numpy()
                    scores = res["scores"].cpu().numpy()
                    labels = res["labels"].cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        if score < cfg.score_threshold:
                            continue
                        x_min, y_min, x_max, y_max = box
                        all_preds.append({
                            "image_id": int(img_id),
                            "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                            "score": float(score),
                            "category_id": int(label),
                        })

                bar.update(task, advance=1)

    pred_json_path = os.path.join(cfg.output_dir, "pred.json")
    with open(pred_json_path, "w") as f:
        json.dump(all_preds, f, indent=4)

    # post-processing: save csv + draw boxes
    num_pred(
        pred_json_path,
        os.path.join(cfg.output_dir, "pred.csv"),
        cfg.score_threshold,
    )

    prediction_draw(
        pred_json_path,
        cfg.test_images,
        cfg.output_dir,
        pred_threshold=cfg.score_threshold,
    )

    print(f"\n[bold green]✅ All done! Results saved to [yellow]{cfg.output_dir}[/yellow]")


if __name__ == "__main__":
    launch_prediction()
