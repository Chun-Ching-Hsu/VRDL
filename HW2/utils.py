import os
import cv2
import json
import random
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def eval_map_score(preds, targets):
    """Compute mAP using torchmetrics' COCO-style evaluator."""
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds=preds, target=targets)
    results = metric.compute()
    return results["map"]


def draw_predictions_on_images(
    pred_json_path,
    img_dir,
    output_dir,
    draw_count=10,
    select_random=True,
    score_thresh=0.8,
):
    """Visualize detection results and save labeled images."""

    with open(pred_json_path, "r") as f:
        pred_data = json.load(f)

    # group predictions by image_id
    image_detections = {}
    for item in pred_data:
        if item["score"] >= score_thresh:
            image_detections.setdefault(item["image_id"], []).append(
                (item["category_id"] - 1, item["bbox"], item["score"])
            )

    # pick images to draw
    if select_random:
        chosen_ids = random.sample(list(image_detections.keys()), min(draw_count, len(image_detections)))
    else:
        chosen_ids = sorted(image_detections.keys())[:draw_count]

    scale_factor = 3  # enlarge output images
    os.makedirs(output_dir, exist_ok=True)

    for img_id in chosen_ids:
        img_path = os.path.join(img_dir, f"{img_id}.png")
        label_img_path = os.path.join(output_dir, f"{img_id}.png")

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

        for cls, bbox, conf in image_detections[img_id]:
            scaled_bbox = [int(coord * scale_factor) for coord in bbox]
            x, y, w, h = scaled_bbox

            # bounding box
            cv2.rectangle(img_scaled, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # class label
            cv2.putText(img_scaled, str(cls), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # confidence score
            label_width = cv2.getTextSize(str(cls), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            cv2.putText(img_scaled, f"{conf:.2f}", (x + label_width + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

        cv2.imwrite(label_img_path, img_scaled)

    print(f"[green]Labeled predictions saved to [bold]{output_dir}[/bold][/green]")
