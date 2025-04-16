import os
import json
import pandas as pd
import argparse
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn


def generate_digit_predictions(pred_json, output_csv, conf_thresh):
    with open(pred_json, "r") as f:
        box_list = json.load(f)

    pred_df = pd.DataFrame(columns=["image_id", "pred_label"])
    image_boxes_map = {}  # key: image_id, value: list of (digit, x_center)

    # Step 1: filter boxes
    with Progress(
        TextColumn("[bold green]Step 1: Filter Boxes"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as p:
        task = p.add_task("Filtering...", total=len(box_list))
        for box in box_list:
            if box["score"] >= conf_thresh:
                x_center = box["bbox"][0] + box["bbox"][2] / 2
                image_boxes_map.setdefault(box["image_id"], []).append(
                    (box["category_id"] - 1, x_center)
                )
            p.update(task, advance=1)

    # Step 2: make predictions
    all_image_ids = {entry["image_id"] for entry in box_list}
    with Progress(
        TextColumn("[cyan bold]Step 2: Compose Predictions"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as p:
        task = p.add_task("Composing...", total=len(all_image_ids))
        for img_id in sorted(all_image_ids):
            if img_id not in image_boxes_map:
                pred_df.loc[len(pred_df)] = [img_id, -1]
            else:
                sorted_digits = sorted(image_boxes_map[img_id], key=lambda x: x[1])
                composed_number = int("".join(str(d[0]) for d in sorted_digits))
                pred_df.loc[len(pred_df)] = [img_id, composed_number]
            p.update(task, advance=1)

    pred_df.to_csv(output_csv, index=False)
    print(f"[bold green]✅ Saved predictions to:[/bold green] [yellow]{output_csv}[/yellow]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--box_json", type=str, default="save_result/pred.json", help="Prediction boxes (json)"
    )
    parser.add_argument(
        "-o", "--output_csv", type=str, default="save_result/pred.csv", help="Final digit CSV output"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.7, help="Confidence threshold"
    )
    args = parser.parse_args()

    generate_digit_predictions(args.box_json, args.output_csv, args.threshold)


if __name__ == "__main__":
    main()
