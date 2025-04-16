import os
import json
import argparse
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich import print
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from Dataset import CustomTrainSet, CustomTestSet  
from utils import COCOmap


def run_training(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0

    with Progress(TextColumn("[bold green]Train"), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", TimeElapsedColumn(), TimeRemainingColumn()) as bar:
        task = bar.add_task("Step", total=len(loader))

        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with autocast():
                losses = model(imgs, targets)
                loss = sum(losses.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            bar.update(task, advance=1)

    return total_loss / len(loader)


def run_evaluation(model, loader, device):
    model.eval()
    pred_results, gt_results = [], []

    with Progress(TextColumn("[cyan]Valid"), BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%", TimeElapsedColumn(), TimeRemainingColumn()) as bar:
        task = bar.add_task("Step", total=len(loader))

        with torch.no_grad():
            for imgs, targets in loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(imgs)
                pred_results.extend({k: v.cpu() for k, v in o.items()} for o in outputs)
                gt_results.extend({k: v.cpu() for k, v in t.items()} for t in targets)
                bar.update(task, advance=1)

    return COCOmap(pred_results, gt_results)


def locate_latest_checkpoint(folder):
    ckpts = [f for f in os.listdir(folder) if f.startswith("epoch") and f.endswith(".pth")]
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda n: int(n.replace("epoch", "").replace(".pth", "")))
    latest = ckpts[-1]
    return os.path.join(folder, latest), int(latest.replace("epoch", "").replace(".pth", ""))


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--train_json", type=str, default="data/train.json")
    parser.add_argument("--val_dir", type=str, default="data/valid")
    parser.add_argument("--val_json", type=str, default="data/valid.json")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="save_model")
    parser.add_argument("--partial_train", type=float, default=-1)
    parser.add_argument("--partial_val", type=float, default=-1)
    args = parser.parse_args()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 11)
    model.to(args.device)

    ckpt_path, start_epoch = locate_latest_checkpoint(args.ckpt_dir)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
        print(f"[yellow]Resumed from {ckpt_path}[/yellow]")
    else:
        print(f"[yellow]Starting from scratch[/yellow]")

    # transforms
    aug_train = T.Compose([
        T.RandomResizedCrop(512, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(),
    ])
    aug_val = T.ToTensor()

    # dataset
    train_set = CustomTrainSet(args.train_dir, args.train_json, transform=aug_train)
    val_set = CustomTestSet(args.val_dir, args.val_json, transform=aug_val)

    if 0 < args.partial_train < 1:
        n = int(len(train_set) * args.partial_train)
        train_set = Subset(train_set, list(range(n)))
    if 0 < args.partial_val < 1:
        n = int(len(val_set) * args.partial_val)
        val_set = Subset(val_set, list(range(n)))

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.jobs, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.jobs, collate_fn=lambda x: tuple(zip(*x)))

    # freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    losses, maps = [], []
    unfreeze_at = 3

    for ep in range(start_epoch, args.epochs):
        if ep == unfreeze_at:
            print("[cyan]Backbone unfrozen![/cyan]")
            for param in model.backbone.parameters():
                param.requires_grad = True

        print(f"\n[bold magenta]Epoch {ep+1}/{args.epochs}[/bold magenta]")
        loss = run_training(model, train_loader, opt, args.device, scaler)
        val_map = run_evaluation(model, val_loader, args.device)

        losses.append(loss)
        maps.append(val_map)
        scheduler.step()

        print(f"[green]Loss: {loss:.4f} | mAP: {val_map:.4f}[/green]")

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"epoch{ep+1}.pth"))

    np.save(os.path.join(args.ckpt_dir, "loss.npy"), np.array(losses))
    np.save(os.path.join(args.ckpt_dir, "mAP.npy"), np.array(maps))
    print(f"\n[bold green]Results saved to {args.ckpt_dir}[/bold green]")


if __name__ == "__main__":
    launch()
