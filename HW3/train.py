import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torch.amp import GradScaler
from rich import print

from dataset_refactored import CellInstanceDataset, get_transform
from training_utils_refactored import (
    parse_train_args,
    build_mask_rcnn,
    count_parameters,
    train_one_epoch,
    evaluate_map,
)

def main() -> None:
    # Parse command-line arguments
    args = parse_train_args()
    device = torch.device(args.device)

    # Prepare dataset and dataloaders
    dataset = CellInstanceDataset(args.data_dir, transform=get_transform(train=True))
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    # Initialize model
    model = build_mask_rcnn(num_classes=5, pretrained=True)
    model.to(device)
    print(f"Model parameters: {count_parameters(model)/1e6:.2f}M")

    # Optimizer and scaler for mixed precision
    optimizer = torch.optim.SGD(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scaler = GradScaler()

    # Training loop
    metrics = {'losses': [], 'map50': []}
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n[yellow2]Epoch [{epoch}/{args.epochs}]")
        loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        map50 = evaluate_map(model, val_loader, device, score_thresh=0.5)

        metrics['losses'].append(loss)
        metrics['map50'].append(map50)

        ckpt_path = Path(args.save_dir) / f"epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Train Loss={loss:.4f}, Val mAP@0.5={map50:.4f}")
        print(f"Checkpoint saved to {ckpt_path}")

    # Save training logs
    np.save(Path(args.save_dir) / "train_losses.npy", np.array(metrics['losses']))
    np.save(Path(args.save_dir) / "val_map50.npy", np.array(metrics['map50']))
    print(f"\n[green]Saved training logs in {args.save_dir}")

if __name__ == '__main__':
    main()
