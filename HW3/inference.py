import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from rich import print
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from dataset_refactored import CellInstanceDataset, get_transform
from training_utils_refactored import (
    parse_train_args,
    build_mask_rcnn,
    count_parameters,
    train_one_epoch,
    evaluate_map
)


def main():
    # Parse command-line arguments
    args = parse_train_args()
    device = torch.device(args.device)

    # Prepare dataset and split
    dataset = CellInstanceDataset(args.data_dir, transform=get_transform(train=True))
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              collate_fn=lambda batch: tuple(zip(*batch)))
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            collate_fn=lambda batch: tuple(zip(*batch)))

    # Build and prepare model
    model = build_mask_rcnn(num_classes=5, pretrained=True)
    model.to(device)
    print(f"[blue]Model trainable parameters: {count_parameters(model) / 1e6:.2f}M[/blue]")

    # Optimizer and scaler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()

    # Create checkpoint directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    train_losses, val_maps = [], []
    for epoch in range(1, args.epochs + 1):
        print(f"\n[yellow]Epoch {epoch}/{args.epochs}[/yellow]")
        # Train one epoch
        loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        # Validate mAP@0.5
        mAP50 = evaluate_map(model, val_loader, device)

        train_losses.append(loss)
        val_maps.append(mAP50)

        # Save model checkpoint
        ckpt_path = Path(args.save_dir) / f"epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[green]Saved checkpoint:[/] {ckpt_path}")
        print(f"Train Loss: {loss:.4f} | Val mAP@0.5: {mAP50:.4f}\n")

    # Save metrics
    np.save(Path(args.save_dir) / "train_losses.npy", np.array(train_losses))
    np.save(Path(args.save_dir) / "val_mAP50.npy", np.array(val_maps))
    print(f"[green]Training complete. Metrics saved in {args.save_dir}[/green]")


if __name__ == '__main__':
    main()