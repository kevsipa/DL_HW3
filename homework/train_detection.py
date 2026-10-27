import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
from .models import load_model, save_model


def compute_loss(logits, pred_depth, track, depth, segmentation_loss):
    seg_loss = segmentation_loss(logits, track)

    error = (pred_depth - depth).abs()
    boundary = (track > 0).to(error.dtype)

    boundary_loss = (
        (error * boundary).sum() / boundary.sum().clamp_min(1)
    )
    depth_loss = error.mean() + boundary_loss

    return seg_loss + 5.0 * depth_loss


def train(num_epoch=60, lr=1e-3, batch_size=32, num_workers=0, seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training detector on {device}")

    loaders = {
        split: load_data(
            f"drive_data/{split}",
            transform_pipeline="default",
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split in ("train", "val")
    }

    model = load_model("detector").to(device)

    segmentation_loss = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 3.0, 3.0], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch
    )

    log_dir = Path("logs") / f"detector_{datetime.now():%Y%m%d_%H%M%S}"
    best_score = -float("inf")

    with SummaryWriter(str(log_dir)) as logger:
        for epoch in range(num_epoch):
            validation = None

            for split, loader in loaders.items():
                training = split == "train"
                model.train(training)

                metric = DetectionMetric()
                loss_sum = 0.0
                count = 0

                with torch.set_grad_enabled(training):
                    for batch in loader:
                        image = batch["image"].to(device)
                        track = batch["track"].to(device).long()
                        depth = batch["depth"].to(device).float()

                        if training:
                            optimizer.zero_grad(set_to_none=True)

                        logits, pred_depth = model(image)

                        loss = compute_loss(
                            logits,
                            pred_depth,
                            track,
                            depth,
                            segmentation_loss,
                        )

                        if training:
                            loss.backward()
                            optimizer.step()
                        else:
                            metric.add(
                                logits.argmax(dim=1),
                                track,
                                pred_depth,
                                depth,
                            )

                        loss_sum += loss.item() * image.size(0)
                        count += image.size(0)

                logger.add_scalar(
                    f"{split}/loss", loss_sum / count, epoch + 1
                )

                if not training:
                    validation = metric.compute()

                    for name, value in validation.items():
                        logger.add_scalar(
                            f"val/{name}", value, epoch + 1
                        )

            # Checkpoint heuristic: reward IoU and penalize depth errors.
            score = (
                validation["iou"]
                - validation["abs_depth_error"]
                - validation["tp_depth_error"]
            )

            if score > best_score:
                best_score = score
                save_model(model)

            scheduler.step()

            print(
                f"Epoch {epoch + 1:03d}/{num_epoch}: "
                f"val_acc={validation['accuracy']:.4f} "
                f"val_iou={validation['iou']:.4f} "
                f"depth_MAE={validation['abs_depth_error']:.4f} "
                f"boundary_MAE={validation['tp_depth_error']:.4f}"
            )

    print("Best checkpoint: homework/detector.th")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))