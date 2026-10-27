import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric
from .models import load_model, save_model


def train(
        num_epoch: int = 40, 
        lr: float = 1e-3, 
        batch_size: int = 128,
        num_workers: int = 0, 
        seed: int = 2024
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training classifier on {device}")

    loaders = {}

    for split, transform in [("train", "aug"), ("val", "default")]:
        dataset = load_data(
            f"classification_data/{split}",
            transform_pipeline=transform,
            return_dataloader=False,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

    model = load_model("classifier").to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch
    )

    log_dir = Path("logs") / f"classifier_{datetime.now():%Y%m%d_%H%M%S}"
    best_accuracy = -1.0

    with SummaryWriter(str(log_dir)) as logger:
        for epoch in range(num_epoch):
            results = {}

            for split, loader in loaders.items():
                training = split == "train"
                model.train(training)

                metric = AccuracyMetric()
                loss_sum = 0.0
                count = 0

                with torch.set_grad_enabled(training):
                    for images, labels in loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        if training:
                            optimizer.zero_grad(set_to_none=True)

                        logits = model(images)
                        loss = criterion(logits, labels)

                        if training:
                            loss.backward()
                            optimizer.step()

                        metric.add(logits.detach().argmax(dim=1), labels)

                        loss_sum += loss.item() * images.size(0)
                        count += images.size(0)

                results[split] = metric.compute()["accuracy"]

                logger.add_scalar(
                    f"{split}/loss", loss_sum / count, epoch + 1
                )
                logger.add_scalar(
                    f"{split}/accuracy", results[split], epoch + 1
                )

            if results["val"] > best_accuracy:
                best_accuracy = results["val"]
                save_model(model)

            scheduler.step()

            print(
                f"Epoch {epoch + 1:03d}/{num_epoch}: "
                f"train_acc={results['train']:.4f} "
                f"val_acc={results['val']:.4f} "
                f"best_val_acc={best_accuracy:.4f}"
            )

    print("Best checkpoint: homework/classifier.th")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))