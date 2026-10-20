"""Evaluate reconstruction by feature group on validation data."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    make_windows,
)
from modules import Embedder, Recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)

    # Load only your own locally produced checkpoint.
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]

    embedder = Embedder(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    recovery = Recovery(
        output_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    embedder.load_state_dict(checkpoint["embedder"])
    recovery.load_state_dict(checkpoint["recovery"])
    embedder.eval()
    recovery.eval()

    messages, books = load_lobster(args.data_dir)
    validation_books = chronological_split(messages, books)["validation"][1]
    _, features = build_features(validation_books)

    # Reuse the training statistics saved with this model.
    mean = checkpoint["mean"].numpy()
    scale = checkpoint["scale"].numpy()
    normalised = (features - mean) / scale

    windows, _ = make_windows(
        normalised, validation_books, length=config["window_length"]
    )
    loader = DataLoader(
        TensorDataset(torch.tensor(windows, dtype=torch.float32)),
        batch_size=32,
        shuffle=False,
    )

    model_error = torch.zeros(40, dtype=torch.float64)
    baseline_error = torch.zeros(40, dtype=torch.float64)
    observations = 0

    with torch.no_grad():
        for (batch,) in loader:
            prediction = recovery(embedder(batch))
            if not torch.isfinite(prediction).all():
                raise ValueError("Non-finite reconstruction.")

            model_error += (
                (prediction - batch).double().square().sum(dim=(0, 1))
            )
            baseline_error += batch.double().square().sum(dim=(0, 1))
            observations += batch.shape[0] * batch.shape[1]

    model_mse = (model_error / observations).numpy()
    baseline_mse = (baseline_error / observations).numpy()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    for label, columns in [
        ("Price returns", slice(0, 20)),
        ("Log volumes", slice(20, 40)),
        ("All features", slice(0, 40)),
    ]:
        error = model_mse[columns].mean()
        baseline = baseline_mse[columns].mean()
        improvement = 100 * (1 - error / baseline) if baseline > 0 else np.nan

        print(
            f"{label}: model MSE={error:.6f} | "
            f"mean-predictor MSE={baseline:.6f} | "
            f"improvement={improvement:.2f}%"
        )


if __name__ == "__main__":
    main()