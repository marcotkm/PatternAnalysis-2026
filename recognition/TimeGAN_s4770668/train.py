"""Stage 1: TimeGAN reconstruction pretraining."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import subprocess
import sys

from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    fit_standardiser,
    make_windows,
)
from modules import Embedder, Recovery


def prepare_windows(data_dir):
    messages, books = load_lobster(data_dir)
    splits = chronological_split(messages, books)

    _, train_features = build_features(splits["train"][1])
    mean, scale = fit_standardiser(train_features)

    tensors = {}
    for name in ("train", "validation"):
        split_books = splits[name][1]
        _, features = build_features(split_books)
        normalised = (features - mean) / scale

        windows, _ = make_windows(
            normalised, split_books, length=100
        )
        tensors[name] = torch.tensor(
            windows, dtype=torch.float32
        )

    return tensors, mean, scale


@torch.no_grad()
def evaluate(embedder, recovery, loader, device):
    embedder.eval()
    recovery.eval()
    squared_error = 0.0
    elements = 0

    for (batch,) in loader:
        batch = batch.to(device)
        prediction = recovery(embedder(batch))

        error = nn.functional.mse_loss(
            prediction, batch, reduction="sum"
        )
        if not torch.isfinite(error):
            raise RuntimeError("Non-finite validation error.")

        squared_error += error.item()
        elements += batch.numel()

    return squared_error / elements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--run-name", default="reconstruction_cpu")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if Path(args.run_name).name != args.run_name:
        raise ValueError("Use a simple folder name for --run-name.")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)

    # A new directory prevents accidentally overwriting an earlier run.
    run_dir = Path("checkpoints") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    tensors, mean, scale = prepare_windows(args.data_dir)

    train_loader = DataLoader(
        TensorDataset(tensors["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    validation_loader = DataLoader(
        TensorDataset(tensors["validation"]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    embedder = Embedder().to(device)
    recovery = Recovery().to(device)
    parameters = list(embedder.parameters()) + list(recovery.parameters())
    optimiser = torch.optim.Adam(parameters, lr=1e-3)

    initial_validation = evaluate(
        embedder, recovery, validation_loader, device
    )
    mean_predictor_mse = tensors["validation"].square().mean().item()

    print(f"Device: {device}")
    print(f"Training windows: {len(tensors['train'])}")
    print(f"Validation windows: {len(tensors['validation'])}")
    print(f"Initial validation MSE: {initial_validation:.6f}")
    print(f"Training-mean predictor validation MSE: {mean_predictor_mse:.6f}")

    best_validation = float("inf")
    started = time.perf_counter()

    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_mse", "validation_mse", "elapsed_seconds"
        ])

        for epoch in range(1, args.epochs + 1):
            embedder.train()
            recovery.train()
            total_error = 0.0
            elements = 0

            for (batch,) in train_loader:
                batch = batch.to(device)
                optimiser.zero_grad(set_to_none=True)

                prediction = recovery(embedder(batch))
                loss = nn.functional.mse_loss(prediction, batch)

                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")

                loss.backward()
                nn.utils.clip_grad_norm_(
                    parameters, max_norm=1.0, error_if_nonfinite=True
                )
                optimiser.step()

                total_error += loss.item() * batch.numel()
                elements += batch.numel()

            train_mse = total_error / elements
            validation_mse = evaluate(
                embedder, recovery, validation_loader, device
            )
            elapsed = time.perf_counter() - started

            writer.writerow([
                epoch, train_mse, validation_mse, elapsed
            ])
            handle.flush()

            print(
                f"Epoch {epoch}: train={train_mse:.6f} | "
                f"validation={validation_mse:.6f} | "
                f"elapsed={elapsed:.1f}s"
            )

            if validation_mse < best_validation:
                best_validation = validation_mse

                torch.save(
                    {
                        "stage": "reconstruction",
                        "epoch": epoch,
                        "embedder": embedder.state_dict(),
                        "recovery": recovery.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "mean": torch.tensor(mean),
                        "scale": torch.tensor(scale),
                        "validation_mse": validation_mse,
                        "config": {
                            "input_dim": 40,
                            "hidden_dim": 24,
                            "num_layers": 2,
                            "window_length": 100,
                            "batch_size": args.batch_size,
                            "learning_rate": 1e-3,
                            "seed": 42,
                            "split": "session_duration_70_15_15",
                        },
                    },
                    run_dir / "reconstruction_best.pt",
                )

    print(f"Best trained validation MSE: {best_validation:.6f}")
    print(f"Saved checkpoint and history in: {run_dir}")


def entrypoint():
    """Dispatch training stages and evaluation to existing implementations."""
    stage_parser = argparse.ArgumentParser(add_help=False)
    stage_parser.add_argument(
        "--stage",
        choices=["reconstruction", "supervisor", "joint", "evaluate"],
        default="reconstruction",
    )
    stage_args, remaining = stage_parser.parse_known_args()

    if stage_args.stage == "reconstruction":
        # Preserve the original reconstruction CLI and implementation.
        sys.argv = [sys.argv[0], *remaining]
        main()
        return

    scripts = {
        "supervisor": "train_supervisor.py",
        "joint": "train_joint.py",
        "evaluate": "evaluate.py",
    }
    script = Path(__file__).resolve().parent / scripts[stage_args.stage]
    subprocess.run(
        [sys.executable, str(script), *remaining],
        check=True,
    )


if __name__ == "__main__":
    entrypoint()