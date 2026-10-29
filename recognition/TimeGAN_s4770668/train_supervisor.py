"""Stage 2: supervisor pretraining with a frozen embedder.

Sources:
    Yoon, Jarrett and van der Schaar (2019),
    "Time-series Generative Adversarial Networks", NeurIPS.
    https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

    Authors' reference implementation:
    https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py

The architecture and/or training objectives are adapted for this project's
PyTorch implementation. This is not an exact reproduction of the reference
implementation. See README.md for implementation differences and AI usage."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    make_windows,
)
from modules import Embedder, Supervisor


@torch.no_grad()
def evaluate(supervisor, loader):
    supervisor.eval()
    model_error = 0.0
    persistence_error = 0.0
    elements = 0

    for (latent,) in loader:
        inputs = latent[:, :-1]
        targets = latent[:, 1:]
        prediction = supervisor(inputs)

        error = nn.functional.mse_loss(
            prediction, targets, reduction="sum"
        )
        if not torch.isfinite(error):
            raise RuntimeError("Non-finite validation loss.")

        model_error += error.item()
        persistence_error += nn.functional.mse_loss(
            inputs, targets, reduction="sum"
        ).item()
        elements += targets.numel()

    return model_error / elements, persistence_error / elements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--run-name", default="supervisor_cpu_01")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("Epochs must be positive.")
    if Path(args.run_name).name != args.run_name:
        raise ValueError("Use a simple folder name for --run-name.")

    torch.manual_seed(42)
    torch.set_num_threads(2)

    source = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if source["stage"] != "reconstruction":
        raise ValueError("Expected a reconstruction checkpoint.")

    config = source["config"]
    embedder = Embedder(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    embedder.load_state_dict(source["embedder"])
    embedder.eval()
    embedder.requires_grad_(False)

    messages, books = load_lobster(args.data_dir)
    splits = chronological_split(messages, books)
    mean = source["mean"].numpy()
    scale = source["scale"].numpy()

    latent_by_split = {}

    # Cache latents: the embedder stays frozen throughout this stage.
    for name in ("train", "validation"):
        split_books = splits[name][1]
        _, features = build_features(split_books)
        windows, _ = make_windows(
            (features - mean) / scale,
            split_books,
            length=config["window_length"],
        )

        feature_loader = DataLoader(
            TensorDataset(torch.tensor(windows, dtype=torch.float32)),
            batch_size=32,
            shuffle=False,
        )

        chunks = []
        with torch.no_grad():
            for (batch,) in feature_loader:
                latent = embedder(batch)
                if not torch.isfinite(latent).all():
                    raise RuntimeError("Non-finite latent values.")
                chunks.append(latent)

        latent_by_split[name] = torch.cat(chunks, dim=0)

    train_loader = DataLoader(
        TensorDataset(latent_by_split["train"]),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    validation_loader = DataLoader(
        TensorDataset(latent_by_split["validation"]),
        batch_size=32,
        shuffle=False,
    )

    supervisor = Supervisor(
        hidden_dim=config["hidden_dim"], num_layers=1
    )
    optimiser = torch.optim.Adam(supervisor.parameters(), lr=1e-3)

    initial_mse, persistence_mse = evaluate(
        supervisor, validation_loader
    )

    print(f"Training latent windows: {len(latent_by_split['train'])}")
    print(f"Validation latent windows: {len(latent_by_split['validation'])}")
    print(f"Initial validation MSE: {initial_mse:.6f}")
    print(f"Validation persistence MSE: {persistence_mse:.6f}")

    run_dir = Path("checkpoints") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    best_mse = float("inf")
    started = time.perf_counter()

    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch", "train_mse", "validation_mse",
            "persistence_mse", "elapsed_seconds",
        ])

        for epoch in range(1, args.epochs + 1):
            supervisor.train()
            total_error = 0.0
            elements = 0

            for (latent,) in train_loader:
                inputs = latent[:, :-1]
                targets = latent[:, 1:]

                optimiser.zero_grad(set_to_none=True)
                prediction = supervisor(inputs)
                loss = nn.functional.mse_loss(prediction, targets)

                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")

                loss.backward()
                nn.utils.clip_grad_norm_(
                    supervisor.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimiser.step()

                total_error += loss.item() * targets.numel()
                elements += targets.numel()

            train_mse = total_error / elements
            validation_mse, persistence_mse = evaluate(
                supervisor, validation_loader
            )
            elapsed = time.perf_counter() - started

            writer.writerow([
                epoch, train_mse, validation_mse,
                persistence_mse, elapsed,
            ])
            handle.flush()

            print(
                f"Epoch {epoch}: train={train_mse:.6f} | "
                f"validation={validation_mse:.6f} | "
                f"elapsed={elapsed:.1f}s"
            )

            if validation_mse < best_mse:
                best_mse = validation_mse

                # Preserve pretrained networks and preprocessing metadata.
                checkpoint = {
                    key: source[key]
                    for key in ("embedder", "recovery", "mean", "scale")
                }
                checkpoint.update({
                    "stage": "supervisor",
                    "epoch": epoch,
                    "supervisor": supervisor.state_dict(),
                    "supervisor_optimiser": optimiser.state_dict(),
                    "validation_mse": validation_mse,
                    "persistence_mse": persistence_mse,
                    "reconstruction_epoch": source["epoch"],
                    "reconstruction_validation_mse":
                        source["validation_mse"],
                    "source_checkpoint": str(args.checkpoint),
                    "config": {
                        **config,
                        "supervisor_num_layers": 1,
                        "supervisor_learning_rate": 1e-3,
                        "supervisor_batch_size": 32,
                    },
                })
                torch.save(
                    checkpoint, run_dir / "supervisor_best.pt"
                )

    print(f"Best trained validation MSE: {best_mse:.6f}")
    print(f"Validation persistence MSE: {persistence_mse:.6f}")
    print(f"Saved checkpoint and history in: {run_dir}")


if __name__ == "__main__":
    main()