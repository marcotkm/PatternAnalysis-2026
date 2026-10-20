"""Check learning on a fixed batch of real training windows."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    fit_standardiser,
    make_windows,
)
from modules import Embedder, Recovery, Supervisor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)

    messages, books = load_lobster(args.data_dir)
    splits = chronological_split(messages, books)
    train_books = splits["train"][1]

    _, features = build_features(train_books)
    mean, scale = fit_standardiser(features)
    normalised = (features - mean) / scale
    windows, _ = make_windows(normalised, train_books, length=100)

    # Sample across the training period rather than only its opening.
    indices = rng.choice(len(windows), size=16, replace=False)
    batch = torch.tensor(windows[indices], dtype=torch.float32)

    embedder = Embedder()
    recovery = Recovery()
    parameters = list(embedder.parameters()) + list(recovery.parameters())
    optimiser = torch.optim.Adam(parameters, lr=1e-3)

    def reconstruction_loss():
        reconstructed = recovery(embedder(batch))
        return nn.functional.mse_loss(reconstructed, batch)

    with torch.no_grad():
        initial_loss = reconstruction_loss().item()
        # A zero standardised prediction corresponds to training means.
        zero_prediction_loss = batch.square().mean().item()

    print(f"Batch shape: {tuple(batch.shape)}")
    print(f"Training-mean prediction MSE: {zero_prediction_loss:.6f}")
    print(f"Initial reconstruction MSE: {initial_loss:.6f}")

    started = time.perf_counter()

    for step in range(1, 301):
        optimiser.zero_grad(set_to_none=True)
        loss = reconstruction_loss()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite reconstruction loss.")

        loss.backward()
        nn.utils.clip_grad_norm_(
            parameters, max_norm=1.0, error_if_nonfinite=True
        )
        optimiser.step()

        if step % 50 == 0:
            print(f"Step {step:3d}: MSE={loss.item():.6f}")

    with torch.no_grad():
        final_loss = reconstruction_loss().item()

    print(f"Final reconstruction MSE: {final_loss:.6f}")
    print(f"Elapsed CPU time: {time.perf_counter() - started:.1f}s")
    print(
        "Reduction from initial loss: "
        f"{100 * (initial_loss - final_loss) / initial_loss:.2f}%"
    )

        # Freeze the learned representation for this isolated supervisor check.
    embedder.eval()
    recovery.eval()

    with torch.no_grad():
        latent = embedder(batch).detach()

    inputs = latent[:, :-1, :]
    targets = latent[:, 1:, :]

    supervisor = Supervisor()
    supervisor_optimiser = torch.optim.Adam(
        supervisor.parameters(), lr=1e-3
    )

    with torch.no_grad():
        initial_supervisor_loss = nn.functional.mse_loss(
            supervisor(inputs), targets
        ).item()
        persistence_loss = nn.functional.mse_loss(
            inputs, targets
        ).item()

    print("\nSupervisor check")
    print(f"Initial supervisor MSE: {initial_supervisor_loss:.6f}")
    print(f"Latent persistence MSE: {persistence_loss:.6f}")

    for step in range(1, 301):
        supervisor_optimiser.zero_grad(set_to_none=True)
        predictions = supervisor(inputs)
        loss = nn.functional.mse_loss(predictions, targets)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite supervisor loss.")

        loss.backward()
        nn.utils.clip_grad_norm_(
            supervisor.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        supervisor_optimiser.step()

        if step % 100 == 0:
            print(f"Supervisor step {step}: MSE={loss.item():.6f}")

    with torch.no_grad():
        final_supervisor_loss = nn.functional.mse_loss(
            supervisor(inputs), targets
        ).item()

    print(f"Final supervisor MSE: {final_supervisor_loss:.6f}")


if __name__ == "__main__":
    main()