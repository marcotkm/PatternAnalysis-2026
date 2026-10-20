"""Shared evaluation under development: VAR generation alignment."""

import argparse
from pathlib import Path

import numpy as np
import torch

from modules import Generator, Supervisor, Recovery

from baseline import fit_var1
from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    fit_standardiser,
)


def generate_var_transitions(
    intercept, transition, residuals, initial_states,
    length=100, seed=42,
):
    """Return new transitions only; do not include copied initial states."""
    rng = np.random.default_rng(seed)
    state = initial_states.copy()
    centred_residuals = residuals - residuals.mean(axis=0)

    generated = np.empty(
        (len(state), length, state.shape[1]), dtype=np.float64
    )

    for step in range(length):
        indices = rng.integers(
            0, len(centred_residuals), size=len(state)
        )
        state = (
            intercept + state @ transition + centred_residuals[indices]
        )
        generated[:, step] = state

    if not np.isfinite(generated).all():
        raise ValueError("Non-finite VAR output.")

    return generated


def reconstruct_books(generated, mean, scale, initial_prices):
    """Undo scaling and reconstruct raw continuous prices and volumes."""
    features = generated.astype(np.float64) * scale + mean

    with np.errstate(over="raise", invalid="raise"):
        prices = initial_prices[:, None, :] * np.exp(
            np.cumsum(features[:, :, :20], axis=1)
        )
        volumes = np.expm1(features[:, :, 20:])

    if not np.isfinite(prices).all() or not np.isfinite(volumes).all():
        raise ValueError("Non-finite reconstructed books.")

    return prices, volumes

def generate_timegan(checkpoint_path, mean, scale, count=32, length=100, seed=42):
    """Generate new feature sequences using a trained TimeGAN checkpoint."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint["stage"] != "joint":
        raise ValueError("Expected a joint-training checkpoint.")

    # Both models must use the same feature scaling.
    np.testing.assert_allclose(
        checkpoint["mean"].numpy(), mean, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        checkpoint["scale"].numpy(), scale, rtol=0, atol=1e-12
    )

    config = checkpoint["config"]
    if config["window_length"] != length:
        raise ValueError("Requested length differs from training window length.")

    hidden = config["hidden_dim"]
    generator = Generator(
        config["noise_dim"], hidden, config["generator_num_layers"]
    )
    supervisor = Supervisor(
        hidden, config["supervisor_num_layers"]
    )
    recovery = Recovery(
        config["input_dim"], hidden, config["num_layers"]
    )

    for name, model in [
        ("generator", generator),
        ("supervisor", supervisor),
        ("recovery", recovery),
    ]:
        model.load_state_dict(checkpoint[name])
        model.eval()

    rng = torch.Generator().manual_seed(seed)
    noise = torch.randn(
        count, length, config["noise_dim"], generator=rng
    )

    with torch.no_grad():
        generated = recovery(supervisor(generator(noise))).numpy()

    if not np.isfinite(generated).all():
        raise ValueError("Non-finite TimeGAN output.")

    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    messages, books = load_lobster(args.data_dir)
    train_books = chronological_split(messages, books)["train"][1]

    _, features = build_features(train_books)
    mean, scale = fit_standardiser(features)
    training = (features - mean) / scale

    intercept, transition, residuals, _ = fit_var1(training)

    # Fix the initial states and anchors independently of generation noise.
    anchor_rng = np.random.default_rng(123)
    indices = anchor_rng.integers(
        0, len(training) - 1, size=32
    )

    initial_states = training[indices]
    initial_prices = train_books[indices + 1, 0::2] / 10000.0

    # Alignment check using the actual next feature, before generation.
    actual_next = training[indices + 1, None, :]
    restored_prices, restored_volumes = reconstruct_books(
        actual_next, mean, scale, initial_prices
    )

    expected_prices = train_books[indices + 2, 0::2] / 10000.0
    expected_volumes = train_books[indices + 2, 1::2]

    np.testing.assert_allclose(
        restored_prices[:, 0], expected_prices, rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        restored_volumes[:, 0], expected_volumes,
        rtol=1e-10, atol=1e-6,
    )

    torch.set_num_threads(2)

    seeds = [42, 43, 44, 45, 46]
    collected = {"VAR(1)": [], "TimeGAN": []}

    labels = [
        "Crossed books",
        "Locked books",
        "Ask ordering violations",
        "Bid ordering violations",
        "Negative-volume books",
    ]

    print("Next-transition price and volume alignment checks passed.")
    print("Fixed training-based anchors; five generation seeds.")

    for seed in seeds:
        var_generated = generate_var_transitions(
            intercept, transition, residuals, initial_states,
            length=100, seed=seed,
        )
        timegan_generated = generate_timegan(
            args.checkpoint, mean, scale,
            count=32, length=100, seed=seed,
        )

        for model_name, generated in [
            ("VAR(1)", var_generated),
            ("TimeGAN", timegan_generated),
        ]:
            prices, volumes = reconstruct_books(
                generated, mean, scale, initial_prices
            )

            asks = prices[:, :, 0::2]
            bids = prices[:, :, 1::2]
            spread = asks[:, :, 0] - bids[:, :, 0]

            failures = [
                spread < 0,
                spread == 0,
                np.any(np.diff(asks, axis=2) <= 0, axis=2),
                np.any(np.diff(bids, axis=2) >= 0, axis=2),
                np.any(volumes < 0, axis=2),
            ]

            rates = np.array([
                100 * mask.mean() for mask in failures
            ])
            collected[model_name].append(rates)

            print(
                f"Seed {seed} | {model_name}: "
                + ", ".join(
                    f"{label}={rate:.2f}%"
                    for label, rate in zip(labels, rates)
                )
            )

    print("\nSummary: mean ± sample SD across generation seeds")
    for model_name, rows in collected.items():
        rates = np.stack(rows)
        means = rates.mean(axis=0)
        stds = rates.std(axis=0, ddof=1)

        print(f"\n{model_name}: 16,000 generated snapshots")
        for label, mean_rate, std_rate in zip(labels, means, stds):
            print(f"{label}: {mean_rate:.2f}% ± {std_rate:.2f} pp")


if __name__ == "__main__":
    main()