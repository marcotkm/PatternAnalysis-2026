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

def market_observations(prices, initial_prices):
    """Extract spreads and mid-price log returns within each sequence."""
    best_ask = prices[:, :, 0]
    best_bid = prices[:, :, 1]
    spreads = best_ask - best_bid
    mid = (best_ask + best_bid) / 2.0

    initial_mid = (
        initial_prices[:, 0] + initial_prices[:, 1]
    ) / 2.0

    if (
        not np.isfinite(mid).all()
        or not np.isfinite(initial_mid).all()
        or np.any(mid <= 0)
        or np.any(initial_mid <= 0)
    ):
        raise ValueError("Cannot calculate returns from invalid mid-prices.")

    # Include the return from each anchor to its first generated book.
    with_initial = np.concatenate(
        [initial_mid[:, None], mid], axis=1
    )
    returns = np.diff(np.log(with_initial), axis=1)

    return {
        "spread": spreads.ravel(),
        "mid_return": returns.ravel(),
    }


def real_window_observations(books, length=100):
    """Extract real observations using the same window/anchor convention."""
    n_windows = (len(books) - 1) // length
    if n_windows == 0:
        raise ValueError("Not enough records for a real reference window.")

    used = n_windows * length
    prices = (
        books[1:used + 1, 0::2].astype(np.float64)
        .reshape(n_windows, length, 20) / 10000.0
    )
    starts = np.arange(n_windows) * length
    initial_prices = books[starts, 0::2] / 10000.0

    return market_observations(prices, initial_prices)


def training_bins(values, interior_bins=50):
    """Training-range bins plus explicit lower and upper tail bins."""
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values used to define histogram bins.")

    lower = float(values.min())
    upper = float(values.max())

    if lower == upper:
        padding = max(abs(lower) * 0.01, 1e-8)
        lower -= padding
        upper += padding

    # Include the training maximum in the interior range.
    upper = np.nextafter(upper, np.inf)
    edges = np.linspace(lower, upper, interior_bins + 1)
    return np.concatenate(([-np.inf], edges, [np.inf]))


def histogram_kl(real, synthetic, edges, smoothing=1e-6):
    """KL(real || synthetic), in nats, with probability smoothing."""
    if (
        len(real) == 0 or len(synthetic) == 0
        or not np.isfinite(real).all()
        or not np.isfinite(synthetic).all()
    ):
        raise ValueError("KL inputs must be nonempty and finite.")

    real_counts = np.histogram(real, bins=edges)[0]
    fake_counts = np.histogram(synthetic, bins=edges)[0]

    if real_counts.sum() != len(real) or fake_counts.sum() != len(synthetic):
        raise ValueError("Histogram discarded observations.")

    p = real_counts / real_counts.sum()
    q = fake_counts / fake_counts.sum()

    # Add the same probability floor, then renormalise.
    p = (p + smoothing) / (1 + smoothing * len(p))
    q = (q + smoothing) / (1 + smoothing * len(q))

    return float(np.sum(p * np.log(p / q)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("Use at least two interior histogram bins.")

    messages, books = load_lobster(args.data_dir)
    splits = chronological_split(messages, books)
    train_books = splits["train"][1]
    validation_books = splits["validation"][1]

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

    training_reference = real_window_observations(train_books)
    validation_reference = real_window_observations(validation_books)

    histogram_edges = {
        name: training_bins(values, interior_bins=args.bins)
        for name, values in training_reference.items()
    }
    kl_collected = {"VAR(1)": [], "TimeGAN": []}

    print(
        f"KL reference: validation; bins: {args.bins} training-range bins "
        "+ 2 tail bins; probability smoothing: 1e-6."
    )
    print(
        "Validation reference transitions:",
        len(validation_reference["spread"]),
    )
    
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

            synthetic_observations = market_observations(
                prices, initial_prices
            )

            if seed == seeds[0]:
                real_returns = validation_reference["mid_return"]
                fake_returns = synthetic_observations["mid_return"]

                print(f"\nReturn diagnostics — {model_name}, seed {seed}")
                for label, values in [
                    ("Real validation", real_returns),
                    ("Synthetic", fake_returns),
                ]:
                    quantiles = np.quantile(
                        values, [0, 0.01, 0.5, 0.99, 1]
                    )
                    print(
                        f"{label}: exact-zero={100 * np.mean(values == 0):.2f}%"
                        f" | std={values.std():.8e}"
                    )
                    print(
                        "  min / p1 / median / p99 / max:",
                        np.array2string(quantiles, precision=8),
                    )

                edges = histogram_edges["mid_return"]
                zero_bin = np.searchsorted(edges, 0.0, side="right") - 1
                print(
                    "Bin containing zero:",
                    edges[zero_bin], "to", edges[zero_bin + 1],
                )

            kl_values = [
                histogram_kl(
                    validation_reference[name],
                    synthetic_observations[name],
                    histogram_edges[name],
                )
                for name in ("spread", "mid_return")
            ]
            kl_collected[model_name].append(kl_values)

            print(
                f"Seed {seed} | {model_name} | "
                f"Spread KL={kl_values[0]:.6f} | "
                f"Mid-return KL={kl_values[1]:.6f}"
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

    print("\nKL(real validation || synthetic): mean ± seed SD, in nats")
    for model_name, rows in kl_collected.items():
        values = np.asarray(rows)
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=1)

        print(f"\n{model_name}")
        print(f"Spread KL: {means[0]:.6f} ± {stds[0]:.6f}")
        print(f"Mid-return KL: {means[1]:.6f} ± {stds[1]:.6f}")


if __name__ == "__main__":
    main()