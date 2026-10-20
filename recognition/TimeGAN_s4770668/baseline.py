"""Fit a VAR(1) baseline to training data and inspect its dynamics."""

import argparse
from pathlib import Path

import numpy as np

from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    fit_standardiser,
)


def fit_var1(features):
    """Fit next-step predictions using training observations only."""
    current = features[:-1]
    following = features[1:]

    # The leading column of ones lets us learn an intercept.
    design = np.column_stack([np.ones(len(current)), current])

    coefficients, _, rank, _ = np.linalg.lstsq(
        design, following, rcond=None
    )

    intercept = coefficients[0]
    transition = coefficients[1:]
    residuals = following - design @ coefficients

    return intercept, transition, residuals, rank

def generate_var1(
    intercept, transition, residuals, initial_states,
    length=100, seed=42
):
    """Generate sequences, including each supplied initial feature state."""
    rng = np.random.default_rng(seed)

    n_sequences, n_features = initial_states.shape
    generated = np.empty(
        (n_sequences, length, n_features), dtype=np.float64
    )
    generated[:, 0] = initial_states

    # Centre residuals before sampling.
    centred_residuals = residuals - residuals.mean(axis=0)

    for step in range(1, length):
        indices = rng.integers(
            0, len(centred_residuals), size=n_sequences
        )
        noise = centred_residuals[indices]

        generated[:, step] = (
            intercept
            + generated[:, step - 1] @ transition
            + noise
        )

    if not np.isfinite(generated).all():
        raise ValueError("Generated features contain non-finite values.")

    return generated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    messages, books = load_lobster(args.data_dir)
    splits = chronological_split(messages, books)

    _, train_features = build_features(splits["train"][1])
    _, validation_features = build_features(splits["validation"][1])

    mean, scale = fit_standardiser(train_features)
    train = (train_features - mean) / scale
    validation = (validation_features - mean) / scale

    intercept, transition, residuals, rank = fit_var1(train)

    if not all(np.isfinite(array).all() for array in
               (intercept, transition, residuals)):
        raise ValueError("VAR fit produced non-finite values.")

    # Largest eigenvalue magnitude: below 1 indicates stable linear dynamics.
    spectral_radius = np.max(np.abs(np.linalg.eigvals(transition)))

    prediction = intercept + validation[:-1] @ transition
    actual = validation[1:]

    var_mse = np.mean((actual - prediction) ** 2)
    persistence_mse = np.mean((actual - validation[:-1]) ** 2)

    print(f"Training transitions: {len(residuals):,}")
    print(f"Design matrix rank: {rank}/41")
    print(f"Residual shape: {residuals.shape}")
    print(f"Spectral radius: {spectral_radius:.8f}")
    print(f"Validation VAR MSE:         {var_mse:.6f}")
    print(f"Validation persistence MSE: {persistence_mse:.6f}")

    if spectral_radius >= 1:
        print("WARNING: fitted dynamics are not strictly stable.")
    if rank < 41:
        print("WARNING: design matrix is rank deficient.")

    rng = np.random.default_rng(123)
    start_indices = rng.integers(0, len(train), size=32)

    generated = generate_var1(
        intercept,
        transition,
        residuals,
        initial_states=train[start_indices],
        length=100,
        seed=42,
    )

    # Undo standardisation.
    restored = generated * scale + mean

    # Feature row s contains the return from book s to book s+1.
    # Start reconstruction at book s.
    initial_prices = (
        splits["train"][1][start_indices, 0::2]
        .astype(np.float64) / 10000.0
    )

    with np.errstate(over="raise", invalid="raise"):
        prices = (
            initial_prices[:, None, :]
            * np.exp(np.cumsum(restored[:, :, :20], axis=1))
        )
        volumes = np.expm1(restored[:, :, 20:])

    if not np.isfinite(prices).all() or not np.isfinite(volumes).all():
        raise ValueError("Reconstruction produced non-finite values.")

    asks = prices[:, :, 0::2]
    bids = prices[:, :, 1::2]
    spreads = asks[:, :, 0] - bids[:, :, 0]

    crossed = spreads < 0
    locked = spreads == 0
    ask_errors = np.any(np.diff(asks, axis=2) <= 0, axis=2)
    bid_errors = np.any(np.diff(bids, axis=2) >= 0, axis=2)
    negative_depth = np.any(volumes < 0, axis=2)

    # Step zero is a copied training observation; audit generated steps only.
    print(f"\nGenerated shape: {generated.shape}")
    for label, failures in [
        ("Crossed books", crossed),
        ("Locked books", locked),
        ("Ask ordering violations", ask_errors),
        ("Bid ordering violations", bid_errors),
        ("Negative-volume books", negative_depth),
    ]:
        print(f"{label}: {100 * failures[:, 1:].mean():.2f}%")

    # Exclude the copied initial observation.
    ask_gaps = np.diff(asks[:, 1:, :], axis=2)
    bid_gaps = -np.diff(bids[:, 1:, :], axis=2)

    print("\nViolation severity — raw continuous outputs")
    print(f"Minimum spread: ${spreads[:, 1:].min():.8f}")
    print(f"Minimum ask-level gap: ${ask_gaps.min():.8f}")
    print(f"Minimum bid-level gap: ${bid_gaps.min():.8f}")
    print(f"Minimum volume: {volumes[:, 1:, :].min():.8f}")

    # Illustrative diagnostic threshold, not a replacement for strict checks.
    tolerance = 0.01
    print(
        "Ask books with an inversion greater than $0.01: "
        f"{100 * np.any(ask_gaps < -tolerance, axis=2).mean():.2f}%"
    )
    print(
        "Bid books with an inversion greater than $0.01: "
        f"{100 * np.any(bid_gaps < -tolerance, axis=2).mean():.2f}%"
    )


if __name__ == "__main__":
    main()