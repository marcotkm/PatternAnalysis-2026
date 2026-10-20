"""Load and validate paired LOBSTER message and order book files."""

import argparse
from pathlib import Path

import numpy as np


PREFIX = "AMZN_2012-06-21_34200000_57600000"


def load_lobster(data_dir: Path):
    """Return message records and raw integer order book values."""
    message_path = data_dir / f"{PREFIX}_message_10.csv"
    book_path = data_dir / f"{PREFIX}_orderbook_10.csv"

    messages = np.loadtxt(message_path, delimiter=",", dtype=np.float64,
                          ndmin=2)
    books = np.loadtxt(book_path, delimiter=",", dtype=np.int64,
                      ndmin=2)

    if messages.shape[1] != 6:
        raise ValueError("Expected 6 message columns.")
    if books.shape[1] != 40:
        raise ValueError("Expected 40 order book columns.")
    if len(messages) != len(books):
        raise ValueError("Message and order book row counts differ.")
    if len(messages) == 0:
        raise ValueError("Dataset is empty.")
    if not np.isfinite(messages).all():
        raise ValueError("Messages contain non-finite values.")
    if np.any(np.diff(messages[:, 0]) < 0):
        raise ValueError("Timestamps are not chronologically ordered.")

    return messages, books

def chronological_split(messages, books):
    """Split by session time, keeping identical timestamps together."""
    session_start = 34200.0  # 09:30
    session_end = 57600.0    # 16:00
    duration = session_end - session_start

    train_end = session_start + 0.70 * duration
    validation_end = session_start + 0.85 * duration
    timestamps = messages[:, 0]

    if np.any((timestamps < session_start) | (timestamps > session_end)):
        raise ValueError("Found timestamps outside the expected session.")

    # 'left' sends every event at a boundary into the later split.
    train_stop = np.searchsorted(timestamps, train_end, side="left")
    validation_stop = np.searchsorted(
        timestamps, validation_end, side="left"
    )

    boundaries = {
        "train": (0, train_stop),
        "validation": (train_stop, validation_stop),
        "test": (validation_stop, len(messages)),
    }

    splits = {}
    for name, (start, stop) in boundaries.items():
        if start == stop:
            raise ValueError(f"{name} split is empty.")
        splits[name] = (messages[start:stop], books[start:stop])

    return splits

def price_log_returns(books):
    """Return the starting prices and subsequent per-level log returns."""
    # Columns 0, 2, 4, ... are prices:
    # ask1, bid1, ask2, bid2, ..., ask10, bid10.
    raw_prices = books[:, 0::2]

    if len(raw_prices) < 2:
        raise ValueError("Need at least two books to calculate returns.")

    if np.any(np.abs(raw_prices) == 9999999999):
        raise ValueError("Missing-level sentinel found in price data.")

    prices = raw_prices.astype(np.float64) / 10000.0

    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Prices must be finite and strictly positive.")

    returns = np.diff(np.log(prices), axis=0)
    return prices[0].copy(), returns


def reconstruct_prices(initial_prices, returns):
    """Recover dollar prices, including the initial observation."""
    cumulative_returns = np.vstack([
        np.zeros((1, returns.shape[1])),
        np.cumsum(returns, axis=0),
    ])
    return initial_prices * np.exp(cumulative_returns)

def build_features(books):
    """Combine price returns with aligned log-transformed volumes."""
    initial_prices, returns = price_log_returns(books)

    # Volume columns: ask1, bid1, ..., ask10, bid10.
    volumes = books[1:, 1::2].astype(np.float64)

    if not np.isfinite(volumes).all() or np.any(volumes < 0):
        raise ValueError("Volumes must be finite and non-negative.")

    # log1p supports zero volume and reduces the scale of large values.
    features = np.concatenate(
        [returns, np.log1p(volumes)], axis=1
    )
    return initial_prices, features


def fit_standardiser(training_features):
    """Estimate feature scaling from the training split only."""
    mean = training_features.mean(axis=0)
    std = training_features.std(axis=0)

    # Avoid division by zero for constant features.
    scale = np.where(std > 1e-12, std, 1.0)
    return mean, scale

def make_windows(normalised_features, books, length=100):
    """Create non-overlapping windows and their starting dollar prices."""
    if length < 1:
        raise ValueError("Window length must be positive.")

    n_windows = len(normalised_features) // length
    if n_windows == 0:
        raise ValueError("Not enough observations for one window.")

    used_rows = n_windows * length

    windows = normalised_features[:used_rows].reshape(
        n_windows, length, normalised_features.shape[1]
    )

    # Feature row s describes the transition from book s to book s+1.
    starts = np.arange(n_windows) * length
    initial_prices = books[starts, 0::2].astype(np.float64) / 10000.0

    return windows, initial_prices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    messages, books = load_lobster(args.data_dir)

    # Each level contains: ask price, ask volume, bid price, bid volume.
    first_ask = books[0, 0] / 10000
    first_bid = books[0, 2] / 10000

    print(f"Messages shape: {messages.shape}")
    print(f"Order books shape: {books.shape}")
    print(f"First timestamp: {messages[0, 0]:.9f}")
    print(f"Last timestamp:  {messages[-1, 0]:.9f}")
    print(f"First best ask: ${first_ask:.4f}")
    print(f"First best bid: ${first_bid:.4f}")
    print(f"First spread:   ${first_ask - first_bid:.4f}")

    splits = chronological_split(messages, books)

    for name, (split_messages, split_books) in splits.items():
        print(
            f"{name}: {len(split_books):,} rows | "
            f"{split_messages[0, 0]:.9f} "
            f"to {split_messages[-1, 0]:.9f} seconds"
        )

        initial_prices, returns = price_log_returns(split_books)
        reconstructed = reconstruct_prices(initial_prices, returns)
        original_prices = split_books[:, 0::2] / 10000.0

        max_error = np.max(np.abs(reconstructed - original_prices))

        if not np.allclose(
            reconstructed, original_prices, rtol=0, atol=1e-8
        ):
            raise ValueError(f"{name}: price reconstruction failed.")

        print(
            f"  Returns shape: {returns.shape} | "
            f"Maximum reconstruction error: ${max_error:.12f}"
        )

        features_by_split = {}

    for name, (_, split_books) in splits.items():
        _, features = build_features(split_books)
        features_by_split[name] = features

    mean, scale = fit_standardiser(features_by_split["train"])

    for name, features in features_by_split.items():
        normalised = (features - mean) / scale
        restored = normalised * scale + mean

        if not np.isfinite(normalised).all():
            raise ValueError(f"{name}: non-finite normalised features.")

        if not np.allclose(restored, features, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name}: feature scaling round trip failed.")

        # Check that log-transformed volumes recover the original volumes.
        restored_volumes = np.expm1(restored[:, 20:])
        original_volumes = splits[name][1][1:, 1::2]

        if not np.allclose(
            restored_volumes, original_volumes,
            rtol=1e-10, atol=1e-6
        ):
            raise ValueError(f"{name}: volume reconstruction failed.")

        print(
            f"{name}: features={normalised.shape} | "
            f"finite=True | scaling and volume checks passed"
        )

        split_books = splits[name][1]
        windows, window_initial_prices = make_windows(
            normalised, split_books, length=100
        )

        # Verify the starting-price alignment for every window.
        recovered_features = windows * scale + mean
        recovered_returns = recovered_features[:, :, :20]

        recovered_prices = (
            window_initial_prices[:, None, :]
            * np.exp(np.cumsum(recovered_returns, axis=1))
        )

        used_rows = len(windows) * windows.shape[1]
        expected_prices = (
            split_books[1:used_rows + 1, 0::2]
            .astype(np.float64)
            .reshape(len(windows), windows.shape[1], 20)
            / 10000.0
        )

        if not np.allclose(
            recovered_prices, expected_prices, rtol=0, atol=1e-8
        ):
            raise ValueError(f"{name}: window price alignment failed.")

        print(
            f"  Windows: {windows.shape} | "
            f"Unused trailing transitions: {len(normalised) - used_rows} | "
            f"price alignment passed"
        )

if __name__ == "__main__":
    main()