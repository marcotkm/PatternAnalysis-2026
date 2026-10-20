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


if __name__ == "__main__":
    main()