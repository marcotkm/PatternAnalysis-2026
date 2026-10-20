"""Plot fixed reference and synthetic periods for failure analysis."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline import fit_var1
from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    fit_standardiser,
)
from evaluate import (
    generate_var_transitions,
    generate_timegan,
    reconstruct_books,
    market_observations,
)


def signed_log_volume(volumes):
    """Compress the volume scale while keeping negative values visible."""
    return np.sign(volumes) * np.log1p(np.abs(volumes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    messages, books = load_lobster(args.data_dir)
    splits = chronological_split(messages, books)
    train_books = splits["train"][1]
    validation_books = splits["validation"][1]

    _, features = build_features(train_books)
    mean, scale = fit_standardiser(features)
    training = (features - mean) / scale
    intercept, transition, residuals, _ = fit_var1(training)

    # Match the evaluator's anchors and generation seed.
    rng = np.random.default_rng(123)
    indices = rng.integers(0, len(training) - 1, size=32)
    anchors = train_books[indices + 1, 0::2] / 10000.0

    var_features = generate_var_transitions(
        intercept, transition, residuals, training[indices],
        length=100, seed=42,
    )
    timegan_features = generate_timegan(
        args.checkpoint, mean, scale,
        count=32, length=100, seed=42,
    )

    var_prices, var_volumes = reconstruct_books(
        var_features, mean, scale, anchors
    )
    gan_prices, gan_volumes = reconstruct_books(
        timegan_features, mean, scale, anchors
    )

    # Use the beginning, middle and end of the validation window collection.
    # Choose synthetic indices in advance, without selecting by appearance.
    n_real_windows = (len(validation_books) - 1) // 100
    if n_real_windows < 3:
        raise ValueError("Need at least three real reference windows.")

    real_indices = [0, n_real_windows // 2, n_real_windows - 1]
    synthetic_indices = [0, 16, 31]

    for case, (real_index, synthetic_index) in enumerate(
        zip(real_indices, synthetic_indices), start=1
    ):
        start = real_index * 100
        real_books = validation_books[start + 1:start + 101]
        real_prices = real_books[:, 0::2] / 10000.0
        real_volumes = real_books[:, 1::2].astype(np.float64)
        real_anchor = validation_books[start, 0::2] / 10000.0

        examples = [
            (
                f"Real validation — window {real_index}",
                real_prices, real_volumes, real_anchor,
            ),
            (
                f"VAR(1) — sample {synthetic_index}",
                var_prices[synthetic_index],
                var_volumes[synthetic_index],
                anchors[synthetic_index],
            ),
            (
                f"TimeGAN — sample {synthetic_index}",
                gan_prices[synthetic_index],
                gan_volumes[synthetic_index],
                anchors[synthetic_index],
            ),
        ]

        # Shared depth colour scale across all panels in this figure.
        transformed_depths = [
            signed_log_volume(example[2]) for example in examples
        ]
        colour_limit = max(
            float(np.abs(depth).max()) for depth in transformed_depths
        )
        colour_limit = max(colour_limit, 1e-6)

        fig, axes = plt.subplots(
            5, 3, figsize=(15, 13),
            sharex=True, sharey="row", constrained_layout=True,
        )
        x = np.arange(1, 101)

        for column, (title, prices, volumes, anchor) in enumerate(examples):
            observations = market_observations(
                prices[None, :, :], anchor[None, :]
            )
            returns = observations["mid_return"]
            spreads = observations["spread"]

            for row, side in [(0, slice(0, 20, 2)),
                              (1, slice(1, 20, 2))]:
                image = axes[row, column].imshow(
                    transformed_depths[column][:, side].T,
                    origin="lower",
                    aspect="auto",
                    extent=(0.5, 100.5, 0.5, 10.5),
                    cmap="coolwarm",
                    vmin=-colour_limit,
                    vmax=colour_limit,
                    interpolation="nearest",
                )
                axes[row, column].set_yticks([1, 5, 10])

            axes[0, column].set_title(title)

            axes[2, column].plot(x, returns * 10000, linewidth=0.9)
            axes[2, column].axhline(0, color="black", linewidth=0.5)

            axes[3, column].plot(x, spreads, linewidth=0.9)
            axes[3, column].axhline(
                0, color="black", linestyle="--", linewidth=0.7
            )
            negative = spreads < 0
            axes[3, column].scatter(
                x[negative], spreads[negative],
                color="red", s=12, zorder=3,
            )
            ask_gaps = np.diff(prices[:, 0::2], axis=1)
            bid_gaps = -np.diff(prices[:, 1::2], axis=1)

            axes[4, column].plot(
                x, ask_gaps.min(axis=1), label="Ask", linewidth=0.9
            )
            axes[4, column].plot(
                x, bid_gaps.min(axis=1), label="Bid", linewidth=0.9
            )
            axes[4, column].axhline(
                0, color="black", linestyle="--", linewidth=0.7
            )
            axes[4, column].legend(fontsize=8)
            axes[4, column].set_xlabel("Event transition")

            for row in (2, 3, 4):
                axes[row, column].grid(alpha=0.2)

        axes[0, 0].set_ylabel("Ask level")
        axes[1, 0].set_ylabel("Bid level")
        axes[2, 0].set_ylabel("Mid-price log return (bp)")
        axes[3, 0].set_ylabel("Spread ($)")
        axes[4, 0].set_ylabel("Minimum adjacent price gap ($)")

        fig.colorbar(
            image,
            ax=axes[:2, :].ravel().tolist(),
            label="sign(volume) × log(1 + |volume|)",
            shrink=0.8,
        )
        fig.suptitle(
            f"Case {case}: reference versus independent synthetic sequences\n"
            "Seed 42; shared scales; raw outputs without repairs",
            fontsize=13,
        )

        output = args.output_dir / f"case_{case:02d}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)

        print(
            f"Saved {output} | real window={real_index} | "
            f"synthetic sample={synthetic_index}"
        )


if __name__ == "__main__":
    main()