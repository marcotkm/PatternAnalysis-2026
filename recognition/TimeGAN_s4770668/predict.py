"""Generate TimeGAN sequences and audit raw reconstructed order books."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    make_windows,
)
from modules import Generator, Supervisor, Recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/prediction"),
    )
    args = parser.parse_args()

    torch.set_num_threads(2)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint["stage"] != "joint":
        raise ValueError("Expected a joint-training checkpoint.")

    config = checkpoint["config"]
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

    # Dedicated seed gives reproducible fresh generation noise.
    noise_rng = torch.Generator().manual_seed(42)
    noise = torch.randn(
        32, config["window_length"], config["noise_dim"],
        generator=noise_rng,
    )

    with torch.no_grad():
        generated = recovery(supervisor(generator(noise))).numpy()

    mean = checkpoint["mean"].numpy()
    scale = checkpoint["scale"].numpy()
    restored = generated.astype(np.float64) * scale + mean

    if not np.isfinite(restored).all():
        raise ValueError("Non-finite generated features.")

    # Use training-only starting prices to reconstruct price levels.
    messages, books = load_lobster(args.data_dir)
    train_books = chronological_split(messages, books)["train"][1]
    _, train_features = build_features(train_books)

    rng = np.random.default_rng(123)
    indices = rng.integers(0, len(train_features), size=len(generated))
    initial_prices = train_books[indices, 0::2] / 10000.0

    with np.errstate(over="raise", invalid="raise"):
        prices = initial_prices[:, None, :] * np.exp(
            np.cumsum(restored[:, :, :20], axis=1)
        )
        volumes = np.expm1(restored[:, :, 20:])

    if not np.isfinite(prices).all() or not np.isfinite(volumes).all():
        raise ValueError("Non-finite reconstructed books.")

    asks = prices[:, :, 0::2]
    bids = prices[:, :, 1::2]
    spreads = asks[:, :, 0] - bids[:, :, 0]
    ask_gaps = np.diff(asks, axis=2)
    bid_gaps = -np.diff(bids, axis=2)

    failures = {
        "Crossed books": spreads < 0,
        "Locked books": spreads == 0,
        "Ask ordering violations": np.any(ask_gaps <= 0, axis=2),
        "Bid ordering violations": np.any(bid_gaps <= 0, axis=2),
        "Negative-volume books": np.any(volumes < 0, axis=2),
        "Ask inversions greater than $0.01": np.any(
            ask_gaps < -0.01, axis=2
        ),
        "Bid inversions greater than $0.01": np.any(
            bid_gaps < -0.01, axis=2
        ),
    }

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Generated shape: {generated.shape}")
    print(f"Audited generated snapshots: {spreads.size}")

    for label, mask in failures.items():
        print(f"{label}: {100 * mask.mean():.2f}%")

    print(f"Minimum spread: ${spreads.min():.8f}")
    print(f"Minimum ask-level gap: ${ask_gaps.min():.8f}")
    print(f"Minimum bid-level gap: ${bid_gaps.min():.8f}")
    print(f"Minimum volume: {volumes.min():.8f}")

        # Compare generated features with real training windows using
    # the same standardisation and sequence length.
    real_windows, _ = make_windows(
        (train_features - mean) / scale,
        train_books,
        length=config["window_length"],
    )

    reference_rng = np.random.default_rng(456)
    reference_indices = reference_rng.choice(
        len(real_windows), size=len(generated), replace=False
    )
    real_reference = real_windows[reference_indices]

    print("\nCross-sequence variability")
    for label, columns in [
        ("Price returns", slice(0, 20)),
        ("Log volumes", slice(20, 40)),
    ]:
        real_std = real_reference[:, :, columns].std(axis=0).mean()
        fake_std = generated[:, :, columns].std(axis=0).mean()

        ratio = fake_std / real_std if real_std > 0 else float("nan")
        print(
            f"{label}: real={real_std:.6f} | "
            f"synthetic={fake_std:.6f} | ratio={ratio:.3f}"
        )

    # Visualise the first sequence from the samples audited above.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, prices.shape[1] + 1)

    ask_depth = volumes[0, :, 0::2].T
    bid_depth = volumes[0, :, 1::2].T

    # Signed log preserves visibility of invalid negative volumes.
    ask_display = np.sign(ask_depth) * np.log1p(np.abs(ask_depth))
    bid_display = np.sign(bid_depth) * np.log1p(np.abs(bid_depth))
    limit = max(
        float(np.abs(ask_display).max()),
        float(np.abs(bid_display).max()),
        1e-8,
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, depth, title in [
        (axes[0], ask_display, "Generated ask depth"),
        (axes[1], bid_display, "Generated bid depth"),
    ]:
        plot = ax.imshow(
            depth,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(0.5, len(steps) + 0.5, 0.5, 10.5),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        ax.set(title=title, ylabel="Level")
        fig.colorbar(
            plot, ax=ax, label="sign(volume) × log(1 + |volume|)"
        )

    axes[2].plot(steps, spreads[0])
    axes[2].axhline(0, color="black", linestyle="--")
    axes[2].set(title="Best-level spread", ylabel="Dollars")

    axes[3].plot(steps, ask_gaps[0].min(axis=1), label="Ask")
    axes[3].plot(steps, bid_gaps[0].min(axis=1), label="Bid")
    axes[3].axhline(0, color="black", linestyle="--")
    axes[3].set(
        title="Minimum adjacent price gaps",
        xlabel="Generated event transition",
        ylabel="Dollars",
    )
    axes[3].legend()

    fig.suptitle(
        f"TimeGAN epoch {checkpoint['epoch']} — sample 0, seed 42\n"
        "Raw generated books without repairs"
    )
    fig.tight_layout()
    destination = args.output_dir / "generated_sample.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction figure: {destination}")


if __name__ == "__main__":
    main()