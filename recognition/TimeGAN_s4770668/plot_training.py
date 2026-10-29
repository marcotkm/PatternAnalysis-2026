"""Plot saved training histories without rerunning training."""

import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "training_curves"


def load_history(run):
    saved_csv = OUTPUT / f"{run}_history.csv"
    checkpoint_csv = ROOT / "checkpoints" / run / "history.csv"

    # Prefer the committed history so figures work without checkpoints.
    if saved_csv.exists():
        source = saved_csv
    elif checkpoint_csv.exists():
        shutil.copy2(checkpoint_csv, saved_csv)
        source = saved_csv
    else:
        raise FileNotFoundError(
            f"No training history found for {run}. "
            f"Expected {saved_csv} or {checkpoint_csv}."
        )

    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Empty history: {source}")

    return {
        key: [float(row[key]) for row in rows]
        for key in rows[0]
    }


def finish(fig, filename):
    fig.tight_layout()
    destination = OUTPUT / filename
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {destination}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    stages = [
        ("reconstruction_cpu_02", "Reconstruction pretraining"),
        ("supervisor_cpu_01", "Supervisor pretraining"),
    ]

    for ax, (run, title) in zip(axes, stages):
        history = load_history(run)
        ax.plot(history["epoch"], history["train_mse"], label="Training")
        ax.plot(history["epoch"], history["validation_mse"], label="Validation")
        if "persistence_mse" in history:
            ax.plot(
                history["epoch"], history["persistence_mse"],
                linestyle="--", label="Validation persistence",
            )
        ax.set(title=title, xlabel="Epoch", ylabel="MSE")
        ax.grid(alpha=0.25)
        ax.legend()

    finish(fig, "pretraining.png")

    runs = [
        ("joint_gpu_ratio_708879", "Constraint weight 0"),
        ("joint_gpu_constraint_709048", "Constraint weight 1"),
        ("joint_gpu_constraint_w5_709072", "Constraint weight 5"),
    ]
    metrics = [
        ("d_loss", "Discriminator loss"),
        ("g_adversarial", "Generator adversarial loss"),
        ("g_moment", "Unweighted moment loss"),
        ("g_temporal_mse", "Supervisor temporal MSE"),
        ("validation_mse", "Validation reconstruction MSE"),
        ("constraint_total", "Unweighted structural penalty"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for (run, label), colour in zip(runs, ["tab:blue", "tab:orange", "tab:green"]):
        history = load_history(run)
        for ax, (key, title) in zip(axes.flat, metrics):
            if key in history:
                ax.plot(history["epoch"], history[key], label=label, color=colour)
            ax.set(title=title, xlabel="Epoch", ylabel="Loss / MSE")
            ax.grid(alpha=0.25)

    for ax in axes.flat:
        ax.legend(fontsize=8)

    fig.suptitle(
        "Joint training: matched discriminator schedule, different constraint weights",
        fontsize=13,
    )
    finish(fig, "joint_constraint_comparison.png")


if __name__ == "__main__":
    main()