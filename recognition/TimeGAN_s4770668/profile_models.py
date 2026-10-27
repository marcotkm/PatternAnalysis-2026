"""CPU generation benchmark for the selected TimeGAN and VAR baseline."""

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from baseline import fit_var1
from dataset import (
    load_lobster, chronological_split,
    build_features, fit_standardiser,
)
from evaluate import generate_var_transitions
from modules import (
    Embedder, Recovery, Generator, Supervisor, Discriminator,
)


def benchmark(function, repeats=30):
    # Warm-up runs are excluded from the measurements.
    for seed in range(3):
        function(seed)

    seconds = []
    for seed in range(100, 100 + repeats):
        started = time.perf_counter()
        function(seed)
        seconds.append(time.perf_counter() - started)

    median = float(np.median(seconds))
    return {
        "repeats": repeats,
        "batch_size": 32,
        "transitions_per_sequence": 100,
        "median_batch_seconds": median,
        "p95_batch_seconds": float(np.percentile(seconds, 95)),
        "sequences_per_second": 32 / median,
        "transitions_per_second": 3200 / median,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    config = checkpoint["config"]
    hidden = config["hidden_dim"]

    models = {
        "embedder": Embedder(
            config["input_dim"], hidden, config["num_layers"]
        ),
        "recovery": Recovery(
            config["input_dim"], hidden, config["num_layers"]
        ),
        "generator": Generator(
            config["noise_dim"], hidden,
            config["generator_num_layers"],
        ),
        "supervisor": Supervisor(
            hidden, config["supervisor_num_layers"]
        ),
        "discriminator": Discriminator(
            hidden, config["discriminator_num_layers"]
        ),
    }

    for name, model in models.items():
        model.load_state_dict(checkpoint[name])
        model.eval()

    counts = {
        name: sum(p.numel() for p in model.parameters())
        for name, model in models.items()
    }

    messages, books = load_lobster(args.data_dir)
    train_books = chronological_split(messages, books)["train"][1]
    _, features = build_features(train_books)
    mean, scale = fit_standardiser(features)
    training = (features - mean) / scale

    intercept, transition, residuals, _ = fit_var1(training)
    residuals = residuals - residuals.mean(axis=0)

    rng = np.random.default_rng(123)
    indices = rng.integers(0, len(training) - 1, size=32)
    initial_states = training[indices]

    def run_var(seed):
        return generate_var_transitions(
            intercept, transition, residuals, initial_states,
            length=100,
            seed=seed,
            residuals_centered=True,
        )

    @torch.inference_mode()
    def run_timegan(seed):
        noise_rng = torch.Generator().manual_seed(seed)
        noise = torch.randn(
            32, 100, config["noise_dim"], generator=noise_rng
        )
        return models["recovery"](
            models["supervisor"](models["generator"](noise))
        )

    var_timing = benchmark(run_var)
    timegan_timing = benchmark(run_timegan)

    # macOS reports bytes; Linux reports KiB.
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = peak_rss if platform.system() == "Darwin" else peak_rss * 1024

    result = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "device": "cpu",
        "torch_cpu_threads": torch.get_num_threads(),
        "checkpoint": str(args.checkpoint),
        "timegan_parameters_by_component": counts,
        "timegan_total_parameters": sum(counts.values()),
        "timegan_generation_parameters": sum(
            counts[name] for name in ("generator", "supervisor", "recovery")
        ),
        "var_regression_coefficients": int(
            intercept.size + transition.size
        ),
        "var_residual_bank_shape": list(residuals.shape),
        "var_residual_bank_bytes": int(residuals.nbytes),
        "VAR_generation": var_timing,
        "TimeGAN_generation": timegan_timing,
        "combined_process_peak_rss_MiB": rss_bytes / 1024**2,
        "measurement_scope": (
            "Batch generation of standardised features, including random "
            "sampling. Excludes loading, fitting, price reconstruction and "
            "evaluation. Peak RSS covers the whole process and both models; "
            "it is not per-model memory or GPU VRAM."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()