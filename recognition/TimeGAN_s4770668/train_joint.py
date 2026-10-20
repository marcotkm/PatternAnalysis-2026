"""Joint TimeGAN training with logging and fixed-noise monitoring."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    make_windows,
)
from modules import (
    Embedder, Recovery, Generator, Supervisor, Discriminator,
)


def activate(models, train_names):
    """Clear stale gradients and enable only the intended parameters."""
    for name, model in models.items():
        enabled = name in train_names
        model.zero_grad(set_to_none=True)
        model.requires_grad_(enabled)
        model.train(enabled)


def update(loss, optimiser, parameters):
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss.")

    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        parameters, max_norm=1.0, error_if_nonfinite=True
    )
    optimiser.step()


@torch.no_grad()
def validation_mse(models, loader, device):
    for model in models.values():
        model.eval()

    error = 0.0
    elements = 0
    for (batch,) in loader:
        batch = batch.to(device)
        reconstructed = models["recovery"](models["embedder"](batch))
        loss = nn.functional.mse_loss(
            reconstructed, batch, reduction="sum"
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss.")
        error += loss.item()
        elements += batch.numel()

    return error / elements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--run-name", default="joint_cpu_01")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("Use positive epochs and a batch size of at least 2.")
    if Path(args.run_name).name != args.run_name:
        raise ValueError("Use a simple run directory name.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)

    source = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if source["stage"] != "supervisor":
        raise ValueError("This script starts from supervisor pretraining.")

    config = source["config"]
    hidden = config["hidden_dim"]
    layers = config["num_layers"]
    length = config["window_length"]

    models = {
        "embedder": Embedder(config["input_dim"], hidden, layers),
        "recovery": Recovery(config["input_dim"], hidden, layers),
        "generator": Generator(hidden, hidden, layers),
        "supervisor": Supervisor(
            hidden, config["supervisor_num_layers"]
        ),
        "discriminator": Discriminator(hidden, layers),
    }
    for name in ("embedder", "recovery", "supervisor"):
        models[name].load_state_dict(source[name])
    for model in models.values():
        model.to(device)

    E = models["embedder"]
    R = models["recovery"]
    G = models["generator"]
    S = models["supervisor"]
    D = models["discriminator"]

    messages, books = load_lobster(args.data_dir)
    splits = chronological_split(messages, books)
    mean = source["mean"].numpy()
    scale = source["scale"].numpy()

    tensors = {}
    for name in ("train", "validation"):
        split_books = splits[name][1]
        _, features = build_features(split_books)
        windows, _ = make_windows(
            (features - mean) / scale, split_books, length=length
        )
        tensors[name] = torch.tensor(windows, dtype=torch.float32)

    shuffle_rng = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        TensorDataset(tensors["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_rng,
        drop_last=True,
    )
    validation_loader = DataLoader(
        TensorDataset(tensors["validation"]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    if len(train_loader) == 0:
        raise ValueError("Batch size exceeds the number of training windows.")

    parameter_groups = {
        "d": list(D.parameters()),
        "gs": list(G.parameters()) + list(S.parameters()),
        "er": list(E.parameters()) + list(R.parameters()),
    }
    optimisers = {
        name: torch.optim.Adam(parameters, lr=1e-3)
        for name, parameters in parameter_groups.items()
    }
    criterion = nn.BCEWithLogitsLoss()

    # Separate random stream: monitoring does not consume training noise.
    monitor_rng = torch.Generator().manual_seed(123)
    fixed_noise = torch.randn(
        32, length, hidden, generator=monitor_rng
    ).to(device)

    run_dir = Path("checkpoints") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Device: {device}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(
        "Initial validation reconstruction MSE: "
        f"{validation_mse(models, validation_loader, device):.6f}"
    )

    fields = [
        "d_loss", "g_adversarial", "g_temporal_mse",
        "g_moment", "g_total", "reconstruction_mse", "er_total",
    ]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch"] + fields + ["validation_mse", "elapsed_seconds"]
        )

        for epoch in range(1, args.epochs + 1):
            totals = dict.fromkeys(fields, 0.0)
            observations = 0

            for (batch,) in train_loader:
                batch = batch.to(device)
                count = batch.shape[0]

                # 1. Discriminator update.
                activate(models, {"discriminator"})
                with torch.no_grad():
                    real_latent = E(batch)
                    raw_fake = G(torch.randn(
                        count, length, hidden, device=device
                    ))
                    refined_fake = S(raw_fake)

                real_logits = D(real_latent)
                raw_logits = D(raw_fake)
                refined_logits = D(refined_fake)

                d_loss = (
                    criterion(real_logits, torch.ones_like(real_logits))
                    + criterion(raw_logits, torch.zeros_like(raw_logits))
                    + criterion(
                        refined_logits, torch.zeros_like(refined_logits)
                    )
                )
                update(d_loss, optimisers["d"], parameter_groups["d"])

                # 2. Generator and supervisor update.
                activate(models, {"generator", "supervisor"})
                with torch.no_grad():
                    real_latent = E(batch)

                raw_fake = G(torch.randn(
                    count, length, hidden, device=device
                ))
                refined_fake = S(raw_fake)
                synthetic = R(refined_fake)

                raw_logits = D(raw_fake)
                refined_logits = D(refined_fake)
                adversarial = (
                    criterion(raw_logits, torch.ones_like(raw_logits))
                    + criterion(
                        refined_logits, torch.ones_like(refined_logits)
                    )
                )
                temporal = nn.functional.mse_loss(
                    S(real_latent[:, :-1]), real_latent[:, 1:]
                )

                real_std = torch.sqrt(
                    batch.var(dim=0, unbiased=False) + 1e-6
                )
                fake_std = torch.sqrt(
                    synthetic.var(dim=0, unbiased=False) + 1e-6
                )
                moment = (
                    (batch.mean(dim=0) - synthetic.mean(dim=0)).abs().mean()
                    + (real_std - fake_std).abs().mean()
                )
                g_loss = (
                    adversarial
                    + 100.0 * torch.sqrt(temporal + 1e-8)
                    + 100.0 * moment
                )
                update(g_loss, optimisers["gs"], parameter_groups["gs"])

                # 3. Embedder and recovery update.
                activate(models, {"embedder", "recovery"})
                latent = E(batch)
                reconstruction = nn.functional.mse_loss(R(latent), batch)
                embedding_temporal = nn.functional.mse_loss(
                    S(latent[:, :-1]), latent[:, 1:]
                )
                er_loss = (
                    10.0 * torch.sqrt(reconstruction + 1e-8)
                    + 0.1 * embedding_temporal
                )
                update(er_loss, optimisers["er"], parameter_groups["er"])

                values = [
                    d_loss, adversarial, temporal, moment,
                    g_loss, reconstruction, er_loss,
                ]
                for name, value in zip(fields, values):
                    totals[name] += value.item() * count
                observations += count

            averages = {
                name: value / observations for name, value in totals.items()
            }
            val_mse = validation_mse(models, validation_loader, device)

            # Same noise every epoch, for comparable monitoring samples.
            with torch.no_grad():
                monitor_features = R(S(G(fixed_noise)))
                if not torch.isfinite(monitor_features).all():
                    raise RuntimeError("Non-finite monitoring samples.")

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

            writer.writerow(
                [epoch] + [averages[name] for name in fields]
                + [val_mse, elapsed]
            )
            handle.flush()

            torch.save(
                {
                    "stage": "joint",
                    "epoch": epoch,
                    **{
                        name: model.state_dict()
                        for name, model in models.items()
                    },
                    "optimisers": {
                        name: optimiser.state_dict()
                        for name, optimiser in optimisers.items()
                    },
                    "mean": source["mean"],
                    "scale": source["scale"],
                    "config": {
                        **config,
                        "noise_dim": hidden,
                        "generator_num_layers": layers,
                        "discriminator_num_layers": layers,
                        "joint_batch_size": args.batch_size,
                        "joint_learning_rate": 1e-3,
                        "joint_update_ratio": "D:GS:ER = 1:1:1",
                        "temporal_weight": 100.0,
                        "moment_weight": 100.0,
                        "reconstruction_weight": 10.0,
                        "embedding_temporal_weight": 0.1,
                        "noise_distribution": "standard_normal",
                    },
                    "metrics": {**averages, "validation_mse": val_mse},
                    "fixed_noise": fixed_noise.cpu(),
                    "monitor_features": monitor_features.cpu(),
                    "source_checkpoint": str(args.checkpoint),
                    "torch_rng_state": torch.get_rng_state(),
                    "shuffle_rng_state": shuffle_rng.get_state(),
                    "cuda_rng_states": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda" else []
                    ),
                },
                run_dir / f"joint_epoch_{epoch:03d}.pt",
            )

            print(
                f"Epoch {epoch}: D={averages['d_loss']:.4f} | "
                f"G={averages['g_total']:.4f} | "
                f"moment={averages['g_moment']:.4f} | "
                f"val reconstruction={val_mse:.6f} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        print(
            "Peak allocated GPU memory: "
            f"{torch.cuda.max_memory_allocated(device) / 2**30:.3f} GiB"
        )
    print(f"Saved history and epoch checkpoints in: {run_dir}")


if __name__ == "__main__":
    main()