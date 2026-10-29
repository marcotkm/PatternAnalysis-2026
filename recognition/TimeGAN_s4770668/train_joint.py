"""Joint TimeGAN training with logging and fixed-noise monitoring.

Sources:
    Yoon, Jarrett and van der Schaar (2019),
    "Time-series Generative Adversarial Networks", NeurIPS.
    https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

    Authors' reference implementation:
    https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py

The architecture and/or training objectives are adapted for this project's
PyTorch implementation. This is not an exact reproduction of the reference
implementation. See README.md for implementation differences and AI usage."""

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
from constraints import reconstruct_generated_prices, price_constraint_loss


def activate(models, train_names):
    """Freeze inactive weights while allowing gradients through their GRUs."""
    for name, model in models.items():
        enabled = name in train_names
        model.zero_grad(set_to_none=True)
        model.requires_grad_(enabled)
        model.train(enabled)

        # Frozen GRUs may still need backward propagation to their inputs.
        for layer in model.modules():
            if isinstance(layer, nn.GRU):
                layer.train()


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
    parser.add_argument("--d-lr", type=float, default=1e-3)
    parser.add_argument("--d-threshold", type=float, default=0.0)
    parser.add_argument("--d-update-every", type=int, default=1)
    parser.add_argument("--constraint-weight", type=float, default=0.0)
    args = parser.parse_args()

    if not 0 <= args.constraint_weight < float("inf"):
        raise ValueError("Constraint weight must be finite and non-negative.")
    if args.d_update_every < 1:
        raise ValueError("Discriminator update interval must be positive.")
    if not 0 <= args.d_threshold < float("inf"):
        raise ValueError("Discriminator threshold must be finite and non-negative.")

    if not 0 < args.d_lr < float("inf"):
        raise ValueError("Discriminator learning rate must be positive and finite.")
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
    constraint_mean = source["mean"].to(
        device=device, dtype=torch.float64
    )
    constraint_scale = source["scale"].to(
        device=device, dtype=torch.float64
    )

    tensors = {}
    anchor_tensors = {}
    for name in ("train", "validation"):
        split_books = splits[name][1]
        _, features = build_features(split_books)
        windows, anchors = make_windows(
            (features - mean) / scale, split_books, length=length
        )
        tensors[name] = torch.tensor(windows, dtype=torch.float32)
        anchor_tensors[name] = torch.tensor(
            anchors, dtype=torch.float64
        )

    shuffle_rng = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        TensorDataset(tensors["train"], anchor_tensors["train"]),
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
    learning_rates = {
        "d": args.d_lr,
        "gs": 1e-3,
        "er": 1e-3,
    }
    optimisers = {
        name: torch.optim.Adam(parameters, lr=learning_rates[name])
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
    print(f"Learning rates: {learning_rates}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(
        "Initial validation reconstruction MSE: "
        f"{validation_mse(models, validation_loader, device):.6f}"
    )

    fields = [
        "d_loss", "g_adversarial", "g_temporal_mse",
        "g_moment", "g_total", "reconstruction_mse", "er_total",
        "constraint_total", "constraint_spread",
        "constraint_ask", "constraint_bid",
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

        global_batch = 0
        
        for epoch in range(1, args.epochs + 1):
            totals = dict.fromkeys(fields, 0.0)
            observations = 0
            d_updates = 0

            for batch, anchors in train_loader:
                batch = batch.to(device)
                anchors = anchors.to(device)
                count = batch.shape[0]
                global_batch += 1

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
                if not torch.isfinite(d_loss):
                    raise RuntimeError("Non-finite discriminator loss.")

                if (
                    global_batch % args.d_update_every == 0
                    and d_loss.item() > args.d_threshold
                ):
                    update(d_loss, optimisers["d"], parameter_groups["d"])
                    d_updates += 1

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
                constraint_loss = synthetic.new_zeros(())
                constraint_terms = {
                    "spread": synthetic.new_zeros(()),
                    "ask_ladder": synthetic.new_zeros(()),
                    "bid_ladder": synthetic.new_zeros(()),
                }

                if args.constraint_weight > 0:
                    generated_prices = reconstruct_generated_prices(
                        synthetic,
                        anchors,
                        constraint_mean,
                        constraint_scale,
                    )
                    constraint_loss, constraint_terms = (
                        price_constraint_loss(generated_prices)
                    )
                g_loss = (
                    adversarial
                    + 100.0 * torch.sqrt(temporal + 1e-8)
                    + 100.0 * moment
                    + args.constraint_weight * constraint_loss
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
                    constraint_loss,
                    constraint_terms["spread"],
                    constraint_terms["ask_ladder"],
                    constraint_terms["bid_ladder"],
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
                        "joint_learning_rates": learning_rates.copy(),
                        "joint_update_ratio": (
                            f"D:GS:ER opportunities = "
                            f"1:{args.d_update_every}:{args.d_update_every}; "
                            "D also gated by loss threshold"
                        ),
                        "discriminator_update_every": args.d_update_every,
                        "temporal_weight": 100.0,
                        "moment_weight": 100.0,
                        "reconstruction_weight": 10.0,
                        "embedding_temporal_weight": 0.1,
                        "noise_distribution": "standard_normal",
                        "discriminator_update_threshold": args.d_threshold,
                        "constraint_weight": args.constraint_weight,
                        "constraint_minimum_gap_dollars": 0.01,
                        "constraint_anchor_source": "paired training windows",
                    },
                    "metrics": {
                        **averages,
                        "validation_mse": val_mse,
                        "discriminator_updates": d_updates,
                    },
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

            print(f"Discriminator updates this epoch: {d_updates}", flush=True)
            
            print(
                f"Epoch {epoch}: D={averages['d_loss']:.4f} | "
                f"G={averages['g_total']:.4f} | "
                f"moment={averages['g_moment']:.4f} | "
                f"val reconstruction={val_mse:.6f} | "
                f"elapsed={elapsed:.1f}s",
                f"constraint={averages['constraint_total']:.4f} | ",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    total_seconds = time.perf_counter() - started
    print(
        "Joint-stage wall time "
        "(training, validation, monitoring and checkpoint saves): "
        f"{total_seconds:.3f}s"
    )

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(
            "Peak allocated GPU memory: "
            f"{torch.cuda.max_memory_allocated(device) / 2**20:.2f} MiB"
        )
        print(
            "Peak reserved GPU memory: "
            f"{torch.cuda.max_memory_reserved(device) / 2**20:.2f} MiB"
        )

    print(f"Saved history and epoch checkpoints in: {run_dir}")


if __name__ == "__main__":
    main()