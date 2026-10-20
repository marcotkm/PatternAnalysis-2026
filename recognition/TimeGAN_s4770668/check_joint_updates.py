"""First joint-training check: one isolated discriminator update."""

import argparse
from logging import config
from pathlib import Path

import torch
from torch import nn

from dataset import (
    load_lobster,
    chronological_split,
    build_features,
    make_windows,
)
from modules import Embedder, Generator, Supervisor, Discriminator, Recovery


def snapshot(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def changed(model, before):
    return any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(2)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint["stage"] != "supervisor":
        raise ValueError("Expected a supervisor checkpoint.")

    config = checkpoint["config"]
    hidden_dim = config["hidden_dim"]

    embedder = Embedder(
        input_dim=config["input_dim"],
        hidden_dim=hidden_dim,
        num_layers=config["num_layers"],
    )
    supervisor = Supervisor(
        hidden_dim=hidden_dim,
        num_layers=config["supervisor_num_layers"],
    )
    generator = Generator(
        noise_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=config["num_layers"],
    )
    discriminator = Discriminator(
        hidden_dim=hidden_dim,
        num_layers=config["num_layers"],
    )

    embedder.load_state_dict(checkpoint["embedder"])
    supervisor.load_state_dict(checkpoint["supervisor"])

    messages, books = load_lobster(args.data_dir)
    train_books = chronological_split(messages, books)["train"][1]
    _, features = build_features(train_books)

    mean = checkpoint["mean"].numpy()
    scale = checkpoint["scale"].numpy()
    windows, _ = make_windows(
        (features - mean) / scale,
        train_books,
        length=config["window_length"],
    )

    indices = torch.randperm(len(windows))[:32]
    batch = torch.tensor(
        windows[indices.numpy()], dtype=torch.float32
    )

    models = {
        "embedder": embedder,
        "supervisor": supervisor,
        "generator": generator,
        "discriminator": discriminator,
    }
    before = {name: snapshot(model) for name, model in models.items()}

    embedder.eval()
    supervisor.eval()
    generator.eval()
    discriminator.train()

    # No gradients into the networks supplying discriminator inputs.
    with torch.no_grad():
        real_latent = embedder(batch)
        noise = torch.randn(
            batch.shape[0], batch.shape[1], hidden_dim
        )
        raw_fake = generator(noise)
        refined_fake = supervisor(raw_fake)

    optimiser = torch.optim.Adam(
        discriminator.parameters(), lr=1e-3
    )
    criterion = nn.BCEWithLogitsLoss()

    real_logits = discriminator(real_latent)
    raw_fake_logits = discriminator(raw_fake)
    refined_fake_logits = discriminator(refined_fake)

    real_loss = criterion(real_logits, torch.ones_like(real_logits))
    raw_fake_loss = criterion(
        raw_fake_logits, torch.zeros_like(raw_fake_logits)
    )
    refined_fake_loss = criterion(
        refined_fake_logits, torch.zeros_like(refined_fake_logits)
    )

    # Initial choice: equal weighting for the two fake-input terms.
    loss = real_loss + refined_fake_loss + raw_fake_loss

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite discriminator loss.")

    optimiser.zero_grad(set_to_none=True)
    loss.backward()

    for name in ("embedder", "supervisor", "generator"):
        if any(p.grad is not None for p in models[name].parameters()):
            raise RuntimeError(f"Unexpected gradient in {name}.")

    for name, parameter in discriminator.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Invalid discriminator gradient: {name}")

    nn.utils.clip_grad_norm_(
        discriminator.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    optimiser.step()

    for name, model in models.items():
        did_change = changed(model, before[name])
        print(f"{name} parameters changed: {did_change}")

        if did_change != (name == "discriminator"):
            raise RuntimeError(f"Unexpected parameter update: {name}")

    print(f"Real loss: {real_loss.item():.6f}")
    print(f"Raw fake loss: {raw_fake_loss.item():.6f}")
    print(f"Refined fake loss: {refined_fake_loss.item():.6f}")
    print(f"Total discriminator loss: {loss.item():.6f}")
    print("Discriminator update isolation check passed.")

    recovery = Recovery(
        output_dim=config["input_dim"],
        hidden_dim=hidden_dim,
        num_layers=config["num_layers"],
    )
    recovery.load_state_dict(checkpoint["recovery"])
    models["recovery"] = recovery

    # Clear gradients left by the preceding discriminator update.
    for model in models.values():
        model.zero_grad(set_to_none=True)

    # Freeze parameters, but preserve differentiation through these networks.
    for model in (embedder, recovery, discriminator):
        model.requires_grad_(False)
        model.eval()

    generator.requires_grad_(True)
    supervisor.requires_grad_(True)
    generator.train()
    supervisor.train()

    before = {name: snapshot(model) for name, model in models.items()}

    gs_parameters = (
        list(generator.parameters()) + list(supervisor.parameters())
    )
    gs_optimiser = torch.optim.Adam(gs_parameters, lr=1e-3)

    with torch.no_grad():
        real_latent = embedder(batch)

    noise = torch.randn(
        batch.shape[0], batch.shape[1], hidden_dim
    )
    raw_fake = generator(noise)
    refined_fake = supervisor(raw_fake)

    # Recovery is frozen, but its operations must remain in the graph
    # so the moment loss can reach generator and supervisor parameters.
    synthetic = recovery(refined_fake)

    raw_logits = discriminator(raw_fake)
    refined_logits = discriminator(refined_fake)

    adversarial_loss = (
        criterion(raw_logits, torch.ones_like(raw_logits))
        + criterion(refined_logits, torch.ones_like(refined_logits))
    )

    temporal_prediction = supervisor(real_latent[:, :-1])
    temporal_loss = nn.functional.mse_loss(
        temporal_prediction, real_latent[:, 1:]
    )

    # Compare per-time-step, per-feature moments across the batch.
    # This does not enforce order book constraints.
    real_mean = batch.mean(dim=0)
    fake_mean = synthetic.mean(dim=0)
    real_std = torch.sqrt(batch.var(dim=0, unbiased=False) + 1e-6)
    fake_std = torch.sqrt(synthetic.var(dim=0, unbiased=False) + 1e-6)

    moment_loss = (
        (real_mean - fake_mean).abs().mean()
        + (real_std - fake_std).abs().mean()
    )

    temporal_term = 100.0 * torch.sqrt(temporal_loss + 1e-8)
    moment_term = 100.0 * moment_loss
    gs_loss = adversarial_loss + temporal_term + moment_term

    if not torch.isfinite(gs_loss):
        raise RuntimeError("Non-finite generator/supervisor loss.")

    gs_optimiser.zero_grad(set_to_none=True)
    gs_loss.backward()

    for name in ("embedder", "recovery", "discriminator"):
        if any(p.grad is not None for p in models[name].parameters()):
            raise RuntimeError(f"Unexpected gradient in {name}.")

    for model_name in ("generator", "supervisor"):
        for name, parameter in models[model_name].named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"No gradient: {model_name}.{name}")
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient: {model_name}.{name}"
                )

    nn.utils.clip_grad_norm_(
        gs_parameters, max_norm=1.0, error_if_nonfinite=True
    )
    gs_optimiser.step()

    print("\nGenerator/supervisor update")
    for name, model in models.items():
        did_change = changed(model, before[name])
        print(f"{name} parameters changed: {did_change}")

        expected = name in ("generator", "supervisor")
        if did_change != expected:
            raise RuntimeError(f"Unexpected parameter update: {name}")

    print(f"Adversarial loss: {adversarial_loss.item():.6f}")
    print(f"Temporal MSE: {temporal_loss.item():.6f}")
    print(f"Weighted temporal term: {temporal_term.item():.6f}")
    print(f"Moment loss: {moment_loss.item():.6f}")
    print(f"Weighted moment term: {moment_term.item():.6f}")
    print(f"Total generator/supervisor loss: {gs_loss.item():.6f}")
    print("Generator/supervisor update isolation check passed.")

    # Clear gradients left by the previous update.
    for model in models.values():
        model.zero_grad(set_to_none=True)

    for model in (generator, supervisor, discriminator):
        model.requires_grad_(False)
        model.eval()

    for model in (embedder, recovery):
        model.requires_grad_(True)
        model.train()

    before = {name: snapshot(model) for name, model in models.items()}

    er_parameters = (
        list(embedder.parameters()) + list(recovery.parameters())
    )
    er_optimiser = torch.optim.Adam(er_parameters, lr=1e-3)

    latent = embedder(batch)
    reconstructed = recovery(latent)

    reconstruction_loss = nn.functional.mse_loss(
        reconstructed, batch
    )

    # Do not detach latent or use no_grad here:
    # the temporal objective must reach the embedder.
    predicted_next = supervisor(latent[:, :-1])
    embedding_temporal_loss = nn.functional.mse_loss(
        predicted_next, latent[:, 1:]
    )

    reconstruction_term = 10.0 * torch.sqrt(
        reconstruction_loss + 1e-8
    )
    embedding_temporal_term = 0.1 * embedding_temporal_loss
    er_loss = reconstruction_term + embedding_temporal_term

    if not torch.isfinite(er_loss):
        raise RuntimeError("Non-finite embedder/recovery loss.")

    er_optimiser.zero_grad(set_to_none=True)
    er_loss.backward()

    for name in ("generator", "supervisor", "discriminator"):
        if any(p.grad is not None for p in models[name].parameters()):
            raise RuntimeError(f"Unexpected gradient in {name}.")

    for model_name in ("embedder", "recovery"):
        for name, parameter in models[model_name].named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"No gradient: {model_name}.{name}")
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient: {model_name}.{name}"
                )

    nn.utils.clip_grad_norm_(
        er_parameters, max_norm=1.0, error_if_nonfinite=True
    )
    er_optimiser.step()

    print("\nEmbedder/recovery update")
    for name, model in models.items():
        did_change = changed(model, before[name])
        print(f"{name} parameters changed: {did_change}")

        expected = name in ("embedder", "recovery")
        if did_change != expected:
            raise RuntimeError(f"Unexpected parameter update: {name}")

    print(f"Reconstruction MSE: {reconstruction_loss.item():.6f}")
    print(f"Embedding temporal MSE: {embedding_temporal_loss.item():.6f}")
    print(f"Total embedder/recovery loss: {er_loss.item():.6f}")
    print("Embedder/recovery update isolation check passed.")


if __name__ == "__main__":
    main()