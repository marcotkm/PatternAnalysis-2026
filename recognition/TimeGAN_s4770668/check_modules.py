"""Small CPU check for reconstruction shapes and gradient flow."""

import torch
from torch import nn

from modules import Embedder, Recovery


def main():
    torch.manual_seed(42)

    embedder = Embedder()
    recovery = Recovery()

    # Artificial input checks the wiring, not performance on real data.
    x = torch.randn(4, 100, 40)

    latent = embedder(x)
    reconstructed = recovery(latent)

    assert latent.shape == (4, 100, 24)
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()

    loss = nn.functional.mse_loss(reconstructed, x)
    loss.backward()

    for model_name, model in [
        ("embedder", embedder),
        ("recovery", recovery),
    ]:
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"No gradient: {model_name}.{name}")
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient: {model_name}.{name}"
                )

    print("Input:", tuple(x.shape))
    print("Latent:", tuple(latent.shape))
    print("Reconstruction:", tuple(reconstructed.shape))
    print(f"Initial reconstruction MSE: {loss.item():.6f}")
    print("Shape and gradient checks passed.")


if __name__ == "__main__":
    main()