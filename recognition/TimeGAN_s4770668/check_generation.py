"""Check the generation path and adversarial gradient flow."""

import torch
from torch import nn

from modules import Generator, Supervisor, Recovery, Discriminator


def main():
    torch.manual_seed(42)

    generator = Generator()
    supervisor = Supervisor()
    recovery = Recovery()
    discriminator = Discriminator()

    noise = torch.randn(4, 100, 24)

    generated_latent = generator(noise)
    refined_latent = supervisor(generated_latent)
    synthetic = recovery(refined_latent)
    logits = discriminator(refined_latent)

    assert generated_latent.shape == (4, 100, 24)
    assert refined_latent.shape == (4, 100, 24)
    assert synthetic.shape == (4, 100, 40)
    assert logits.shape == (4, 100, 1)

    for tensor in (generated_latent, refined_latent, synthetic, logits):
        assert torch.isfinite(tensor).all()

    # Isolate a generator-side adversarial update.
    # Freeze discriminator parameters but preserve gradients through its input.
    discriminator.requires_grad_(False)
    logits = discriminator(refined_latent)

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits)
    )
    loss.backward()

    for model_name, model in [
        ("generator", generator),
        ("supervisor", supervisor),
    ]:
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"No gradient: {model_name}.{name}")
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient: {model_name}.{name}"
                )

    assert all(p.grad is None for p in discriminator.parameters())

    print("Synthetic features:", tuple(synthetic.shape))
    print("Discriminator logits:", tuple(logits.shape))
    print(f"Generator adversarial loss: {loss.item():.6f}")
    print("Generation shape and adversarial gradient checks passed.")


if __name__ == "__main__":
    main()