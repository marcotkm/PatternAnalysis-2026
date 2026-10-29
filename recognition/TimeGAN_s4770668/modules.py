"""Neural network components for TimeGAN.

Sources:
    Yoon, Jarrett and van der Schaar (2019),
    "Time-series Generative Adversarial Networks", NeurIPS.
    https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

    Authors' reference implementation:
    https://github.com/jsyoon0823/TimeGAN/blob/master/timegan.py

The architecture and/or training objectives are adapted for this project's
PyTorch implementation. This is not an exact reproduction of the reference
implementation. See README.md for implementation differences and AI usage. """

import torch
from torch import nn

class Embedder(nn.Module):
    """Map observed feature sequences into latent sequences."""

    def __init__(self, input_dim=40, hidden_dim=24, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        hidden, _ = self.rnn(x)
        return torch.sigmoid(self.projection(hidden))


class Recovery(nn.Module):
    """Reconstruct standardised features from latent sequences."""

    def __init__(self, output_dim=40, hidden_dim=24, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        hidden, _ = self.rnn(latent)
        return self.projection(hidden)

class Supervisor(nn.Module):
    """Predict next-step latent states from preceding latent states."""

    def __init__(self, hidden_dim=24, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, latent):
        hidden, _ = self.rnn(latent)
        return torch.sigmoid(self.projection(hidden))

class Generator(nn.Module):
    """Map random noise sequences into latent sequences."""

    def __init__(self, noise_dim=24, hidden_dim=24, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, noise):
        hidden, _ = self.rnn(noise)
        return torch.sigmoid(self.projection(hidden))


class Discriminator(nn.Module):
    """Produce real-versus-generated logits for latent sequences."""

    def __init__(self, hidden_dim=24, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, latent):
        hidden, _ = self.rnn(latent)
        return self.projection(hidden)