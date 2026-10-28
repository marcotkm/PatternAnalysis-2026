"""Check reconstruction, violation detection and gradient flow."""

import torch

from constraints import (
    reconstruct_generated_prices,
    price_constraint_loss,
)

dtype = torch.float64

# A valid ten-level book with two-cent adjacent gaps.
asks = 100.02 + 0.02 * torch.arange(10, dtype=dtype)
bids = 100.00 - 0.02 * torch.arange(10, dtype=dtype)
anchors = torch.stack((asks, bids), dim=-1).reshape(1, 20)

mean = torch.zeros(40, dtype=dtype)
scale = torch.ones(40, dtype=dtype)

# Zero returns should reproduce the anchor at every step.
features = torch.zeros(1, 4, 40, dtype=dtype)
prices = reconstruct_generated_prices(features, anchors, mean, scale)

torch.testing.assert_close(
    prices, anchors.unsqueeze(1).expand(1, 4, 20)
)
valid_loss, _ = price_constraint_loss(prices)
assert valid_loss.item() == 0.0
print("Valid-book reconstruction and zero-penalty checks passed.")

# Test each violation independently.
for name, column, value in [
    ("crossed spread", 0, 99.99),
    ("ask inversion", 2, 100.01),
    ("bid inversion", 3, 100.01),
]:
    invalid = prices.clone()
    invalid[..., column] = value
    loss, terms = price_constraint_loss(invalid)
    assert loss.item() > 0.0
    print(name, {key: round(v.item(), 6) for key, v in terms.items()})

# Force best ask to fall below best bid, then backpropagate
# through reconstruction all the way to generated features.
bad_features = features.clone()
bad_features[0, 0, 0] = -0.001
bad_features.requires_grad_(True)

bad_prices = reconstruct_generated_prices(
    bad_features, anchors, mean, scale
)
loss, _ = price_constraint_loss(bad_prices)
loss.backward()

assert bad_features.grad is not None
assert torch.isfinite(bad_features.grad).all()
assert bad_features.grad[..., :20].abs().sum().item() > 0.0
assert bad_features.grad[..., 20:].abs().sum().item() == 0.0

print("Finite, nonzero price gradients passed.")
print("All constraint checks passed.")