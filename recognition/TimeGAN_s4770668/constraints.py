"""Differentiable price reconstruction and order-book penalties.

Project-specific extension to the TimeGAN objectives, using penalties on
best-level spread and adjacent ask/bid price gaps. These penalties are not
part of the original TimeGAN method.

See README.md for the mathematical definition, limitations and AI usage."""

import torch


def reconstruct_generated_prices(features, anchors, mean, scale):
    """Reconstruct 100 new snapshots, excluding the initial anchor.

    features: (batch, time, 40), standardised
    anchors: (batch, 20), dollar prices
    Price order: ask1, bid1, ..., ask10, bid10.
    """
    # Double precision helps resolve small gaps between ~$200 prices.
    # Casting preserves the gradient path back to the generator.
    returns = (
        features[..., :20].double() * scale[:20].double()
        + mean[:20].double()
    )
    prices = anchors.double().unsqueeze(1) * torch.exp(
        torch.cumsum(returns, dim=1)
    )

    if not torch.isfinite(prices).all():
        raise RuntimeError("Non-finite reconstructed prices.")

    return prices


def price_constraint_loss(prices, minimum_gap=0.01):
    """Penalise spreads and adjacent gaps smaller than one cent.

    Each term is a mean shortfall measured in one-cent units.
    """
    asks = prices[..., 0::2]
    bids = prices[..., 1::2]

    spread = asks[..., 0] - bids[..., 0]
    ask_gaps = asks[..., 1:] - asks[..., :-1]
    bid_gaps = bids[..., :-1] - bids[..., 1:]

    def shortfall(gaps):
        return torch.relu(
            (minimum_gap - gaps) / minimum_gap
        ).mean()

    terms = {
        "spread": shortfall(spread),
        "ask_ladder": shortfall(ask_gaps),
        "bid_ladder": shortfall(bid_gaps),
    }
    return sum(terms.values()), terms