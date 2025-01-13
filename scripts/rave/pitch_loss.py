import torch
from .penn_utils import *
import torch.nn.functional as F


def loss(logits, target):
    """Compute loss function"""

    # Convert to pitch bin categories
    bins = frequency_to_bins(target)
    # Determine voiced regions (when target_pitch > 0)
    voiced = target > 0
    # Set unvoiced bins to random values
    bins = torch.where(voiced.to(logits.device),
                       bins.to(logits.device),
                       torch.randint(0, PITCH_BINS, bins.shape, dtype=torch.long).to(logits.device))
    
    # Reshape inputs
    logits = logits.permute(0, 2, 1).reshape(-1, PITCH_BINS)
    bins = bins.flatten().to(logits.device)

    # Maybe blur target
    if GAUSSIAN_BLUR:
        # Cache cents values to evaluate distributions at
        if not hasattr(loss, 'cents'):
            loss.cents = bins_to_cents(
                torch.arange(PITCH_BINS))[:, None]

        # Ensure values are on correct device (no-op if devices are the same)
        loss.cents = loss.cents.to(bins.device)
        # Create normal distributions
        distributions = torch.distributions.Normal(bins_to_cents(bins), 25)
        # Sample normal distributions
        bins = torch.exp(distributions.log_prob(loss.cents)).permute(1, 0)
        # Normalize
        bins = bins / (bins.max(dim=1, keepdims=True).values + 1e-8)
    else:
        # One-hot encoding
        bins = torch.nn.functional.one_hot(bins, PITCH_BINS).float()

    if LOSS == 'binary_cross_entropy':
        # Compute binary cross-entropy loss
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            bins)

    elif LOSS == 'categorical_cross_entropy':
        # Compute categorical cross-entropy loss
        return torch.nn.functional.cross_entropy(logits, bins)
    else:
        raise ValueError(f'Loss {LOSS} is not implemented')
    