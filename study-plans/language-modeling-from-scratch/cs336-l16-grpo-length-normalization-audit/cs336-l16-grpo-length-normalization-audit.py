import torch
import numpy as np

def grpo_length_normalization_audit(
    token_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
) -> dict:

    n_i = token_mask.sum(dim=1)

    sequence_summed = (
        token_log_probs * token_mask
    ).sum(dim=1) * advantages

    token_averaged = sequence_summed / n_i.clamp(min=1)

    numerator = sequence_summed.abs().sum()
    denominator = token_averaged.abs().sum()

    magnitude_ratio = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(denominator)
    )

    return {
        "sequence_summed": sequence_summed,
        "token_averaged": token_averaged,
        "magnitude_ratio": magnitude_ratio
    }