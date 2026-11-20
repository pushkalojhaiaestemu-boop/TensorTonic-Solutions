import torch

def denoising_score_matching_loss(x0: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> dict:
    """
    Returns a dict with the batch loss "loss" (Python float) and "per_sample_loss" (tensor of shape (B,)).
    """
    # pass
    sigmas = sigmas[:, None]
    x_cap = x0 + sigmas*noise

    s = (x_cap@weight + bias) / sigmas
    

    p_l = []
    for i in range(len(s)):
        p_l.append(0.5*(sigmas[i]**2)*torch.sum((s[i] + noise[i]/ sigmas[i])**2))

    p_l = torch.tensor(p_l)

    return {"loss": float(p_l.mean()), "per_sample_loss": p_l}
