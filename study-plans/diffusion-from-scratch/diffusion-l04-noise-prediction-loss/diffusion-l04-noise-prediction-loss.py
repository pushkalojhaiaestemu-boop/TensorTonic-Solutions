import torch

def noise_prediction_loss(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    alpha_bars: torch.Tensor,
    weight: torch.Tensor,
    time_weight: torch.Tensor,
    bias: torch.Tensor
) -> dict:

    B = x0.shape[0]
    T = len(alpha_bars)

    alpha_t = alpha_bars[timesteps]
    alpha_t = alpha_t.view(-1, *([1] * (x0.ndim - 1)))

    xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

    xt_t = xt.reshape(B, -1)
    noise_t = noise.reshape(B, -1)

    e_t = xt_t @ weight + (timesteps.to(x0.dtype).unsqueeze(1) / T) * time_weight + bias

    li = torch.mean((e_t - noise_t) ** 2, dim=1)

    L = li.mean()

    return {
        "loss": L.item(),
        "per_sample_loss": li
    }