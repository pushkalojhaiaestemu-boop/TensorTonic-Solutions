import torch
import numpy as np

def squared_error_loss_gradients(predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
    """
    Returns a scalar loss tensor and a prediction-shaped gradient tensor.
    """
    # pass
    preds_np, targ_np = np.array(predictions), np.array(targets)
    L = np.sum((preds_np - targ_np)**2)
    gi = 2*(preds_np - targ_np)

    return torch.tensor(L), torch.tensor(gi)
