import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    # pass
    r_t, f_t = np.array(real_scores), np.array(fake_scores)
    L = np.mean(f_t) - np.mean(r_t)
    return float(L)