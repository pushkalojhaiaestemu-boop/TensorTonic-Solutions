import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    # Write code here
    # pass
    p, q = np.array(p), np.array(q)
    d_kl = np.sum((p+eps)*np.log((p+eps)/(q+eps)))

    return float(d_kl)
    
    