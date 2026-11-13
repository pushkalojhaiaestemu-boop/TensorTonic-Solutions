import math
import numpy as np

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    # Write code here
    # pass
    x1_t, x2_t = np.array(x1), np.array(x2)
    cos_loss = np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

    L = 0
    if label == 1:
        L = float(1-cos_loss)
    else:
        L = float(max(0, cos_loss - margin))

    return L