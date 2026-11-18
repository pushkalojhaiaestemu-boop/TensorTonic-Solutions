import numpy as np

def eu_dist(x, y):
    d_xy = np.sum((x-y)**2)
    return d_xy

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    # pass

    anchor, positive, negative = np.array(anchor), np.array(positive), np.array(negative)

    range_len = 0
    if len(anchor.shape) == 1:
        range_len = 1
    else:
        range_len = anchor.shape[0]

    L = []
    for i in range(range_len):
        L.append(max(0, eu_dist(anchor[i], positive[i]) - eu_dist(anchor[i], negative[i]) + margin))

    return float(np.mean(np.array(L)))
        
        