import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    # pass
    y_true_t, y_score_t = np.array(y_true), np.array(y_score)
    li = []
    for i in range(len(y_true_t)):
        li.append(max(0, margin - y_true_t[i]*y_score_t[i]))

    li = np.array(li)
    if reduction == 'sum':
        return float(np.sum(li))
    else:
        return float(np.mean(li))

    