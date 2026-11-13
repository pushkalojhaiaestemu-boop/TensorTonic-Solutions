import math
import numpy as np

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    # Write code here
    # pass
    y_truet, y_predt = np.array(y_true), np.array(y_pred)
    p = -1
    L = []
    for i in range(len(y_truet)):
        p = min(1 - eps, max(eps, y_predt[i]))
        loss = -(y_truet[i]*np.log(p) + (1-y_truet[i])*np.log(1-p))
        L.append(loss)

    return L
        