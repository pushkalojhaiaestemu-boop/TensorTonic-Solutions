import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    # pass
    y_true_t, y_pred_t = np.array(y_true), np.array(y_pred)
    error = y_true_t - y_pred_t
    L_e = []
    for i in range(len(error)):
        e = error[i]
        if abs(e)<=delta:
            L_e.append(0.5*(e**2))
        else:
            L_e.append(delta*(abs(e) - 0.5*delta))


    return float(np.mean(np.array(L_e)))