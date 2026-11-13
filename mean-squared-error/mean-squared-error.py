import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    # Write code here
    # pass
    y_predt, y_truet = np.array(y_pred), np.array(y_true)
    mse = np.mean((y_predt - y_truet)**2)
    return mse