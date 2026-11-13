import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Returns trained weights and bias as (w, b).
    """

    N, D = X.shape

    # 1. Initialize weights and bias
    w = np.zeros(D)
    b = 0.0

    # 2. Gradient descent
    for step in range(steps):

        # Forward pass
        z = X @ w + b

        # Predictions
        p = _sigmoid(z)

        # 3. Binary cross entropy loss
        eps = 1e-15
        p_clipped = np.clip(p, eps, 1 - eps)

        loss = -(1 / N) * np.sum(
            y * np.log(p_clipped)
            + (1 - y) * np.log(1 - p_clipped)
        )

        # 4. Calculate gradients
        dz = p - y

        dw = (1 / N) * (X.T @ dz)
        db = (1 / N) * np.sum(dz)

        # 5. Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b

    