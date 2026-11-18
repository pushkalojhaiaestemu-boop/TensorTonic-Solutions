import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    a, b = np.array(a), np.array(b)
    y = np.array(y)

    d = np.linalg.norm(a - b, axis=-1)

    # Handle a single embedding pair (1D input)
    if d.ndim == 0:
        d = np.array([d])

    loss = []

    for i in range(len(y)):
        temp_loss = y[i] * (d[i]**2) + (1-y[i]) * (max(0, margin - d[i])**2)
        loss.append(temp_loss)

    loss = np.array(loss)

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))