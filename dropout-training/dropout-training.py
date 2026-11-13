import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    # pass
    xt = np.array(x)
    if rng is None:
        dropper = np.random.random(xt.shape)
    else:
        dropper = rng.random(xt.shape)

    drop_mask = dropper >= p
    dropout_pattern = drop_mask / (1-p)
    
    xt = xt*dropout_pattern
    return (xt, dropout_pattern)