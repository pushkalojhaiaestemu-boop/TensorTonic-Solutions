import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    # pass
    PE = np.zeros([seq_len, d_model])

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, min(i+1, d_model-1)] = np.cos(pos / (base**(i/d_model)))
            PE[pos, i] = np.sin(pos / (base**(i/d_model)))
            

    return PE