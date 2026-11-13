import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    # pass
    if len(seqs) == 0:
        return np.ndarray([0, 0], dtype = int)   
    if max_len is None:
        max_len = max([len(seq) for seq in seqs])
    N, L = len(seqs), max_len
    padded_arr = np.ndarray([N, L], dtype = int)
    i = 0
    for seq in seqs: 
        if max_len > len(seq):
            padded_arr[i, :len(seq)] = seq
            padded_arr[i, len(seq):] = pad_value
        else:
            padded_arr[i, :] = seq[:max_len]
            
        i += 1
    return padded_arr