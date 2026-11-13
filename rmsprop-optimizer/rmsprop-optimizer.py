import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    # pass
    st1, gt1, wt1 = np.array(s), np.array(g), np.array(w)

    s_new = beta*st1 + (1-beta)*(gt1**2)
    w_new = wt1 - (lr/np.sqrt(s_new + eps))*gt1

    return (w_new, s_new)
    






    