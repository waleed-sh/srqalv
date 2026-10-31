import math
import numpy as np
Array = np.ndarray


def four_valent_commutator_offdiag(two_j: int, upper_channel: int) -> float:
    t = int(two_j)
    x = int(upper_channel)
    if t <= 0:
        raise ValueError("two_j must be positive.")
    if x <= 0 or x > 2 * t or x % 2:
        raise ValueError("upper_channel must be an even doubled channel in 2..2*two_j.")


    term = (2 * t + x + 2) * x * x * (2 * t - x + 2)
    return abs(float(term)) / (16.0 * math.sqrt((x + 1) * (x - 1)))


def four_valent_q_commutator(two_j: int) -> tuple[Array, Array]:
    channels = np.arange(0, 2 * int(two_j) + 1, 2, dtype=np.int32)
    dim = channels.size
    q_comm = np.zeros((dim, dim), dtype=np.complex128)
    for upper_idx in range(1, dim):
        mag = four_valent_commutator_offdiag(two_j, int(channels[upper_idx]))
        q_comm[upper_idx - 1, upper_idx] = -mag
        q_comm[upper_idx, upper_idx - 1] = mag
    return channels, q_comm

