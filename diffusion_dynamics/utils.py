import numpy as np
from typing import Any, Tuple
from scipy.linalg import block_diag
from dataclasses import dataclass
import torch


def unpack_state(x, nx) -> Tuple[np.ndarray, np.ndarray]:
    single_sample = x.shape == (nx,)

    if single_sample:
        x = x.reshape(1, nx)

    q = x[:, : nx // 2]
    q_dot = x[:, nx // 2 :]

    if single_sample:
        return q.reshape((-1,)), q_dot.reshape((-1,))

    return q, q_dot


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_logit(x):
    return np.log(x / (1 - x))


def ptu_torch_to_numpy(x):
    return x.detach().cpu().numpy()
