from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch
from scipy.linalg import block_diag


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unpack_state(x, nx) -> Tuple[np.ndarray, np.ndarray]:
    single_sample = x.shape == (nx, )

    if single_sample:
        x = x.reshape(1, nx)

    q = x[:, :nx // 2]
    q_dot = x[:, nx // 2:]

    if single_sample:
        return q.reshape((-1, )), q_dot.reshape((-1, ))

    return q, q_dot


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_logit(x):
    return np.log(x / (1 - x))


def torch_to_numpy(*args):
    return [x.detach().cpu().numpy() for x in args] if len(args) > 1 else args[0].detach().cpu().numpy()


def numpy_to_torch(*args):
    return [torch.tensor(x, dtype=torch.float32)
            for x in args] if len(args) > 1 else torch.tensor(args[0], dtype=torch.float32)


if __name__ == "__main__":
    x = torch.tensor([1, 2, 3], dtype=torch.float32)
    print(torch_to_numpy(x))
    print(torch_to_numpy(x, x))
    print(torch_to_numpy(x, x, x))
