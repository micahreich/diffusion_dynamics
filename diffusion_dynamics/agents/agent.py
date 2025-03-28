import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pytz
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_dynamics.models.utils import (
    SaveModelParams,
    TensorDataset1D,
    TensorDataset1DStats,
)
from diffusion_dynamics.utils import np_logit, np_sigmoid


class BehaviorCloningAgent:
    def __init__(self, model):
        self.model = model
        self.stats: Optional[TensorDataset1DStats] = None

    def to(self, device) -> "BehaviorCloningAgent":
        self.model.to(device)
        return self

    def train(
        self,
        dataset: Dataset,
        n_epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        accumulation_steps=2,
        save_model_params: Optional[SaveModelParams] = None,
    ) -> None:
        raise NotImplementedError

    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _save_model(self, save_model_params: SaveModelParams):
        # Save the trained model weights
        if save_model_params.save_model_name is None:
            nyc_tz = pytz.timezone('America/New_York')
            time_str = datetime.now(nyc_tz).strftime("%Y-%m-%d__%H-%M-%S")
            save_model_params.save_model_name = f"diffusion1d_{time_str}"

        save_fpath_full = os.path.join(save_model_params.save_full_fpath, save_model_params.save_model_name)
        root, ext = os.path.splitext(save_fpath_full)

        if ext and ext == ".pt":
            save_fpath_full = root
        elif ext and ext != ".pt":
            save_fpath_full = f"{root}.pt"
        else:
            save_fpath_full = f"{save_fpath_full}.pt"

        print(f"Saving model to {save_fpath_full}...")
        torch.save(self, save_fpath_full)

    @classmethod
    def load_trained_model(cls, saved_model_path: str) -> "BehaviorCloningAgent":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.load(saved_model_path, map_location=device)
