import os
from collections import OrderedDict
from datetime import datetime

import einops
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.downsampling import Downsample1D
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.resnet import Conv1dBlock, ResidualTemporalBlock1D
from diffusers.models.upsampling import Upsample1D
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion_dynamics.models.mlp import MLP
from diffusion_dynamics.models.utils import (
    TensorDataset1D,
    TensorDataset1DStats,
    activation_dict,
)
from diffusion_dynamics.utils import torch_to_numpy


# Sinusoidal positional embedding (same as your original code)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x is assumed to be a tensor of shape (batch,)
        return get_timestep_embedding(x, self.dim)


# class MLP(nn.Module):
#     def __init__(
#         self,
#         dim_list,
#         append_dim=0,
#         append_layers=None,
#         activation_type="Tanh",
#         out_activation_type="Identity",
#         use_layernorm=False,
#         use_layernorm_final=False,
#         dropout=0,
#         use_drop_final=False,
#     ):
#         super(MLP, self).__init__()

#         # Construct module list: if use `Python List`, the modules are not
#         # added to computation graph. Instead, we should use `nn.ModuleList()`.
#         self.moduleList = nn.ModuleList()
#         self.append_layers = append_layers
#         num_layer = len(dim_list) - 1
#         for idx in range(num_layer):
#             i_dim = dim_list[idx]
#             o_dim = dim_list[idx + 1]
#             if append_dim > 0 and idx in append_layers:
#                 i_dim += append_dim
#             linear_layer = nn.Linear(i_dim, o_dim)

#             # Add module components
#             layers = [("linear_1", linear_layer)]
#             if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
#                 layers.append(("norm_1", nn.LayerNorm(o_dim)))
#             if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
#                 layers.append(("dropout_1", nn.Dropout(dropout)))

#             # add activation function
#             act = (
#                 activation_dict[activation_type]
#                 if idx != num_layer - 1
#                 else activation_dict[out_activation_type]
#             )
#             layers.append(("act_1", act))

#             # re-construct module
#             module = nn.Sequential(OrderedDict(layers))
#             self.moduleList.append(module)

#     def forward(self, x, append=None):
#         for layer_ind, m in enumerate(self.moduleList):
#             if append is not None and layer_ind in self.append_layers:
#                 x = torch.cat((x, append), dim=-1)
#             x = m(x)
#         return x


class FiLMTwoLayerPreActivationResNetLinear(nn.Module):
    """
    A two-layer residual block that uses FiLM conditioning.
    The conditioning vector (e.g. concatenated timestep and context embeddings)
    is used to generate scale and shift parameters that modulate the activations.
    """
    def __init__(self, hidden_dim, cond_dim, activation_type="Mish", use_layernorm=False, dropout=0):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_dict[activation_type]

        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        else:
            self.norm1 = self.norm2 = None

        # FiLM conditioning layers for each sub-block:
        # They map the conditioning vector to a pair of scale and shift parameters.
        self.film1 = nn.Linear(cond_dim, hidden_dim * 2)
        self.film2 = nn.Linear(cond_dim, hidden_dim * 2)

        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for FiLM residual MLP!")

    def forward(self, x, cond):
        """
        x: Tensor of shape [batch, hidden_dim]
        cond: Conditioning vector of shape [batch, cond_dim]
        """
        x_input = x

        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.l1(self.act(x))
        # Compute FiLM parameters for first layer
        film_params1 = self.film1(cond)  # shape: [batch, hidden_dim * 2]
        scale1, shift1 = film_params1.chunk(2, dim=-1)
        x = x * scale1 + shift1

        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.l2(self.act(x))
        # FiLM conditioning after second layer
        film_params2 = self.film2(cond)
        scale2, shift2 = film_params2.chunk(2, dim=-1)
        x = x * scale2 + shift2

        return x + x_input


class FiLMResidualMLP(nn.Module):
    """
    A residual MLP that uses FiLM conditioning in its residual blocks.
    The network is intended for tasks like noise prediction in diffusion models,
    where the denoising is conditioned on both a timestep embedding and a context vector.
    
    Arguments:
      dim_list: List of dimensions, e.g. [input_dim, hidden_dim, ..., output_dim]
      cond_dim: Dimension of the conditioning vector (e.g. concatenated timestep & context)
      activation_type: Name of the activation to use (per activation_dict)
      out_activation_type: Activation at the output (per activation_dict)
      use_layernorm: Whether to use LayerNorm in residual blocks
      use_layernorm_final: Whether to apply LayerNorm at the output layer
      dropout: Dropout probability (not implemented in this example)
    """
    def __init__(
        self,
        dim_list,
        cond_dim,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
    ):
        super().__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 2  # excluding input & output layers
        assert num_hidden_layers % 2 == 0, "Number of hidden layers should be even"

        self.input_layer = nn.Linear(dim_list[0], hidden_dim)

        # Create FiLM-conditioned residual blocks.
        # Each block here is FiLMTwoLayerPreActivationResNetLinear.
        # We assume that each block uses FiLM conditioning.
        self.residual_blocks = nn.ModuleList([
            FiLMTwoLayerPreActivationResNetLinear(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                activation_type=activation_type,
                use_layernorm=use_layernorm,
                dropout=dropout,
            ) for _ in range(num_hidden_layers // 2)
        ])

        # Final layers
        layers = [nn.Linear(hidden_dim, dim_list[-1])]
        if use_layernorm_final:
            layers.append(nn.LayerNorm(dim_list[-1]))
        layers.append(activation_dict[out_activation_type])
        self.output_layers = nn.Sequential(*layers)

    def forward(self, x, cond):
        """
        x: Input tensor of shape [batch, input_dim]
        cond: Conditioning vector of shape [batch, cond_dim]
              (e.g. concatenation of timestep sinusoidal embedding and context vector)
        """
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x, cond)
        x = self.output_layers(x)
        return x


class ConditionalDenoisingMLP(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=128, n_blocks=4, cond_predict_scale=False):
        """
        Args:
            input_dim: Dimension of the input vector.
            cond_dim: Dimension of the condition vector.
            hidden_dim: Hidden dimension used in the network.
            n_blocks: Number of residual blocks.
            cond_predict_scale: Whether to use FiLM conditioning that predicts scale and bias.
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        time_embed_dim = hidden_dim
        cond_embed_dim = hidden_dim

        self.pos_emb = nn.Sequential(*[
            SinusoidalPosEmb(dim=time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        ])

        self.cond_emb = nn.Sequential(
            *[nn.Linear(cond_dim, cond_embed_dim * 4),
              nn.Mish(),
              nn.Linear(cond_embed_dim * 4, cond_embed_dim)])

        dim_list = [input_dim] + [hidden_dim] * n_blocks + [input_dim]

        # self.network = FiLMResidualMLP(
        #     dim_list=dim_list,
        #     cond_dim=cond_embed_dim + time_embed_dim,
        #     activation_type="Mish",
        #     out_activation_type="Identity",
        #     use_layernorm=False,
        # )
        self.network = MLP(
            dim_list=dim_list,
            append_dim=cond_embed_dim + time_embed_dim,
            append_layers=[0],
            activation_type="Tanh",
            out_activation_type="Identity",
        )

    def forward(self, x, t, cond):
        _, T, nu = x.shape
        assert T * nu == self.input_dim, f"Input dimension mismatch: {T * nu} != {self.input_dim}"

        # x has shape (batch, T, nu), we need to flatten it to (batch, T*nu) in case of multiple u predictions
        x = einops.rearrange(x, 'b h c -> b (h c)')

        t_emb = self.pos_emb(t)
        cond_emb = self.cond_emb(cond)
        context_emb = torch.cat([t_emb, cond_emb], dim=-1)

        out = self.network(x, context_emb)
        out = einops.rearrange(out, 'b (h c) -> b h c', c=nu)

        return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from diffusers.schedulers import DDPMScheduler

    batch_size = 128
    input_dim = 1
    cond_dim = 1
    n_blocks = 4

    mlp = ConditionalDenoisingMLP(input_dim=input_dim,
                                  cond_dim=cond_dim,
                                  hidden_dim=128,
                                  n_blocks=n_blocks,
                                  cond_predict_scale=True)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        clip_sample=False,
        #   variance_type="fixed_small_log",
        prediction_type="epsilon")

    # Dummy dataset
    modes = torch.tensor([10.0, -10.0])
    N = 2048
    u_pred_len = 1

    class MultiModalDataset(Dataset):
        def __init__(self, modes):
            self.modes = modes

            self.seeds = modes[torch.randint(0, len(modes), (N, 1))]
            self.dists = torch.randn(N, u_pred_len) + self.seeds

            self.u_pred_len = u_pred_len
            self.obs_history_len = 1
            self.nu = 1
            self.nx = 1

            self.stats = TensorDataset1DStats(self.nu,
                                              self.nx,
                                              self.u_pred_len,
                                              self.obs_history_len,
                                              N_samples=N,
                                              normalized=False)

        def __len__(self):
            return len(self.seeds)

        def __getitem__(self, idx):
            return self.seeds[idx], torch.reshape(self.dists[idx], shape=(u_pred_len, 1))

    # Dummy inputs:
    dataset = MultiModalDataset(modes)
    rand_idxs = torch.randint(0, len(dataset), (5, ))

    for i in rand_idxs:
        print(f"Sample index {i}:", dataset[i])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_model = ConditionalDiffusionAgent(mlp, scheduler)

    diffusion_model.train(dataset, n_epochs=100, batch_size=batch_size, learning_rate=1e-3, accumulation_steps=1)

    sample_cond = torch.cat([10 * torch.ones(128).reshape(-1, 1), -10 * torch.ones(128).reshape(-1, 1)], dim=0)

    sample_dist = diffusion_model.sample(n_samples=sample_cond.shape[0],
                                         cond=sample_cond,
                                         data_stats=dataset.stats,
                                         device=device,
                                         num_inference_steps=200)

    sample_dist = torch_to_numpy(sample_dist.reshape(-1, 1))

    plt.hist(sample_dist.flatten(), bins=30, color='blue', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')
    plt.show()

    # for (cond, sample) in zip(sample_cond, sample_dist):
    #     print("Condition:", cond)
    #     print("Sample:", sample.flatten())
