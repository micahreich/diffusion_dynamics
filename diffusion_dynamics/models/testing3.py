import torch
from torch.utils.data import DataLoader, Dataset
from diffusion_dynamics.models.diffusion_unet import ConditionalUnet1D, ConditionalUnet1DModel
from diffusers.schedulers import DDPMScheduler
from diffusion_dynamics.models.utils import TensorDataset1D
import torch
from torch.distributions import Normal, Uniform

class ExampleModel(ConditionalUnet1DModel):
    def __init__(self):
        unet = ConditionalUnet1D(
            in_channels=1,
            cond_dim=1,
            base_channels=16,
            dim_mults=[1, 2, 4],
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=True
        )
        
        scheduler = DDPMScheduler(num_train_timesteps=1000,
                                  clip_sample=False,
                                #   variance_type="fixed_small_log",
                                  prediction_type="epsilon")

        super().__init__(unet, scheduler)


if __name__ == "__main__":
    modes = torch.tensor([10.0, -10.0])
    N = 8 * 500
    u_pred_len = 8
    
    class MultiModalDataset(Dataset):
        def __init__(self, modes):
            self.modes = modes
            
            self.seeds = modes[torch.randint(0, len(modes), (N, 1))]
            self.dists = 0.01 * torch.randn(N, u_pred_len) + self.seeds
            
            self.u_pred_len = u_pred_len
            self.obs_history_len = 1
            self.nu = 1
            self.nx = 1
            
        def __len__(self):
            return len(self.seeds)
        
        def __getitem__(self, idx):
            return self.seeds[idx], self.dists[idx].unsqueeze(-1)

    
    # xhist = modes[torch.randint(0, len(modes), (N // 8, 1))]
    # xhist = xhist.repeat_interleave(u_pred_len)
        
    # uhist = 0.01 * torch.randn(N - 1) + xhist[:-1]

    # xhist = xhist.unsqueeze(0).unsqueeze(-1)
    # uhist = uhist.unsqueeze(0).unsqueeze(-1)

    # print(xhist.shape)
    # print(uhist.shape)

    dataset = MultiModalDataset(modes)
    rand_idxs = torch.randint(0, len(dataset), (10,))
    
    for i in rand_idxs:
        print(dataset[i])
        print("--------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ExampleModel()
    model.train(
        dataset,
        n_epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        accumulation_steps=1
    )
    
    model.unet.eval()
    model.scheduler.set_timesteps(num_inference_steps=200)
    
    # Start from random Gaussian noise
    n_samples = 6
    cond = torch.tensor([
        10.0,
        10.0,
        10.0,
        -10.0,
        -10.0,
        -10.0
    ]).reshape(n_samples, 1).to(device)
    
    sample = torch.randn((n_samples, dataset.u_pred_len, dataset.nu), device=device)
    
    # scheduler.timesteps is an iterable of timesteps in descending order
    for t in model.scheduler.timesteps:    
        with torch.no_grad():
            # For each diffusion step, create a batch of the current timestep
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Predict the noise residual
            model_out = model.unet(sample, t_batch, cond)
            
            # Compute the previous sample (one denoising step)
            sample = model.scheduler.step(model_out, t, sample)["prev_sample"]

    print(cond)
    print(sample)
    
    
