from re import X
import torch

import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
import torchvision.transforms.functional as TF

class VarianceScheduler:
    """
    This class is used to keep track of statistical variables used in the diffusion model
    and also adding noise to the data
    """
    def __init__(self, beta_start: float=0.0001, beta_end: float=0.02, num_steps:int=1000):
        self.betas = torch.linspace(beta_start, beta_end, num_steps) # defining the beta variables
        self.alphas = 1.0-self.betas # defining the alpha variables
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) # defining the alpha bar variables
        self.num_steps = num_steps
        self.sigma_t = torch.zeros(num_steps)
        for t in range(num_steps):
            self.sigma_t[t] = torch.sqrt(self.betas[t])
    
    def add_noise(self, x: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method receives the input data and the timestep, generates a noise according to the 
        timestep, perturbs the data with the noise, and returns the noisy version of the data and
        the noise itself
        
        Args:
            x (torch.Tensor): input image [B, 1, 28, 28]
            timestep (torch.Tensor): timesteps [B]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: noisy_x [B, 1, 28, 28], noise [B, 1, 28, 28]
        """
        timestep = timestep.long().to('cpu')
        alphas_t = self.alphas[timestep]
        alpha_bars_t = self.alpha_bars[timestep]
        x = x.to('cpu')
        noise = torch.randn_like(x)
        alphas_t = alphas_t.view(-1, 1, 1, 1)
        alpha_bars_t = alpha_bars_t.view(-1, 1, 1, 1)

        noisy_x = torch.sqrt(alpha_bars_t) * x + torch.sqrt(1 - alpha_bars_t) * noise

        return noisy_x, noise

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding


def _make_te(self, dim_in, dim_out):
  return nn.Sequential(
    nn.Linear(dim_in, dim_out),
    nn.SiLU(),
    nn.Linear(dim_out, dim_out)
  )

class NoiseEstimatingNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim: int=100, class_emb_dim: int=100, num_classes: int=10):
        super(NoiseEstimatingNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        self.class_embed = nn.Embedding(num_classes, class_emb_dim)
        # First half
        combined_emb_dim = time_emb_dim + class_emb_dim
        # combined_emb_dim = time_emb_dim
        self.te1 = self._make_te(combined_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(combined_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(combined_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(combined_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(combined_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(combined_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(combined_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t, y):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        c_emb = self.class_embed(y)
        combined_emb = torch.cat([t, c_emb], dim=1)
        # combined_emb = t + c_emb
        n = len(x)
        out1 = self.b1(x + self.te1(combined_emb).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(combined_emb).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(combined_emb).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(combined_emb).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(combined_emb).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(combined_emb).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(combined_emb).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

class DiffusionModel(nn.Module):
    """
    The whole diffusion model put together
    """
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler):
        """

        Args:
            network (nn.Module): your noise estimating network
            var_scheduler (VarianceScheduler): variance scheduler for getting 
                                the statistical variables and the noisy images
        """
        
        super().__init__()
        
        self.network = network
        self.var_scheduler = var_scheduler
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.float32:
        """
        The forward method for the diffusion model gets the input images and 
        their corresponding labels
        
        Args:
            x (torch.Tensor): the input image [B, 1, 28, 28]
            y (torch.Tensor): labels [B]

        Returns:
            torch.float32: the loss between the actual noises and the estimated noise
        """
        
        # step1: sample timesteps
        device=x.device        
        B = x.size(0)
        # Step 1: Sample timesteps uniformly
        timesteps = torch.randint(0, len(self.var_scheduler.betas), (B,), device=x.device)
        # Step 2: Compute the noisy versions of the input image
        noisy_x, noise = self.var_scheduler.add_noise(x, timesteps)
        noisy_x, noise = noisy_x.to(device), noise.to(device)
        # Step 3: Estimate the noises using the noise estimating network
        estimated_noise = self.network(noisy_x, timesteps, y)
        # Step 4: Compute the loss (mean squared error) between the estimated and true noises
        loss = F.mse_loss(estimated_noise, noise)

        return loss
    
    @torch.no_grad()
    def generate_sample(self, num_images: int, y, device) -> torch.Tensor:
      samples = torch.randn(num_images, 1, 28, 28).to(device)
      for t in reversed(range(0, self.var_scheduler.num_steps)):
        noise = torch.randn_like(samples)
        beta_t = self.var_scheduler.betas[t]
        alpha_t = self.var_scheduler.alphas[t]
        alpha_bar_t = self.var_scheduler.alpha_bars[t]
        sigma_t = self.var_scheduler.sigma_t[t]
        epsilon_theta = self.network(samples, torch.full((num_images,), t, device=device, dtype=torch.long), y)
        samples = (samples - (beta_t*epsilon_theta)/torch.sqrt(1.0 - alpha_bar_t))/torch.sqrt(alpha_t)
        if t > 0:
          samples = samples + sigma_t*noise
        
      return samples

def load_diffusion_and_generate():
    device = torch.device('cuda')
    var_scheduler = VarianceScheduler(beta_start=0.0001, beta_end=0.02, num_steps=1000)
    time_emb_dim = 128
    class_emb_dim = 128
    num_classes = 10
    #time_emb_dim=time_emb_dim, class_emb_dim=class_emb_dim
    network = NoiseEstimatingNet(time_emb_dim = time_emb_dim, class_emb_dim = class_emb_dim).to(device)
    diffusion = DiffusionModel(network=network, var_scheduler=var_scheduler) 
    
    # loading the weights of VAE
    diffusion.load_state_dict(torch.load('diffusion.pt'))
    diffusion = diffusion.to(device)
    
    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = diffusion.generate_sample(50, desired_labels, device)
    
    return generated_samples