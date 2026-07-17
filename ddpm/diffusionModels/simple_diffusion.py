from utils.network_helper import *
from diffusionModels.simple_diffusion import VarianceSchedule
from torch.nn import functional as F

class DiffusionModel(nn.Module):
    def __init__(self,
                 schedule_name = "linear_beta_schedule",
                 timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None):
        super().__init__()
        self.denoise_model =denoise_model

        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name,beta_start=beta_start,beta_start=beta_start)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)

        #define alpha 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev =  F.pad(self.alphas_cumprod[:-1],(1,0),value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculateions for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for inverse
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    
    def q_sample(self,x_start, t, noise=None):
        """前向加噪"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t =  extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    

    def compute_loss(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.rand_like(x_start)

        x_noisy = self.q_sample(x_start,t,noise)
        predicted_noise = self.denoise_model(x_noisy, t)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        return loss