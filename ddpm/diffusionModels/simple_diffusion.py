from utils.network_helper import *
from diffusionModels.simple_diffusion import VarianceSchedule
from torch.nn import functional as F
import tqdm

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

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        反向去噪
        """
        betas_t = extract(self.betas,t,x.shape)
        sqrt_one_minus_alphas_cumprod_t =  extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = (x - self.denoise_model(x, t) * betas_t / sqrt_one_minus_alphas_cumprod_t) * sqrt_recip_alphas_t

        if t_index == 0:
            return model_mean
        
        else:
            posterior_variance_t = extract(self.posterior_variance,t,x.shape)
            noise = torch.rand_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device
        batch = shape[0]

        img = torch.randn(shape,device=device)
        imgs = []
        for i in tqdm(reversed(range(0,self.timesteps)), desc="sampling loop time step", total=self.timesteps):
            img = self.p_sample(img, torch.full((batch,),i,device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        #return self.p_sample_loop
        pass
    