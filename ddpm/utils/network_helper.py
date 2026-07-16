import torch
from torch import nn
from inspect import isfunction
from einops.layers.torch import Rearrange
import math
from torchvision.transforms import Compose, Lambda, ToPILImage
import numpy as np


def exists(x):
    """
    判断数值是否为空
    """
    return  x is not None

def default(val, d):
    """
    获取变量默认值
    如果val存在，返回val；反之返回d函数的默认值
    """
    if exists(val):
        return  val
    return d() if isfunction(d) else d


def num_to_groups(num,divisor):
    groups = num // divisor
    remain = num % divisor
    arr = [divisor] * groups
    if remain > 0:
        arr.append(remain)
    return arr

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"), #使用最近邻填充，在长宽上翻倍
        nn.Conv2d(dim, default(dim_out, dim), 3, padding= 1) #消除锯齿，进行平滑和局部特征提取
    )


def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1 = 2, p2 = 2),
        nn.Conv2d(dim*4,default(dim_out, dim), 1)
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])