import numpy as np
import torch
import torch.nn as nn

class MixUpAugmentation(nn.Module):
    def __init__(self, alpha=0.4):
        super(MixUpAugmentation, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        batch_size = x.size(0)

        index = torch.randperm(batch_size).to(x.device)
        x_exp = torch.exp(x)
        x_exp_k = torch.exp(x[index])

        x = self.alpha * x_exp + (1. - self.alpha) * x_exp_k

        return torch.log(x + torch.finfo(x.dtype).eps)
    
class post_normalize(nn.Module):
    def __init__(self, eps=1e-5):
        super(post_normalize, self).__init__()
        self.eps = eps

    def forward(self, x):
        min_val = 0
        max_val = 255
        mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        std = torch.std(x, dim=(0, 2, 3), unbiased=False, keepdim=True) + self.eps
        mean = (mean-min_val)/(max_val-min_val)
        std = std/(max_val-min_val)
        x_normalized = (x - mean) / std
        return x_normalized