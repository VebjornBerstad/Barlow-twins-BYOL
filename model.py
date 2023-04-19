import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from copy import deepcopy
from typing import Callable, Tuple

from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf

import random

class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if random.random() > self.p else self.fn(x)


def default_augmentation(image_size: Tuple[int, int] = (224, 224)) -> nn.Module:
    return nn.Sequential(
        tf.Resize(size=image_size),
        RandomApply(aug.ColorJitter(0, 0, 0, 0), p=0),
        aug.RandomGrayscale(p=0.2),
        aug.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((0, 0), (0, 0)), p=0),
        aug.RandomResizedCrop(size=image_size),
        aug.Normalize(
            mean=torch.tensor([0, 0, 0]),
            std=torch.tensor([0, 0, 0]),
        ),
    )

class barlowBYOL(pl.LightningModule):
    def __init__(self, 
                 encoder, 
                 projector, 
                 image_size, 
                 lr=3e-4, 
                 tau=0.99
                 ):
        
        super().__init__()
        self.augment = default_augmentation(image_size)

        self.lr = lr
        self.tau = tau

        self.online = nn.Sequential(encoder, projector)
        self._target = None

        # Create and initialize the projector with a dummy tensor
        self.online(torch.zeros(2,3, *image_size))

    @property
    def target(self):
        if self._target is None:
            self._target = deepcopy(self.online)
        return self._target
    
    def forward(self, x):
        return self.online(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x = batch[0]

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        y1, y2 = self.forward(x1), self.forward(x2)

        with torch.no_grad():
            target_y1, target_y2 = self.target(x1), self.target(x2)

        loss = cross_corr_loss(y1, y2, target_y1, target_y2)
        
        self.log('train_loss', loss)

        return loss

    def cross_corr_loss(self, y1, y2, target_y1, target_y2):
        batch_size = y1.size(0)
        y1_norm = F.normalize(y1, dim=1)
        y2_norm = F.normalize(y2, dim=1)
        target_y1_norm = F.normalize(target_y1, dim=1)
        target_y2_norm = F.normalize(target_y2, dim=1)

        c = torch.matmul(y1_norm, y2_norm.T) / batch_size
        c_target = torch.matmul(target_y1_norm, target_y2_norm.T) / batch_size
        c_diff = c - c_target.detach()
        loss = torch.sum(c_diff ** 2) / (batch_size ** 2)

        return loss
    
    def on_after_backward(self):
        for online_param, target_param in zip(self.online.parameters(), self.target.parameters()):
            target_param.data = target_param.data * self.tau + online_param.data * (1 - self.tau)