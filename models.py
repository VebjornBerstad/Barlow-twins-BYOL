import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as T


from copy import deepcopy
from typing import Callable, Tuple

from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf

import random

def set_requires_grad(model: nn.Module, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = requires_grad

class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if random.random() > self.p else self.fn(x)

def default_augmentation(image_size: Tuple[int, int] = (32, 32)) -> nn.Module:
    return nn.Sequential(
        aug.RandomHorizontalFlip(),
        aug.RandomCrop(size=image_size, padding=4, padding_mode='reflect'),
        aug.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        aug.RandomGrayscale(p=0.1),
        aug.Normalize(
            mean=torch.tensor([0.4914, 0.4822, 0.4465]),
            std=torch.tensor([0.2470, 0.2435, 0.2616]),
        ),
    )

def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )

class SmallConvNet(nn.Module):
    def __init__(self, output_dim):
        super(SmallConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(32 * 8 * 8, output_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class barlowBYOL(pl.LightningModule):
    def __init__(self, 
                 encoder, 
                 projector, 
                 image_size: Tuple[int, int], 
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
        x = self.online[0](x)  # Pass input through the ResNet-18 encoder
        x = x.view(x.size(0), -1)  # Flatten the output tensor
        x = self.online[1](x)  # Pass the flattened tensor through the projector
        return x

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
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
    
    def training_step(self, batch, batch_idx):
        x = batch[0]

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        y1, y2 = self.forward(x1), self.forward(x2)

        with torch.no_grad():
            target_y1, target_y2 = self.target(x1), self.target(x2)

        loss = self.cross_corr_loss(y1, y2, target_y1, target_y2)
        
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        y1, y2 = self.forward(x1), self.forward(x2)

        with torch.no_grad():
            target_y1, target_y2 = self.target(x1), self.target(x2)

        loss = self.cross_corr_loss(y1, y2, target_y1, target_y2)

        self.log('val_loss', loss)

        return loss
    
    def on_after_backward(self):
        for online_param, target_param in zip(self.online.parameters(), self.target.parameters()):
            target_param.data = target_param.data * self.tau + online_param.data * (1 - self.tau)

class LinearEvalModel(pl.LightningModule):
    def __init__(self, encoder, num_classes, lr=3e-4):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(512, num_classes)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        set_requires_grad(self.encoder, False)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss