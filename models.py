import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import pytorch_lightning as pl

from copy import deepcopy
from typing import Callable, Tuple, Sequence, Union

from kornia import augmentation as aug

import random

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
    )

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)

class barlowBYOL(pl.LightningModule):
    def __init__(self, 
                 encoder, 
                 encoder_out_dim, 
                 image_size: Tuple[int, int], 
                 lr=3e-4, 
                 tau=0.99,
                 ):
        
        super().__init__()
        self.augment = default_augmentation(image_size)

        self.lr = lr
        self.tau = tau

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim)
        self._target = None

        self.loss = BarlowTwinsLoss(batch_size=256)

        # Create and initialize the projector with a dummy tensor
        # self.online(torch.zeros(2,3, *image_size))

    @property
    def target(self):
        if self._target is None:
            target_encoder = deepcopy(self.encoder)
            target_projection_head = deepcopy(self.projection_head)
        return nn.Sequential(target_encoder, target_projection_head)
    
    def forward(self, x):
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        x = batch[0]

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
            target_y = self.projection_head(self.encoder(x2))
        y = self.projection_head(self.encoder(x1))

        loss = self.loss(y, target_y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        print(f'\nTrain loss: {loss}')
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
            target_y = self.projection_head(self.encoder(x2))
        y = self.projection_head(self.encoder(x1))

        loss = self.loss(y, target_y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_after_backward(self):
        for online_param, target_param in zip(self.encoder.parameters(), self.target[0].parameters()):
            target_param.data = target_param.data * self.tau + online_param.data * (1 - self.tau)
        for online_param, target_param in zip(self.projection_head.parameters(), self.target[1].parameters()):
            target_param.data = target_param.data * self.tau + online_param.data * (1 - self.tau)

class LinearEvaluationCallback(pl.Callback):
    def __init__(
            self,
            encoder_output_dim: int,
            num_classes: int
            ):
        super().__init__()
        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.linear_classifier = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.linear_classifier.parameters(), lr=1e-4)

    def extract_batch(self, batch: Sequence, device: Union[str, torch.device]):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        return x, y
    
    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int
    ):
        x, y = self.extract_batch(batch, pl_module.device)

        with torch.no_grad():
            features = pl_module.forward(x)
        
        features = features.detach()
        preds = pl_module.linear_classifier(features)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pred_labels = torch.argmax(preds, dim=1)
        acc = accuracy(pred_labels, y, task="multiclass", num_classes=10)
        pl_module.log("online_train_acc", acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int
    ):
        x, y = self.extract_batch(batch, pl_module.device)

        with torch.no_grad():
            features = pl_module.forward(x)
        
        features = features.detach()
        preds = pl_module.linear_classifier(features)
        loss = F.cross_entropy(preds, y)
        
        pred_labels = torch.argmax(preds, dim=1)
        acc = accuracy(pred_labels, y, task="multiclass", num_classes=10)
        pl_module.log("online_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
