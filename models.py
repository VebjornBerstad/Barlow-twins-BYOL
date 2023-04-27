import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from torchvision import transforms

import pytorch_lightning as pl

from copy import deepcopy
from typing import Callable, Tuple, Sequence, Union

from kornia import augmentation as aug

import numpy as np
import random
from functools import partial

class MixUpAugmentation(nn.Module):
    def __init__(self, alpha=0.4):
        super(MixUpAugmentation, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)

        index = torch.randperm(batch_size).to(x.device)
        x_exp = torch.exp(x)
        x_exp_k = torch.exp(x[index])

        mixed_x = torch.log((1 - lam) * x_exp + lam * x_exp_k)

        return mixed_x
    
class BatchNormalize(nn.Module):
    def __init__(self, eps=1e-5):
        super(BatchNormalize, self).__init__()
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

def augmentation_pipeline(image_size=(128, 216), mixup_alpha=0.1):
    pipeline = nn.Sequential(
        # transforms.RandomApply([RandomMixUp(alpha=mixup_alpha)], p=1),
        # MixUpAugmentation(alpha=0.01),
        transforms.RandomApply([transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1), antialias=None)], p=1),
        BatchNormalize()
    )
    return pipeline

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
        assert z1.shape == z2.shape
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        # cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size
        cross_corr = torch.matmul(z1_norm.T, z2_norm) / z1.shape[0]

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
    

class ConvNet(nn.Module):
    def __init__(self, in_channels=1, out_features=128):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64*54*32, out_features)
        # self.sigmoid =nn.Sigmoid()
        # self.batch = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.batch(x)
        return x

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
                 tau=0.99
                 ):
        
        super().__init__()
        self.augment = augmentation_pipeline(image_size=image_size, mixup_alpha=0.01)

        self.lr = lr
        self.tau = tau

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim)
        self._target = None

        self.loss = BarlowTwinsLoss(batch_size=64)

    @property
    def target(self):
        if self._target is None:
            target_encoder = deepcopy(self.encoder)
            target_projection_head = deepcopy(self.projection_head)
            self._target = nn.Sequential(target_encoder, target_projection_head)
        return self._target
    
    def forward(self, x):
        return self.encoder(x)
    
    def training_step(self, batch, batch_idx):
        x = batch[0]

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
            target_y = self.target(x2)
        y = self.projection_head(self.encoder(x1))

        loss = self.loss(y, target_y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0]

        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
            target_y = self.target(x2)
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

class RunningLoss(pl.Callback):
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.running_loss = 0.0
        pl_module.loss_count = 0.0
