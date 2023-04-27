from models import barlowBYOL, LinearEvaluationCallback, ConvNet, RunningLoss
from dataset import CustomImageDataset

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import torch.nn as nn

from torchvision.models import resnet18

from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt


def main():

    mean = 0.4403
    std = 0.1809
    transform = transforms.Compose([
        # transforms.Resize((216, 128)),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
        ])
    
    root_dir = './mel_spectrogram'
    batch_size = 64

    # Create the custom dataset
    dataset = CustomImageDataset(root_dir, transform=transform)

    # Split the dataset into train and validation sets (optional)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

    # Create DataLoaders for the train and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    # encoder = resnet18()
    # encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    # encoder.fc = nn.Identity()

    encoder = ConvNet(in_channels=1, out_features=512)

    logger = TensorBoardLogger("logs", name="Barlow_BYOL")
    
    barlow_byol = barlowBYOL(encoder=encoder, image_size=(128, 216), lr=1e-4, tau=0.99, encoder_out_dim=512)

    linear_evaluation = LinearEvaluationCallback(encoder_output_dim=512, num_classes=10)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=50, save_top_k=-1, save_last=True)

    barlow_byol_trainer = Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=500,
        callbacks=[linear_evaluation, checkpoint_callback],
        logger=logger,  
        # log_every_n_steps=10
        )
    barlow_byol_trainer.fit(barlow_byol, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()