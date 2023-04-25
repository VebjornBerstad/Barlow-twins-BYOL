from models import barlowBYOL, LinearEvaluationCallback
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

    transform = transforms.Compose([
        # transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])
    
    root_dir = './mel_spectrogram'
    batch_size = 16

    # Create the custom dataset
    dataset = CustomImageDataset(root_dir, transform=transform)

    # Split the dataset into train and validation sets (optional)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, valid_size])

    # Create DataLoaders for the train and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load the CIFAR-10 dataset
    # train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    encoder = resnet18()
    encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    encoder.fc = nn.Identity()

    # Load the barlowBYOL model and train it

    logger = TensorBoardLogger("logs", name="Barlow_BYOL")
    
    barlow_byol = barlowBYOL(encoder=encoder, image_size=(216, 128), lr=3e-4, tau=0.99, encoder_out_dim=512)

    linear_evaluation = LinearEvaluationCallback(encoder_output_dim=512, num_classes=10)
    checkpoint_callback = ModelCheckpoint(every_n_epochs=100, save_top_k=-1, save_last=True)

    barlow_byol_trainer = Trainer(
        devices=1,
        accelerator='gpu',
        max_epochs=500,
        callbacks=[linear_evaluation, checkpoint_callback],
        logger=logger,
        log_every_n_steps=1
        )
    barlow_byol_trainer.fit(barlow_byol, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()