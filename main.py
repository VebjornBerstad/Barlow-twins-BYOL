from models import barlowBYOL, mlp, SmallConvNet, LinearEvalModel

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

import torch.nn as nn
from torch.nn import Flatten

from torchvision.models import resnet18

from pytorch_lightning.loggers import TensorBoardLogger


def main():

    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Load the CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Load the pre-trained SmallConvNet encoder
    # pretrained_encoder = SmallConvNet(512)

    def resnet18_encoder(pretrained=True):
        resnet = resnet18(pretrained=pretrained)
        encoder = nn.Sequential(
            *list(resnet.children())[:-2],  # Remove the avgpool and fc layers
            nn.AdaptiveAvgPool2d((1, 1)),   # Add adaptive average pooling to reduce the spatial dimensions to 1x1
            Flatten(),                      # Add a Flatten layer to convert the output tensor to a 1D tensor
        )
        return encoder

    # Load the barlowBYOL model and train it
    barlow_byol = barlowBYOL(encoder=resnet18_encoder(), projector=mlp(512), image_size=(32, 32), lr=3e-4, tau=0.99)
    barlow_byol_trainer = Trainer(accelerator="gpu", max_epochs=10)
    barlow_byol_trainer.fit(barlow_byol, train_dataloader)

    # Remove the projector from the pre-trained encoder
    pretrained_encoder = barlow_byol.online[0]

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    val_dataset = CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Perform linear evaluation
    tb_logger = TensorBoardLogger("logs/")

    linear_eval_model = LinearEvalModel(encoder=pretrained_encoder, num_classes=10, lr=3e-4)
    linear_eval_trainer = Trainer(logger=tb_logger, accelerator="gpu", max_epochs=10)
    linear_eval_trainer.fit(linear_eval_model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    main()
