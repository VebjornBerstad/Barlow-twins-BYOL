import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.filepaths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.png'):
                    self.filepaths.append(os.path.join(class_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

