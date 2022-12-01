from torch.utils.data import Dataset
import gen_dataset as gd
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch


class TreeDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.num_elements = int(len(os.listdir(directory))/4)
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        image = Image.open(os.path.join(
            self.directory, f'{gd.DATANAME_PREFIX}{idx}{gd.IMAGE_SUFFIX}'))
        image = image.convert('RGB')
        tree_label = np.fromfile(os.path.join(
            self.directory, f'{gd.DATANAME_PREFIX}{idx}{gd.TREE_LABEL_SUFFIX}'))
        digit_labels = np.fromfile(os.path.join(
            self.directory, f'{gd.DATANAME_PREFIX}{idx}{gd.DIGIT_LABELS_SUFFIX}'))
        return {
            'image': torch.squeeze(self.transforms(image)), 
            'tree_label': tree_label, 
            'digit_labels': digit_labels
        }
