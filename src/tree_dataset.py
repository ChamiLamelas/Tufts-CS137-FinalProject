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
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        image_name = f'{gd.DATANAME_PREFIX}{idx}'
        image = Image.open(os.path.join(
            self.directory, f'{image_name}{gd.IMAGE_SUFFIX}'))
        # https://numpy.org/doc/stable/reference/generated/numpy.load.html
        tree_label = np.load(os.path.join(
            self.directory, f'{image_name}{gd.TREE_LABEL_SUFFIX}'))
        digit_labels = np.load(os.path.join(
            self.directory, f'{image_name}{gd.DIGIT_LABELS_SUFFIX}'))
        return {
            'image_name': image_name,
            'image': self.transforms(image), 
            # https://pytorch.org/docs/stable/generated/torch.from_numpy.html
            'tree_label': torch.from_numpy(tree_label), 
            'digit_labels': torch.from_numpy(digit_labels)
        }

if __name__ == '__main__':
    dataset = TreeDataset(os.path.join('..', 'data', 'newtest2'))
    print(dataset[0])
