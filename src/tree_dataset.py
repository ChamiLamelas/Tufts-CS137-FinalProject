from torch.utils.data import Dataset
import gen_dataset as gd
import os
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import torch 

class TreeDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.num_elements = int(len(os.listdir(directory))/4)
        self.tensorconverter = ToTensor()

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        image = Image.open(os.path.join(
            self.directory, f'{gd.DATANAME_PREFIX}{idx}{gd.IMAGE_SUFFIX}'))
        # https://numpy.org/doc/stable/reference/generated/numpy.load.html
        tree_label = np.load(os.path.join(
            self.directory, f'{gd.DATANAME_PREFIX}{idx}{gd.TREE_LABEL_SUFFIX}'))
        digit_labels = np.load(os.path.join(
            self.directory, f'{gd.DATANAME_PREFIX}{idx}{gd.DIGIT_LABELS_SUFFIX}'))
        return {
            'image': self.tensorconverter(image), 
            # https://pytorch.org/docs/stable/generated/torch.from_numpy.html
            'tree_label': torch.from_numpy(tree_label), 
            'digit_labels': torch.from_numpy(digit_labels)
        }

if __name__ == '__main__':
    dataset = TreeDataset(os.path.join('..', 'data', 'newtest2'))
    print(dataset[0])
