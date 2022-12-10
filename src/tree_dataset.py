from torch.utils.data import Dataset
import gen_dataset as gd
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class TreeDataset(Dataset):
    def __init__(self, directory, preprocess):
        self.directory = directory
        self.num_elements = int(len(os.listdir(directory))/4)
        self.preprocess = preprocess

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        image_name = f'{gd.DATANAME_PREFIX}{idx}'
        image = Image.open(os.path.join(
            self.directory, f'{image_name}{gd.IMAGE_SUFFIX}')).convert('RGB')
        # https://numpy.org/doc/stable/reference/generated/numpy.load.html
        tree_label = np.load(os.path.join(
            self.directory, f'{image_name}{gd.TREE_LABEL_SUFFIX}'))
        digit_labels = np.load(os.path.join(
            self.directory, f'{image_name}{gd.DIGIT_LABELS_SUFFIX}'))
        return {
            'img_name': image_name,
            'image': self.preprocess(image),
            # https://pytorch.org/docs/stable/generated/torch.from_numpy.html
            'tree_label': torch.from_numpy(tree_label),
            'digit_labels': torch.from_numpy(digit_labels)
        }


class NNodeTreeDataset(Dataset):
    def __init__(self, directory, preprocess, n):
        self.directory = directory
        self.nnode_names = list()
        for entry in os.scandir(directory):
            if entry.name.endswith(gd.DIGIT_LABELS_SUFFIX):
                digit_labels = np.load(
                    os.path.join(self.directory, entry.name))
                if np.sum(digit_labels) == n:
                    self.nnode_names.append(
                        entry.name[:-len(gd.DIGIT_LABELS_SUFFIX)])
        self.preprocess = preprocess

    def __len__(self):
        return len(self.nnode_names)

    def __getitem__(self, idx):
        image_name = self.nnode_names[idx]
        image = Image.open(os.path.join(
            self.directory, f'{image_name}{gd.IMAGE_SUFFIX}')).convert('RGB')
        # https://numpy.org/doc/stable/reference/generated/numpy.load.html
        tree_label = np.load(os.path.join(
            self.directory, f'{image_name}{gd.TREE_LABEL_SUFFIX}'))
        digit_labels = np.load(os.path.join(
            self.directory, f'{image_name}{gd.DIGIT_LABELS_SUFFIX}'))
        return {
            'img_name': image_name,
            'image': self.preprocess(image),
            # https://pytorch.org/docs/stable/generated/torch.from_numpy.html
            'tree_label': torch.from_numpy(tree_label),
            'digit_labels': torch.from_numpy(digit_labels)
        }


def show_img(tensor):
    image = transforms.ToPILImage()(tensor)
    image.show()


if __name__ == '__main__':
    dataset = TreeDataset(os.path.join(
        '..', 'data', 'newtest2'), preprocess=transforms.ToTensor())
    print(dataset[0])

    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))
    print(batch['img_name'])

    # there are 4 ten node trees in trainset 2
    ten_dataset = NNodeTreeDataset(os.path.join(
        '..', 'data', 'trainset2'), preprocess=transforms.ToTensor(), n=10)
    print(len(ten_dataset))

    dataloader = DataLoader(ten_dataset, batch_size=1)
    for data in dataloader:
        print(data['img_name'])
