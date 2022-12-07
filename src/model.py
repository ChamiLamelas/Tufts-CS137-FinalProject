import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tree_dataset import TreeDataset
from vit_pytorch import ViT


def get_device():
    if torch.cuda.is_available():
        print(
            f'Identified CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        return torch.device('cuda')
    else:
        print(f'Did not identify CUDA device')
        return torch.device('cpu')


def digits_model():
    """
    https://github.com/lucidrains/vit-pytorch
    """

    v = ViT(
        image_size=512,
        patch_size=32,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        channels=1
    )

    # apparently its better to note stack sigmoid here, because BCEWithLogitsLoss is more stable
    # for predictions, may want to pass through sigmoid first
    model = v
    return model.to(get_device())


def train(model, learning_rate, epochs, train_dir, batch_size, labels_key):
    device = get_device()

    train_loader = DataLoader(TreeDataset(train_dir), batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # https://stackoverflow.com/a/52859411
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # https://discuss.pytorch.org/t/what-kind-of-loss-is-better-to-use-in-multilabel-classification/32203
    criterion = nn.BCEWithLogitsLoss()

    train_loss = list()

    for i in range(epochs):
        model.train()

        running_train_loss = 0
        nbatches = 0

        for data in train_loader:
            img = data['image'].to(device)
            labels = data[labels_key].to(device)
            optimizer.zero_grad()
            outputs = model(img)
            print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.cpu().item()
            nbatches += 1

        train_loss.append(running_train_loss/nbatches)
        print(f'Epoch {i+1} loss: {train_loss}')
