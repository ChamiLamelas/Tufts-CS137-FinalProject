import torch
import torch.nn as nn
from vit_pytorch import ViT
from graphs import Graph, prims
import os
import numpy as np


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

    # apparently its better to not stack sigmoid here, because BCEWithLogitsLoss is more stable
    # for predictions, may want to pass through sigmoid first
    model = v
    return model

def tree_model():
    """
    https://github.com/lucidrains/vit-pytorch
    """

    v = ViT(
        image_size=512,
        patch_size=32,
        num_classes=45,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        channels=1
    )

    # apparently its better to not stack sigmoid here, because BCEWithLogitsLoss is more stable
    # for predictions, may want to pass through sigmoid first
    model = v
    return model


def digits_predict(model, img):
    sigmoid = nn.Sigmoid()
    outputs = sigmoid(model(img))
    return (outputs >= 0.5).long()


def iscorrect(outputs, labels):
    return torch.all(outputs.cpu() == labels.cpu()).item()


def tree_predict(model, img, digits_model):
    tree_outputs = digits_predict(model, img)
    digit_outputs = digits_predict(digits_model, img)
    g = Graph(digit_outputs, tree_outputs)
    return torch.from_numpy(prims(g))


def predict(model, data_loader, device, config, digits_model):
    model.eval()

    with torch.no_grad():

        ncorrect = 0
        total = 0

        for data in data_loader:
            img = data['image'].to(device)
            labels = data[config['labels_key']].to(device)
            if digits_model is None:
                outputs = digits_predict(model, img)
            else:
                outputs = tree_predict(model, img, digits_model)
            ncorrect += iscorrect(outputs, labels)
            total += 1

    return ncorrect / total


def train(model, learning_rate, epochs, train_loader, val_loader, device, model_dir, digits_model):
    if digits_model is None:
        config = {'labels_key': 'digit_labels', 'model_file': 'digit-model.pt',
                  'train_loss': 'digit_train_loss.npy', 'val_acc': 'digit_val_acc.npy'}
    else:
        config = {'labels_key': 'tree_label', 'model_file': 'tree-model.pt',
                  'train_loss': 'tree_train_loss.npy', 'val_acc': 'tree_val_acc.npy'}

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # https://stackoverflow.com/a/52859411
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # https://discuss.pytorch.org/t/what-kind-of-loss-is-better-to-use-in-multilabel-classification/32203
    criterion = nn.BCEWithLogitsLoss()

    train_loss = list()
    val_acc = list()

    min_val_acc = None

    for i in range(epochs):
        model.train()

        running_train_loss = 0
        nbatches = 0

        for data in train_loader:
            img = data['image'].to(device)
            labels = data[config['labels_key']].to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels.to(torch.float32))
            loss.backward()
            optimizer.step()
            running_train_loss += loss.cpu().item()
            nbatches += 1

        curr_val_acc = predict(model, val_loader, device, config, digits_model)
        if min_val_acc is None or curr_val_acc < min_val_acc:
            torch.save(model, os.path.join(model_dir, config['model_file']))

        train_loss.append(running_train_loss/nbatches)
        val_acc.append(curr_val_acc)

        print(
            f'Epoch {i+1} done, train loss: {train_loss[-1]:.4f} val acc: {curr_val_acc:.4f}')
    
    np.save(os.path.join(model_dir, config['train_loss']), np.array(train_loss))
    np.save(os.path.join(model_dir, config['val_acc']), np.array(val_acc))
