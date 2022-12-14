import torch
import torch.nn as nn
from vit_pytorch import ViT
from graphs import Graph, prims
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
# from torchvision.models import ResNet18_Weights

# https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html
# https://pytorch.org/vision/master/models.html
# https://discuss.pytorch.org/t/how-to-access-latest-torchvision-models-e-g-vit/145880

def get_device():
    if torch.cuda.is_available():
        print(
            f'Identified CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        return torch.device('cuda')
    else:
        print(f'Did not identify CUDA device')
        return torch.device('cpu')


def nice_time_print(s):
    s = int(s)
    min = int(s/60)
    sec = s % 60
    print(f'{min}m {sec}s')

def untrained_digit_model():
    v = ViT(
        image_size=512,
        patch_size=32,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    return v


def untrained_tree_model():
    v = ViT(
        image_size=512,
        patch_size=32,
        num_classes=45,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    return v


def resnet_preprocess():
    return transforms.Compose([
        # transforms.RandomRotation(30),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])


# def pretrained_resnet_model():
#     return torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)


# def make_resnet_model(pretrained):
#     # https://discuss.pytorch.org/t/how-to-modify-the-final-fc-layer-based-on-the-torch-model/766/25?page=2
#     # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#     pretrained.fc = nn.Linear(512, 10)


# def tuned_resnet_model():
#     return torch.load(os.path.join('..', 'models', 'resnet', 'digit-model.pt'))


def digits_predict(model, img):
    sigmoid = nn.Sigmoid()
    outputs = sigmoid(model(img))
    return (outputs >= 0.5).long()


def batchcorrect(outputs, labels, names, show_failing):
    equality = torch.all(outputs.cpu() == labels.cpu(), dim=1)
    if show_failing:
        out = [f'{n}: {output}' for i, (n, output) in enumerate(zip(names, outputs)) if not equality[i]]
        if len(out) > 0:
            print("\n".join(out))
    return torch.sum(equality).item()


def tree_predict(model, img_batch, digits_model):
    tree_outputs = nn.Sigmoid()(model(img_batch))
    digit_outputs = digits_predict(digits_model, img_batch)
    outputs = list()
    for tree_output, digit_output in zip(tree_outputs, digit_outputs):
        g = Graph(digit_output, tree_output)
        outputs.append(prims(g))
    return torch.from_numpy(np.array(outputs))


def predict(model, data_loader, device, config, digits_model, show_failing=False, use_prims=True):
    model.eval()

    with torch.no_grad():

        ncorrect = 0
        total = 0

        for data in data_loader:
            img = data['image'].to(device)
            labels = data[config['labels_key']].to(device)
            names = data['img_name']
            if digits_model is None:
                outputs = digits_predict(model, img)
            else:
                outputs = tree_predict(
                    model, img, digits_model) if use_prims else digits_predict(model, img)
            ncorrect += batchcorrect(outputs, labels, names, show_failing)
            total += img.size()[0]

    return ncorrect / total

def set_torch_vit_dropouts(model, dropout):
    model.encoder.dropout = nn.Dropout(p=dropout, inplace=False)
    for i in range(12):
        exec(f'model.encoder.layers.encoder_layer_{i}.dropout=nn.Dropout(p={dropout},inplace=False)')
        exec(f'model.encoder.layers.encoder_layer_{i}.mlp[2]=nn.Dropout(p={dropout},inplace=False)')
        exec(f'model.encoder.layers.encoder_layer_{i}.mlp[4]=nn.Dropout(p={dropout},inplace=False)')

def float_eqs(f1, f2):
    return abs(f1 - f2) < (10 ** (-6))


def train(model, learning_rate, weight_decay, epochs, train_loader, val_loader,
          device, model_dir, digits_model, show_failing=False, use_prims=True):
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    if digits_model is None:
        config = {'labels_key': 'digit_labels', 'model_file': 'digit-model.pt',
                  'train_loss': 'digit_train_loss.npy', 'val_acc': 'digit_val_acc.npy', 'final_model_file': 'final-digit-model.pt'}
    else:
        config = {'labels_key': 'tree_label', 'model_file': 'tree-model.pt',
                  'train_loss': 'tree_train_loss.npy', 'val_acc': 'tree_val_acc.npy', 'final_model_file': 'final-tree-model.pt'}

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # https://stackoverflow.com/a/52859411
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    # https://discuss.pytorch.org/t/what-kind-of-loss-is-better-to-use-in-multilabel-classification/32203
    criterion = nn.BCEWithLogitsLoss()

    train_loss = list()
    val_acc = list()

    min_train_loss = None
    max_val_acc = None

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

        info_epoch = (i + 1) % 10 == 0 and i > 0

        curr_train_loss = running_train_loss/nbatches
        curr_val_acc = predict(model, val_loader, device, config,
                               digits_model, info_epoch and show_failing, use_prims)

        # save 1st epoch model, if model has better val acc, or has equal val acc and less train loss
        if max_val_acc is None or curr_val_acc > max_val_acc or (float_eqs(curr_val_acc, max_val_acc) and curr_train_loss < min_train_loss):
            torch.save(model, os.path.join(model_dir, config['model_file']))

        # update tracked metrics
        if max_val_acc is None or curr_val_acc > max_val_acc:
            max_val_acc = curr_val_acc
        if min_train_loss is None or curr_train_loss < min_train_loss:
            min_train_loss = curr_train_loss

        train_loss.append(curr_train_loss)
        val_acc.append(curr_val_acc)

        if info_epoch:
            print(
                f'Epoch {i+1} done, train loss: {train_loss[-1]:.4f} val acc: {curr_val_acc:.4f}')

    np.save(os.path.join(
        model_dir, config['train_loss']), np.array(train_loss))
    np.save(os.path.join(model_dir, config['val_acc']), np.array(val_acc))
    torch.save(model, os.path.join(model_dir, config['final_model_file']))
