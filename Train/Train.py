import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Lenet import LeNet5
from gb688Dataset import gb688Dataset
from os import getcwd
from copy import deepcopy

normal_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914), (0.2023))
])

augmentation_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomRotation(5),
    # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.Normalize((0.4914), (0.2023))
])
# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 31
LEARNING_RATE = 0.002
BATCH_SIZE = 25
N_EPOCHS = 30
TEST_PERCENTAGE = 0.5
APPLY_AUGMENTATION = True

# Model saving path
PATH = "/Users/chenziyang/Documents/Ziyang/Crawler/中石化.nosync/验证码/CAPTCHA_Validation/checkpoint"
model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

normal_dataset = gb688Dataset("../Data", transform=normal_transforms)
augmentation_dataset = gb688Dataset("../Data", transform=augmentation_transforms)
train_dataset, test_dataset = random_split(normal_dataset,
                                           [int(len(normal_dataset) * (1 - TEST_PERCENTAGE)), int(len(normal_dataset) *
                                            TEST_PERCENTAGE)], generator=torch.Generator().manual_seed(42))
train_dataset = ConcatDataset([train_dataset, augmentation_dataset])
train_loader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE)


def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''

    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)
            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.show()

    # change the plot style to default
    plt.style.use('default')


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)


model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, test_loader, N_EPOCHS, DEVICE)
# torch.save(model.state_dict(), PATH+"/gb688"+".pth")
