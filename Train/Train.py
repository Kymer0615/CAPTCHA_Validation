import json
import logging
from os import getcwd

import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from CAPTCHA_Validation.Train.Lenet import LeNet5
from CAPTCHA_Validation.Train.CapitalDataset import CapitalDataset


class Train:
    def __init__(self, name):
        self.name = name
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.jsonPath = getcwd() + "/Configs/" + name + ".json"
        self.normal_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914), (0.2023))
        ])
        self.criterion = nn.CrossEntropyLoss()
        self.checkpointPath = getcwd() + "/Checkpoints"
        self.augmentation_transforms = None
        self.LEARNING_RATE = None
        self.BATCH_SIZE = None
        self.N_EPOCHS = None
        self.TEST_PERCENTAGE = None
        self.print_every = None
        self.getParamFromJson()
        self.normal_dataset = CapitalDataset(getcwd() + "/Data/%s" % self.name, transform=self.normal_transforms)
        self.N_CLASSES = self.normal_dataset.getLabelNum()
        self.model = LeNet5(self.N_CLASSES).to(self.DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def getDataLoader(self):
        train_dataset, test_dataset = random_split(self.normal_dataset,
                                                   [int(len(self.normal_dataset) * (1 - self.TEST_PERCENTAGE)),
                                                    int(len(self.normal_dataset) *
                                                        self.TEST_PERCENTAGE)],
                                                   generator=torch.Generator().manual_seed(42))
        if self.augmentation_transforms:
            augmentation_dataset = CapitalDataset(getcwd() + "/Data/%s" % self.name,
                                                  transform=self.augmentation_transforms)
            train_dataset = ConcatDataset([train_dataset, augmentation_dataset])

        trainLoader = torch.utils.data.DataLoader(train_dataset, self.BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, self.BATCH_SIZE)
        return trainLoader, test_loader

    def setParam(self, param, arg):
        execStr = "self.%s=%s" % (param, arg)
        logging.info("Parameter info: %s" % execStr)
        exec(execStr)

    def getParamFromJson(self):
        with open(self.jsonPath) as f:
            configs = json.load(f)["Train"]
        for param in configs.keys():
            if param == "augmentation_transforms":
                execStr = "self.augmentation_transforms=transforms.Compose(["
                execStr = execStr + "".join(["transforms." + j + "," for j in configs[param]['args']][:-1]) + "])"
                logging.info("Parameter info: %s" % execStr)
                exec(execStr)
            else:
                for arg in configs[param]['args']: self.setParam(param, arg)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def train(trainLoader, model, criterion, optimizer, device):
        '''
        Function for the training step of the training loop
        '''

        model.train()
        running_loss = 0

        for X, y_true in trainLoader:
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

        epoch_loss = running_loss / len(trainLoader.dataset)
        return model, optimizer, epoch_loss

    @staticmethod
    def validate(testLoader, model, criterion, device):
        '''
        Function for the validation step of the training loop
        '''

        model.eval()
        running_loss = 0

        for X, y_true in testLoader:
            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass and record loss
            y_hat, _ = model(X)
            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(testLoader.dataset)

        return model, epoch_loss

    def training_loop(self):
        '''
        Function defining the entire training loop
        '''

        # set objects for storing metrics
        best_loss = 1e10
        train_losses = []
        valid_losses = []

        trainLoader, testLoader = self.getDataLoader()
        # Train model
        for epoch in range(0, self.N_EPOCHS):

            # training
            self.model, self.optimizer, train_loss = Train.train(trainLoader, self.model, self.criterion,
                                                                 self.optimizer, self.DEVICE)
            train_losses.append(train_loss)

            # validation
            with torch.no_grad():
                self.model, valid_loss = Train.validate(testLoader, self.model, self.criterion, self.DEVICE)
                valid_losses.append(valid_loss)

            if epoch % self.print_every == (self.print_every - 1):
                train_acc = Train.get_accuracy(self.model, trainLoader, device=self.DEVICE)
                valid_acc = Train.get_accuracy(self.model, testLoader, device=self.DEVICE)
                print(f'{datetime.now().time().replace(microsecond=0)} --- '
                      f'Epoch: {epoch}\t'
                      f'Train loss: {train_loss:.4f}\t'
                      f'Valid loss: {valid_loss:.4f}\t'
                      f'Train accuracy: {100 * train_acc:.2f}\t'
                      f'Valid accuracy: {100 * valid_acc:.2f}')

        Train.plot_losses(train_losses, valid_losses)

        return self.model, self.optimizer, (train_losses, valid_losses)

    def saveModel(self):
        torch.save(self.model.state_dict(), self.checkpointPath + "/" + self.name + ".pth")
