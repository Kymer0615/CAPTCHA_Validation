import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt


class LeNet5(nn.Module):

    def __init__(self, n_classes, param):
        super(LeNet5, self).__init__()
        self.width = param[0]
        self.height = param[1]
        self.channel_num = param[2]
        self._to_linear = None
        # Test variable to get the ouput size of the hidden layers

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.channel_num, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )
        if self._to_linear is None:
            y = torch.randn(self.width * 3, self.height * 3).view(-1, self.channel_num, self.width, self.height)
            y = self.feature_extractor(y)
            self._to_linear = y[0].shape[0] * y[0].shape[1] * y[0].shape[2]
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self._to_linear, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        # print(torch.flatten(x, 1).shape)
        # print(self._to_linear)
        # x = x.view(-1, 225)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
