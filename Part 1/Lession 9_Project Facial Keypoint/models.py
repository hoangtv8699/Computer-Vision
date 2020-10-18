import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# define a neural network with a single convolutional layer with four filters
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.maxpool2d = nn.MaxPool2d(2, 2)
        # conv layer
        self.conv2d1 = nn.Conv2d(1, 32, 4)
        self.drop1 = nn.Dropout(p=0.1)

        self.conv2d2 = nn.Conv2d(32, 64, 3)
        self.drop2 = nn.Dropout(p=0.2)

        self.conv2d3 = nn.Conv2d(64, 128, 2)
        self.drop3 = nn.Dropout(p=0.3)

        self.conv2d4 = nn.Conv2d(128, 256, 1)
        self.drop4 = nn.Dropout(p=0.4)
        # dense layer
        self.dense1 = nn.Linear(256 * 5 * 5, 1000)
        self.drop5 = nn.Dropout(p=0.5)

        self.dense2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout(p=0.6)

        self.dense3 = nn.Linear(1000, 136)

    def forward(self, x):
        # (1, 96, 96) -> (32, 93, 93) -> (32, 46, 46)
        x = self.maxpool2d(F.relu(self.conv2d1(x)))
        x = self.drop1(x)
        # (32, 46, 46) -> (64, 44, 44) -> (64, 22, 22)
        x = self.maxpool2d(F.relu(self.conv2d2(x)))
        x = self.drop2(x)
        # (64, 22, 22) -> (128, 21, 21) -> (128, 10, 10)
        x = self.maxpool2d(F.relu(self.conv2d3(x)))
        x = self.drop3(x)
        # (128, 10, 10) -> (256, 10, 10) -> (256, 5, 5)
        x = self.maxpool2d(F.relu(self.conv2d4(x)))
        x = self.drop4(x)
        # (256, 5, 5) -> (256*5*5)
        x = x.view(x.size(0), -1)
        # 6400 -> 1000
        x = F.relu(self.dense1(x))
        x = self.drop5(x)
        # 1000 -> 1000
        x = F.relu(self.dense2(x))
        x = self.drop6(x)
        # 1000 -> 136
        x = self.dense3(x)
        return x



