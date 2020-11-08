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
        self.drop3 = nn.Dropout(p=0.25)

        self.conv2d4 = nn.Conv2d(128, 256, 1)
        self.drop4 = nn.Dropout(p=0.25)
        # dense layer
        self.dense1 = nn.Linear(256 * 13 * 13, 1000)
        self.drop5 = nn.Dropout(p=0.3)

        self.dense2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout(p=0.4)

        self.dense3 = nn.Linear(1000, 136)

    def forward(self, x):
        # new = (w,h - f) / s + 1, f is kernel size, s is trike
        # (1, 224, 224) -> (32, 221, 221) -> (32, 110, 110)
        x = self.maxpool2d(F.relu(self.conv2d1(x)))
        x = self.drop1(x)
        # (32, 110, 110) -> (64, 108, 108) -> (64, 54, 54)
        x = self.maxpool2d(F.relu(self.conv2d2(x)))
        x = self.drop2(x)
        # (64, 54, 54) -> (128, 53, 53) -> (128, 26, 26)
        x = self.maxpool2d(F.relu(self.conv2d3(x)))
        x = self.drop3(x)
        # (128, 26, 26) -> (256, 26, 26) -> (256, 13, 13)
        x = self.maxpool2d(F.relu(self.conv2d4(x)))
        x = self.drop4(x)
        # (256, 13, 13) -> (256*13*13)
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



