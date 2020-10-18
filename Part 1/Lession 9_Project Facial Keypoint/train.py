import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Net

if __name__ == '__main__':
    net = Net()
    print(net)