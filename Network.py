import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.utils.data as D
import torch.optim as optim
from torch.autograd import Variable

import time
import random
import numpy as np
import matplotlib.pyplot as plt

from BinaryNetwork import *


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            BinaryLinear(28 * 28, 1024),
            nn.BatchNorm1d(1024),
            nn.Relu(),
            BinaryLinear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Relu(),
            BinaryLinear(1024, 10),
            nn.LogSoftmax()
        )

# Creating network object
network = Network()