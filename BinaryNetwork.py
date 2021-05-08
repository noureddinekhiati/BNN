import torch
import torch.nn as nn
from torch.autograd import Function as F
from StandardBinarized import *
import math

class BinaryLinear(nn.Linear):

    def forward(self, input):
        binary_weight = StandBinarize.apply(self.weight)
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)



