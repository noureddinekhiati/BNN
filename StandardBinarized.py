import torch
from torch.autograd import Function


class StandBinarize(Function):

    def forward(ctx,input):
        bin_output = input.new(input.size())
        bin_output[input >= 0] = 1
        bin_output[input < 0] = -1
        return bin_output

    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input