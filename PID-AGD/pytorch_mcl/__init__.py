import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def _mcl(output, target, Beta, size_average = True):


    sco = torch.unsqueeze(target, 1)
    c_output = output.gather(1, sco)
    output_minus = output - c_output
    exp_minus = torch.exp(output_minus-70)
    loss = torch.sum((torch.log(torch.sum(exp_minus, dim=1)+(Beta-1)*math.exp(-70))+70 - math.log(Beta))) / 100

    # sco = torch.unsqueeze(target, 1)
    # softmax_func = nn.Softmax(dim=1)
    # softmax=softmax_func(output+50)
    # c_output = softmax.gather(1, sco)
    # k=(1-c_output)/c_output
    # k2=torch.log(k+0.00001)-math.log(0.00001)
    # loss = torch.sum(k2)/100
    return loss

class mcl(torch.nn.Module):
    def __init__(self, size_average = True):
        super(mcl, self).__init__()
        self.size_average = size_average

    def forward(self, output, target,Beta):

        return _mcl(output, target, Beta,self.size_average)

