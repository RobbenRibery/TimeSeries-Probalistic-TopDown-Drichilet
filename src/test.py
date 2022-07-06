from cgi import test
from tkinter import N
from torch.nn import (
    LSTM, 
    MultiheadAttention, 
    Linear
)
from torch import tensor, rand, zeros, matmul, permute, log, lgamma
from torch import nn
from torch import optim 
import torch.nn.functional as F
from dirichlet import * 
import numpy as np 
import math 

no_child = 8
H = 3
input_dim = 1
embediing_dim = 12
lstm_hidden_dim = 48
lstm_layer_size = 1
number_attention_head = 4
number_attention_layer = 6 
input_dim = 3*H+embediing_dim
output_dim = input_dim
batch_size = 4

input_tensor = input = rand(2,H,1)
print(input_tensor)

output_tensor = F.softmax(input_tensor, dim = 1)

print(output_tensor)

print(output_tensor.sum(dim=1))


test_tensor = tensor([[1,2,3,4],
                      [5,6,7,8]])

print(test_tensor[:,-1])
print(test_tensor[:,-1].squeeze())