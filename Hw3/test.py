import sys

import numpy as np
import scipy.special

import textwrap


import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self,dimensions):
          super(Attention, self).__init__()

          self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
          self.softmax = nn.Softmax(dim=-1)
          self.tanh = nn.Tanh()

    def forward(self, query, key, value, mask, scale=True):

    


        mask_size = torch.matmul(query,torch.swapaxes(key,-1,-2)).shape


    
        mask = torch.tril(torch.ones(mask_size))
        mask = (mask > 0)
        #.type(torch.uint8)  

        depth = query.shape[-1]

        dots = (torch.matmul(query,np.swapaxes(key,-1,-2)) / torch.sqrt(torch.tensor(depth))).type(torch.LongTensor) 
        print(dots.type)
        if mask is not None:
            dots = torch.where(mask, dots, torch.full_like(dots, -1e9))
        
        # Calculate softmax
        # The scipy.special.logsumexp function avoids problems with dividing by large numbers

        logsumexp = torch.logsumexp(dots, axis=-1, keepdims=True)

        # Getting sotmax
        #dots = nn.Softmax(dots)
        dots = torch.exp(dots - logsumexp)
        # Multiply dots by value to get self-awareness
        # Use np.matmul()

        #attention = torch.matmul(torch.tensor(dots), value)

        return dots

import math



q = torch.tensor([[[1, 0, 0], [0, 1, 0]]])
k = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
v = torch.tensor([[[0, 1, 0], [1, 0, 1]]])
m = torch.tensor([[[0, 0], [-1e9, 0]]]).float()



attention = Attention(1)
query = q
context = k
output, weights = attention(query, context, v, m)
output.size()
weights.size()


depth = q.shape[-1]
attention = Attention(depth)
output, weights = attention(q1, k1, v1, m1)
assert output.size() == torch.Size([1, 2, 3])
assert weights.size() == torch.Size([1, 2, 2])
assert np.allclose(
    output.detach().numpy(),
    np.array([[[0.29135218,  0.49732763,  0.4825794], [-0.09023539, -0.33851343, -0.18972817]]])
)







assert np.allclose(np.array([[[0.29135218,  0.49732763,  0.4825794], [-0.09023539, -0.33851343, -0.18972817]]]), np.array([[[0., 1., 0.],[0.84967455, 0.15032545, 0.84967455]]]))