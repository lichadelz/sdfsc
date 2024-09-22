# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ReLU6, ELU, Dropout, BatchNorm1d as BN, LayerNorm as LN, Tanh
from torch.nn.parallel import parallel_apply

def MLP(channels, act_fn=ReLU, islast = False):
    """Automatic generation of mlp given some

    Args:
        channels (int): number of channels in input
        dropout_ratio (float, optional): dropout used after every layer. Defaults to 0.0.
        batch_norm (bool, optional): batch norm after every layer. Defaults to False.
        act_fn ([type], optional): activation function after every layer. Defaults to ReLU.
        layer_norm (bool, optional): layer norm after every layer. Defaults to False.
        nerf (bool, optional): use positional encoding (x->[sin(x),cos(x)]). Defaults to True.

    Returns:
        nn sequential layers 
    """
    if not islast:
        layers = [Seq(Lin(channels[i - 1], channels[i]), act_fn())
                  for i in range(1, len(channels))]
    else:
        layers = [Seq(Lin(channels[i - 1], channels[i]), act_fn())
                  for i in range(1, len(channels)-1)]
        layers.append(Seq(Lin(channels[-2], channels[-1])))
    
    layers = Seq(*layers)
    return layers
class myMLP(nn.Module):
    def __init__(self, input_size, output_size, mlp_layers=[128],act_fn=ReLU):
        super(myMLP, self).__init__()
        mlp_arr = []
        mlp_arr.append(mlp_layers)
        
        mlp_arr[-1].append(output_size)
        mlp_arr[0].insert(0,input_size)
        self.layers = nn.ModuleList()
        
        for arr in mlp_arr[0:-1]:
            self.layers.append(MLP(arr,act_fn=act_fn, islast=False))
        self.layers.append(MLP(mlp_arr[-1],act_fn=act_fn, islast=True))
        
    def forward(self, x):
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            y = layer(torch.cat((y,x), dim=1))
        return y
    
