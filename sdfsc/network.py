# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ReLU6, ELU, Dropout, BatchNorm1d as BN, LayerNorm as LN, Tanh
from torch.nn.parallel import parallel_apply

def MLP(channels, act_fn=ReLU, islast = False):
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

class parallel_CombinedModel(nn.Module):
    def __init__(self, model_list):
        super(parallel_CombinedModel, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.num=len(model_list)
    def forward(self, x):
        outputs = []
        for idx, model in enumerate(self.models):
                if idx < x.size(0):  
                    output_i = model(x[idx])
                    outputs.append(output_i)
        combined_output = torch.cat(outputs, dim=1) 
        return combined_output
    
class parallel_mutilgpu_CombinedModel(nn.Module):
    def __init__(self, model_list):
        super(parallel_mutilgpu_CombinedModel, self).__init__()
        self.models = nn.ModuleList(model_list)
        
    def forward(self, x):
        outputs = []
        inputs = [x[i] for i in range(len(self.models))]
        outputs = parallel_apply(self.models, inputs)
        combined_output = torch.cat(outputs, dim=1)
        return combined_output
    
class parallel_cudastream_CombinedModel(nn.Module):
    def __init__(self, model_list):
        super(parallel_cudastream_CombinedModel, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.streams = [torch.cuda.Stream() for _ in model_list]
        self.models = self.models.to('cuda')
    def forward(self, x):
        outputs = []
        for idx, model in enumerate(self.models):
                if idx < x.size(0):  
                    self.streams[idx].wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.streams[idx]):
                         output_i = model(x[idx].unsqueeze(0)).to('cuda', non_blocking=True)
                    outputs.append(output_i)
        combined_output = torch.cat(outputs, dim=1)
        return combined_output








