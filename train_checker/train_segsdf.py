# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
from network import myMLP
import pickle
import math

def train_seg_positocdist(number, i):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")


    file_path = 'dataset/seg_positocdist'+str(i+1)+'.pkl'

    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    print("-----第%d关节训练-----" % (i+1))
    batch_size = len(dataset) // 10000  # 
    print("batch_size=", batch_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("len(dataloader)=", len(dataloader))
    epochs = 100
    input_size=3
    print("input_size=",input_size)
    output_size = 1
    bp_model = myMLP(input_size, output_size, mlp_layers=[64, 64,64])
    bp_model = bp_model.to(device)

    criterion = nn.MSELoss()  
    criterion = criterion.to(device)
    optimizer = optim.Adam(bp_model.parameters(), lr=0.005)  

    print("bp_model's state_dict:")
    for param_tensor in bp_model.state_dict():
        print(param_tensor, "\t", bp_model.state_dict()[param_tensor].size())
    time0 = time()
    Training_bias = []
    for e in range(epochs):
        running_loss = 0
        for input_batch, target_batch in dataloader:
                
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            output_batch = bp_model.forward(input_batch)

            loss = criterion(output_batch, target_batch)

            loss.backward()

            optimizer.step()
            running_loss += loss.item()
        else:

            mse_loss = running_loss/len(dataloader)
            bias = math.sqrt(mse_loss)

            print("Epoch {} - MSE loss: {}  - Bias: {}".format(e, mse_loss, bias))
            Training_bias.append(bias)
    print("\nTraining Time (in minutes) =", (time()-time0)/60)
    directory = "model/seg_positocdist/train" + str(number)
    if not os.path.exists(directory):
        os.makedirs(directory)

    PATH1 = os.path.join(directory, 'train-seg'+str(i+1)+'.pt')
    torch.save(bp_model, PATH1)

    PATH2 = os.path.join(directory, 'Checkpoint-seg'+str(i+1))
    torch.save({
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
        'Training_bias': Training_bias,
    }, PATH2)

    PATH3 = os.path.join(directory, 'Training_bias_plot-seg'+str(i+1)+'.png')
    plt.plot(range(epochs), Training_bias)
    plt.savefig(PATH3)
if __name__ == '__main__':
    number = 364
    for i in range(7):
        train_seg_positocdist(number,i)
