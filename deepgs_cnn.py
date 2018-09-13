import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Deepgs, GSdata, plot

import pdb

save = False
num_markers = 5000
train_set = GSdata(num_markers,False,0.1,0)
val_set = GSdata(num_markers,True,0.1,0)
train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
val_loader = DataLoader(val_set, batch_size=20, shuffle=True)

net = Deepgs()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-6, nesterov=False)
criterion = nn.L1Loss()

train_loss = []
val_loss = []
for epoch in range(25):
    print('Epoch {}:'.format(epoch+1))

    net.train()
    acc_loss = 0
    for i_batch, sample_batched in enumerate(tqdm(train_loader,ascii=True)):

        inp, target = sample_batched

        output = net(inp.float()).squeeze()
        loss = criterion(output,target)
   
        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()        

        acc_loss += loss.item()

    train_loss.append(acc_loss / (i_batch+1))
    
    net.eval()
    acc_loss = 0
    for i_batch, sample_batched in enumerate(val_loader):   

        inp, target = sample_batched
        
        with torch.no_grad(): 
            output = net(inp.float()).squeeze()
            loss = criterion(output,target)

        acc_loss += loss.item()

    val_loss.append(acc_loss / (i_batch+1)) 

    print('Train loss: {}, validation loss: {}'.format(train_loss[-1],val_loss[-1]))
    print()

if save:
    torch.save(net.state_dict(),'model_cnn.torch')

plot(net, val_set, train_loss, val_loss)
