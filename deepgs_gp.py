import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import Deepgs, GSdata, DKLModel, plot

import pdb

save = False
num_markers = 5000
train_set = GSdata(num_markers,False,0.1,0)
val_set = GSdata(num_markers,True,0.1,0)

train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
val_loader = DataLoader(val_set, batch_size=20, shuffle=True)

model = DKLModel()
likelihood = GaussianLikelihood()
#likelihood = gpytorch.likelihoods.SoftmaxLikelihood(n_features=model.n_features, n_classes=1)


# Find optimal model hyperparameters
model.train()
likelihood.train()

lr = 0.01
optimizer = torch.optim.SGD([
    {'params': model.feature_extractor.parameters(), 'lr': 0.01, 'weight_decay' : 1e-6, 'momentum': 0.9},
    {'params': model.gp_layer.parameters(), 'lr': lr},
    {'params': likelihood.parameters(), 'lr': lr},
], lr=lr, momentum=0, nesterov=False, weight_decay=0)

mll = gpytorch.mlls.VariationalMarginalLogLikelihood(likelihood, model, n_data=train_set.target.size)
mae = nn.L1Loss()

train_likelihood = []
train_loss = []
val_loss = []
for epoch in range(50):
    print('Epoch {}:'.format(epoch+1))
    acc_loss = 0
    acc_likelihood = 0
    for i_batch, sample_batched in enumerate(tqdm(train_loader,ascii=True)):

        inp, target = sample_batched

        model.train()
        likelihood.train()
        with gpytorch.settings.use_toeplitz(False):
            try:
                output = model(inp)
            except:
                pdb.set_trace()
            loss = -mll(output,target)
   
        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()        

        acc_likelihood += loss.item()

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False): 
            try:
                output = model(inp)
            except:
                pdb.set_trace()
            loss = mae(output.mean(),target)

        acc_loss += loss.item()     

    train_loss.append(acc_loss / (i_batch+1))
    train_likelihood.append(acc_likelihood / (i_batch+1))
    
    model.eval()
    likelihood.eval()
    acc_loss = 0
    for i_batch, sample_batched in enumerate(val_loader):   

        inp, target = sample_batched
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False): 
            output = model(inp)
            loss = mae(output.mean(),target)

        acc_loss += loss.item()

    val_loss.append(acc_loss / (i_batch+1)) 

    print('Train likelihood: {}, train mae: {}, validation mae: {}'.format(train_likelihood[-1],train_loss[-1],val_loss[-1]))
    print()

if save:
    torch.save(model.state_dict(),'model_gp.torch')

plot(model, val_set, train_loss, val_loss, True)
