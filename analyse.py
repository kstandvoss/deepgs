import numpy as np 
import pandas as pd 
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import Deepgs, DKLModel, GSdata, MNV, plot

import pdb

num_markers = 5000
train_set = GSdata(num_markers,False,0.1,0)
val_set = GSdata(num_markers,True,0.1,0)
val_x, val_y = val_set.get_data()

model = DKLModel()
model.load_state_dict(torch.load('model_gp.torch'))

pdb.set_trace()

#plot(model, val_set, [], [], True)


