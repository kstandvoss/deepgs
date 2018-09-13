import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, GridInterpolationKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.random_variables import GaussianRandomVariable
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

import pdb

def plot(net, val_set, train_loss, val_loss, gp=False):
    
    if gp:
        train_str = 'Mean absolute Error'
    else:
        train_str = 'ELBO'

    val_x, val_y = val_set.get_data()
    
    with torch.no_grad():
        pred = net(val_x.float())
        if gp:
            pred = pred.mean().view(-1,1)    
        val_y = val_y.view(-1,1).float()
        data = torch.cat((pred,val_y),1).numpy()

    mnvs = []
    for k in range(1,len(val_x)+1):
        mnvs.append(MNV(k,data))  

    f = plt.figure()
    f.suptitle('Training Error')
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel(train_str)

    f = plt.figure()
    f.suptitle('Validation Error')
    plt.plot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Mean absolute Error')

    f = plt.figure()
    f.suptitle('MNV')
    plt.plot(mnvs)
    plt.xlabel(r'Top $\mathbf{\alpha}\%$')
    plt.xlim([1,100])
    plt.ylabel('MNV')

    plt.show()

def MNV(k, data, sort_idx=None):
    #pdb.set_trace()
    if not k:
        return 0
    if sort_idx is None:
        pi_x = (-data[:,0]).argsort()
        pi_y = (-data[:,1]).argsort()
        sort_idx = [pi_x, pi_y]
    else:
        pi_x, pi_y = sort_idx

    idx = np.arange(k)    
    d = 1/(np.log2(idx+2))
    return 1/k * ((k-1)*MNV(k-1,data, sort_idx) + np.sum(data[pi_x][idx,1]*d)/np.sum(data[pi_y][idx,1]*d))


class GSdata(Dataset):

    def __init__(self, num_markers, val, val_split, val_idx):

        np.random.seed(42)

        data = pd.read_pickle('x.pkl')
        #phenotypes y
        #0:tkw    1:testw   2:length  3:width   4:Hard    5:Prot    6:SDS 7:PHT
        target = pd.read_pickle('y.pkl')
        
        l = int(len(data) * val_split)
        idx = l * val_idx
        mask = np.zeros(len(data), dtype=bool)
        mask[idx:idx+l] = True
            
        if not val:
            mask = ~mask
        
        marker_idx = np.random.choice(np.arange(data.shape[1]), num_markers, replace=False)
        self.data = data.as_matrix()[mask][:,marker_idx]
        #pdb.set_trace()
        self.target = target.as_matrix()[mask,2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return torch.tensor(self.data[idx]).unsqueeze(0), torch.tensor(self.target[idx])

    def get_data(self):
        return torch.from_numpy(self.data).unsqueeze(1), torch.from_numpy(self.target)    


class Deepgs(nn.Module):

    def __init__(self, n_features=1, dropout=True):
        super(Deepgs, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 18)
        self.pool1 = nn.MaxPool1d(4, 4)
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(9960, 32)
        self.drop2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, n_features)

        self.dropout = dropout

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        if self.dropout:
            x = self.drop1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.drop2(x)

        x = self.fc2(x)
        return x

class GPLayer(gpytorch.models.AdditiveGridInducingVariationalGP):

    def __init__(self, n_features, grid_size=100, grid_bounds=(-1.1, 1.1)):
            #super(GPLayer, self).__init__(grid_size=grid_size, grid_bounds=n_features*[grid_bounds])
            super(GPLayer, self).__init__(grid_size=grid_size, grid_bounds=[grid_bounds],
                                              n_components=n_features, mixing_params=False, sum_output=True)
            self.grid_bounds = grid_bounds
            self.mean_module = ConstantMean()
            self.covar_module = RBFKernel()#ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        #pdb.set_trace()
        return GaussianRandomVariable(mean_x, covar_x)
 
class DKLModel(gpytorch.Module):
        def __init__(self):
            super(DKLModel, self).__init__()
            self.n_features = 2
            self.feature_extractor = Deepgs(self.n_features,dropout=False)
            self.gp_layer = GPLayer(self.n_features)

        def forward(self, x):
            features = self.feature_extractor(x.float())
            #print(features)
#            if features.allclose(features[0]):
                #features.fill_(1)
#                pdb.set_trace()
            #else:
            features = gpytorch.utils.scale_to_bounds(features, self.gp_layer.grid_bounds[0], self.gp_layer.grid_bounds[1])      
            self.features = features    
            return self.gp_layer(features.unsqueeze(-1))
            
        