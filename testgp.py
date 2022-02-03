#import urllib.request
import os
from scipy.io import loadmat
#from matplotlib import pyplot as plt

import torch

from train import train_gp

# this is for running the notebook in our testing framework
smoke_test = False #('CI' in os.environ)
num_epochs = 1 if smoke_test else 4

#if not smoke_test and not os.path.isfile('../elevators.mat'):
#print('Downloading \'elevators\' UCI dataset...')
#urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')

if smoke_test:  # this is for running the notebook in our testing framework
    X, y = torch.randn(1000, 3), torch.randn(1000)
else:
    data = torch.Tensor(loadmat('elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]

inducing_points = X.contiguous()[:500, :]

train_gp(X, y, inducing_points, num_epochs)
