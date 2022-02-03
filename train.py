import torch
import gpytorch

from train_utils import *
from model_nets import GPModel

def train_gp(X, y, inducing_points, num_epochs):
    # Prepare dataloader
    train_loader, test_loader, num_data = getDataLoader(X, y)
    # Initialize model
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
                    ], lr=0.01)
    # Training step
    train_gp_step(model, likelihood, optimizer, num_epochs, train_loader, num_data)
    # Testing step
    test_gp_step(model, likelihood, test_loader)
    

