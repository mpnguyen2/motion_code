import torch
import gpytorch

from train_utils import *
from model_nets import MultitaskGPModel

def train_multioutput_gp(x_train, y_train, num_tasks, num_latents, inducing_points, num_epochs=10, lr=1e-2):
    # Load data to CUDA if available
    num_data = y_train.shape[0]
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
    # Initialize model
    model = MultitaskGPModel(inducing_points, num_tasks, num_latents)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
                ], lr=lr)
    # Marginal loglikelihood: ELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    # Training step
    train_gp_step(model, mll, optimizer, x_train, y_train ,num_epochs)

    return model, likelihood