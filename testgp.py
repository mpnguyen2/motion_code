import math

import torch
import gpytorch

from train import  train_multioutput_gp
from train_utils import test_gp_step
from utils import plot_gp

if __name__ == '__main__':
    # Train data
    x_train = torch.linspace(0, 1, 100)
    y_train = torch.stack([
        torch.sin(x_train * (2 * math.pi)) + torch.randn(x_train.size()) * 0.2,
        torch.cos(x_train * (2 * math.pi)) + torch.randn(x_train.size()) * 0.2,
        torch.sin(x_train * (2 * math.pi)) + 2 * torch.cos(x_train * (2 * math.pi)) + torch.randn(x_train.size()) * 0.2,
        -torch.cos(x_train * (2 * math.pi)) + torch.randn(x_train.size()) * 0.2,
    ], -1)
    # Test data
    x_test = torch.linspace(0, 1, 51)
    y_test = torch.stack([
        torch.sin(x_test * (2 * math.pi)),
        torch.cos(x_test * (2 * math.pi)),
        torch.sin(x_test * (2 * math.pi)) + 2 * torch.cos(x_test * (2 * math.pi)),
        -torch.cos(x_test * (2 * math.pi)),
    ], -1)
    # Data info
    num_tasks = 4
    num_latents = 3
    inducing_points = torch.rand(num_latents, 16, 1)
    # Training
    model, likelihood = train_multioutput_gp(x_train, y_train, num_tasks, num_latents, inducing_points, num_epochs=500, lr=1e-2)
    # Testing
    test_gp_step(model, likelihood, x_test, y_test, num_tasks)
    learned_inducing_points = model.state_dict()['variational_strategy.base_variational_strategy.inducing_points'].cpu()
    print(torch.mean(torch.abs(inducing_points - learned_inducing_points)))

    # Plotting
    #plot_gp(x_test, x_train, y_train, num_tasks, model, likelihood)