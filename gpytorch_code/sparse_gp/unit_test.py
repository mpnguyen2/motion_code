import math

import torch
import gpytorch

from train import train_multioutput_gp
from train_utils import test_gp_step

import unittest

class TestClass(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestClass, self).__init__(*args, **kwargs)
    
    def test_gp(self):
        # Train and test data for multi-output gp
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
        MAE_err = test_gp_step(model, likelihood, x_test, y_test, num_tasks)
        self.assertLess(MAE_err, 0.1, 'Produce wrong GP model')

if __name__ == '__main__':
    unittest.main()