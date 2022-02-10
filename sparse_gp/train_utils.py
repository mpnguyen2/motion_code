import tqdm
import math
from math import floor
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

def train_gp_step(model, mll, optimizer, x_train, y_train, num_epochs=10):
    print('Start training Gaussian process...')
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        epochs_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
    print('Training complete.\n')

def predict_gp(model, likelihood, x):
    if torch.cuda.is_available():
        x = x.cuda()
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(x))
        mean = predictions.mean
        mean = mean.cpu()
        lower, upper = predictions.confidence_region()
        lower, upper = lower.cpu(), upper.cpu()

    return mean, lower, upper

def test_gp_step(model, likelihood, x_test, y_test, num_tasks):
    model.eval()
    likelihood.eval()
    means, _, _ = predict_gp(model, likelihood, x_test)

    return torch.mean(torch.abs(means - y_test))