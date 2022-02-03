import tqdm
import math
from math import floor

import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

def getDataLoader(X, y, train_percent=0.8, train_batch_size=1024, test_batch_size=1024):
    # Train test split
    train_n = int(floor(train_percent * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()
    # Put data to cuda
    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    # Tensorize and create trainloader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    # Tensorize and create testloader
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_y.size(0)

def train_gp_step(model, likelihood, optimizer, num_epochs, train_loader, num_data):
    model.train()
    likelihood.train()
    print('Start training Gaussian process...')
    # We're using the VariationalELBO loss
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
    print('Training complete.\n')


def test_gp_step(model, likelihood, test_loader):
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    test_y = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
            test_y = torch.cat([test_y, y_batch.cpu()])
    means = means[1:]
    test_y = test_y[1:]
    print('Test MAE: {}'.format(torch.mean(torch.abs(means - test_y))))