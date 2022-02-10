import torch
from sparse_gp.train import train_multioutput_gp

def learn_motion_latent(X, y, latent_dim, num_epochs=500, lr=1e-2):
    num_tasks = y.shape[1]
    num_latents = num_tasks
    inducing_points = torch.rand(num_latents, latent_dim, X.shape[1])
    # Training
    model, _ = train_multioutput_gp(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float), 
                num_tasks, num_latents, inducing_points, num_epochs, lr)
    # Return inducing points
    learned_inducing_points = model.state_dict()['variational_strategy.base_variational_strategy.inducing_points'].cpu()
    learned_inducing_points = learned_inducing_points.numpy().swapaxes(0, 1).reshape(latent_dim, -1)

    return learned_inducing_points

def find_trajectories_latent(trajectories, latent_dims, num_epochs):
    latent_timeseries = []
    # Tranform each trajectory into a vector of inducing points
    for i in range(len(latent_dims)):
        X, y = trajectories[i]
        latent_timeseries.append(learn_motion_latent(X, y, latent_dims[i], num_epochs=num_epochs))
    return latent_timeseries