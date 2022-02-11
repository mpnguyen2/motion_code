import numpy as np
import torch
from sparse_gp.train import train_multioutput_gp
from sparse_gp.train_utils import predict_gp

def learn_motion_latent(X, y, times, num_induce_pts, mode='space', num_epochs=500, lr=1e-2):
    # number of tasks and of output latent is the same as number of components of velocity
    num_tasks = y.shape[1]
    num_y_latents = num_tasks
    # Find latent velocity-valued fct on inducing point when f(X) = y maps pose to velocity
    if mode == 'space':
        inducing_points = torch.rand(num_y_latents, num_induce_pts, X.shape[1])
        # Training
        model, likelihood = train_multioutput_gp(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float), 
                    num_tasks, num_y_latents, inducing_points, num_epochs, lr)
        # Return inducing points
        learned_inducing_points = model.state_dict()['variational_strategy.base_variational_strategy.inducing_points'].cpu()
        #learned_inducing_points = learned_inducing_points.numpy().swapaxes(0, 1).reshape(num_induce_pts, -1)
        latent_result = []
        for i in range(num_y_latents):
            mean, _, _ = predict_gp(model, likelihood, learned_inducing_points[i])
            latent_result.append(mean.cpu().numpy()[:, i].reshape(-1))
        latent_result = np.array(latent_result).reshape(-1)
        #latent_result = np.concatenate((latent_result, learned_inducing_points.cpu().numpy().reshape(-1)))

    # Find latent time-series when we focus on velocity y = f(t) maps time to velocity
    if mode == 'time':
        inducing_points = torch.rand(num_y_latents, num_induce_pts, 1)
        # Training
        model, likelihood = train_multioutput_gp(torch.tensor(times.reshape(-1, 1), dtype=torch.float), torch.tensor(y, dtype=torch.float), 
                    num_tasks, num_y_latents, inducing_points, num_epochs, lr)
        # Return inducing points
        learned_inducing_points = model.state_dict()['variational_strategy.base_variational_strategy.inducing_points'].cpu()
        #learned_inducing_points = learned_inducing_points.numpy().swapaxes(0, 1).reshape(num_induce_pts, -1)
        inducing_times = []
        latent_result = []
        for i in range(num_y_latents):
            ti = learned_inducing_points.numpy()[i]
            inducing_times.extend(ti)
        inducing_times = torch.tensor(np.array(inducing_times).reshape(-1, 1), dtype=torch.float)
        mean, _, _ = predict_gp(model, likelihood, inducing_times)
        return mean
        
    # Find latent time-series when f(X, t) = y maps (pose, time) to velocity
    if mode == 'space-time':
        pass
    
    return latent_result

def find_trajectories_latent(trajectories, latent_dims, num_epochs, mode='space'):
    latents = []
    # Tranform each trajectory into a vector of inducing points
    for i in range(len(latent_dims)):
        X, y, times = trajectories[i]
        latents.append(learn_motion_latent(X, y, times, latent_dims[i], mode=mode, num_epochs=num_epochs))
    return latents