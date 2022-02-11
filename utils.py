import numpy as np
from matplotlib import pyplot as plt
import torch
import gpytorch

# Convert next line (from text data with handle f) data to a list of appropriate type
def read_array(f, type):
    next_line = f.readline().split()
    return_arr = []
    if type == 'int':
        for d in next_line:
            return_arr.append(int(d))
    if type == 'double':
        for d in next_line:
            return_arr.append(float(d))
    return np.array(return_arr)

def read_trajectory(input_file):
    """ Structure of input file:
        First line: Number of trajectories
        Next we have paragraphs where each corresponds to a trajectory
            First line: num_pts num_se3 num_so3
            Second line: time stamps (num_pts double numbers)
            Each of the next 3*num_pts line include:
                First: num_se3 tuples of 3 double numbers for each SE3 component position
                Second: num_se3 tuples of 3 double numbers for each SE3 component angle
                Third: num_so3 tuples of 3 double numbers for each SO3 component angle
            Each of the next 2*num_pts line include:
                First: num_se3 tuples of 3 double numbers for each SE3 component velocity
                First: num_so3 tuples of 3 double numbers for each SO3 component velocity
    """
    f = open(input_file, "r")
    trajectories = []
    num_traj = int(f.readline())
    for _ in range(num_traj):
        # Get number of pts on trajectory and configuration of the rigid obj (num_se3, num_so3)
        config_data = read_array(f, type='int')
        num_pts, num_se3, num_so3 = config_data[0], config_data[1], config_data[2]
        # Get timestamps
        times = read_array(f, type='double')
        # Now read in the pose of obj in each of the num_pts
        # The pose in encoded in a (num_se3*3 + num_se3*3 + num_so3*3)-dim vector
        X = np.zeros((num_pts, num_se3*6 + num_so3*3), dtype=float)
        for i in range(num_pts):
            # Read SE3 position data
            X[i, :num_se3*3] = read_array(f, type='double')
            # Read SE3 angular data
            X[i, num_se3*3:num_se3*6] = read_array(f, type='double')
            # Read SO3 angular data
            X[i, num_se3*6:] = read_array(f, type='double')
        # Now read velocity (motion) along this trajectory traveled by the rigid obj
        y = np.zeros((num_pts, num_se3*3+num_so3*3))
        for i in range(num_pts):
            # Read SE3 velocity data
            y[i, :num_se3*3] = read_array(f, type='double')
        for i in range(num_pts):
            # Read SO3 velocity data
            y[i,num_se3*3:] = read_array(f, type='double')
        # Add new pair (X, y) to trajectories list
        trajectories.append((X, y, times))
    # Close file 
    f.close()

    return trajectories

def plot_gp(x_test, x_train, y_train, num_tasks, model, likelihood, save_file='test_gp.png'):
    model.eval()
    likelihood.eval()
    # Initialize plots
    fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(x_test.cuda()))
        mean = predictions.mean
        mean = mean.cpu()
        lower, upper = predictions.confidence_region()
        lower, upper = lower.cpu(), upper.cpu()
    # Plot test data against train data for each separate task or individual component of y
    for task, ax in enumerate(axs):
        # Plot training data as black stars
        ax.plot(x_train.detach().numpy().reshape(-1), y_train[:, task].detach().numpy(), 'k*')
        # Predictive mean as blue line
        ax.plot(x_test.numpy(), mean[:, task].numpy(), 'b')
        # Shade in confidence
        ax.fill_between(x_test.numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(f'Task {task + 1}')

    fig.tight_layout()
    plt.savefig(save_file)

if __name__ == '__main__':
    trajectories = read_trajectory('motions.txt')
    for traj in trajectories:
        X, y = traj
        print(X.shape, y.shape)