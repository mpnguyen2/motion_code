from matplotlib import pyplot as plt
import torch
import gpytorch

def read_trajectory():
    pass

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