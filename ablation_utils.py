import math, random
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sparse_gp import sigmoid, sigmoid_inv, softplus_inv
from sparse_gp import unpack_params_single, elbo_fn_single, pack_params


COLOR_LIST = ['red', 'green', 'orange', 'blue', 'purple', 'black', 'brown', 'grey', 'yellow', 'black', 'hotpink']
MARKER_COLOR_LIST = ['green', 'blue', 'red', 'orange', 'purple', 'black', 'brown', 'grey', 'yellow', 'black', 'hotpink']
markers = ["." , "," , "o" , "v" , "^" , "<", ">"]


def get_informative_pts_for_individual_series(data, num_motion, m, Q):
    X_train, Y_train, labels_train, X_test, Y_test, labels_test = data

    # Initialize parameters
    X_m_start = sigmoid_inv(np.linspace(0.1, 0.9, m))
    Sigma_start = softplus_inv(np.ones(Q))
    W_start = softplus_inv(np.ones(Q))

    # Get informative points list
    X_m_list = [[] for _ in range(2*num_motion)]
    train_len = len(X_train)
    dims = (m, Q)
    for i in tqdm(range(train_len + len(X_test)),
                  desc="Optimizing informative points"):
        if i < train_len:
            X = X_train[i]; Y = Y_train[i]
        else:
            X = X_test[i-train_len]; Y = Y_test[i-train_len]
        # Optimize X_m, and kernel parameters including Sigma, W
        res = minimize(fun=elbo_fn_single(X, Y, sigma_y=0.1, dims=dims),
                        x0 = pack_params([X_m_start, Sigma_start, W_start]),
                        method='L-BFGS-B', jac=True)
        X_m, _, _ = unpack_params_single(res.x, dims=dims)
        if i < train_len:
            X_m_list[labels_train[i]].append(sigmoid(X_m))
        else:
            X_m_list[labels_test[i-train_len] + num_motion].append(sigmoid(X_m)) 
        
    return X_m_list


# Reorganize data
def truncate_data(data, max_num=50):
    X_train, Y_train, labels_train, X_test, Y_test, labels_test = data

    # Truncate train
    combined = list(zip(X_train, Y_train, labels_train))
    random.shuffle(combined)
    X_train, Y_train, labels_train = zip(*combined)
    max_num = min(max_num, len(X_train))
    X_train, Y_train, labels_train = X_train[:max_num], Y_train[:max_num], labels_train[:max_num]

    # Truncate test
    combined = list(zip(X_test, Y_test, labels_test))
    random.shuffle(combined)
    X_test, Y_test, labels_test = zip(*combined)
    max_num = min(max_num, len(X_test))
    X_test, Y_test, labels_test = X_test[:max_num], Y_test[:max_num], labels_test[:max_num]
    return X_train, Y_train, labels_train, X_test, Y_test, labels_test


def group_series(data, X_m_list, num_motion):
    X_train, Y_train, labels_train, X_test, Y_test, labels_test = data
    grouped_data = {
        'train': [[] for _ in range(num_motion)],
        'test': [[] for _ in range(num_motion)]
    }

    # Track how many samples processed per label
    num_samples_processed_train = [0] * num_motion
    num_samples_processed_test = [0] * num_motion

    # Group training data
    for x, y, label in zip(X_train, Y_train, labels_train):
        Xm_idx = num_samples_processed_train[label]
        X_m = X_m_list[label][Xm_idx]
        Y_m = np.interp(X_m, x, y)
        grouped_data['train'][label].append((x, y, X_m, Y_m))
        num_samples_processed_train[label] += 1

    # Group test data
    for x, y, label in zip(X_test, Y_test, labels_test):
        idx = label + num_motion
        Xm_idx = num_samples_processed_test[label]
        X_m = X_m_list[idx][Xm_idx]
        Y_m = np.interp(X_m, x, y)
        grouped_data['test'][label].append((x, y, X_m, Y_m))
        num_samples_processed_test[label] += 1

    return grouped_data


# Plotting
def _plot_series(groups, max_per_group, split_type, marker, label_names, plot_path):
    num_motions = len(groups)
    groups_capped = [random.sample(g, min(len(g), max_per_group)) for g in groups]
    # groups_capped = [g[:min(len(g), max_per_group)] for g in groups]
    n_cols = max([len(g) for g in groups_capped])
    n_rows = num_motions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True)
    
    # Normalize axes to 2D list for easy indexing
    axes = np.array(axes).reshape(n_rows, n_cols)
    legend_handles = []
    total_n_rows = 0
    for label, group in enumerate(groups_capped):
        line_color = COLOR_LIST[label]
        point_color = MARKER_COLOR_LIST[(label+1) % num_motions]
        n_samples = len(group)
        cur_n_rows = math.ceil(n_samples / n_cols)
        label_str = label_names[label] if label_names else f"Class {label}"

        # Legend entry for each class
        legend_handles.append(Line2D(
            [0], [0], color=line_color, marker=marker, markersize=6,
            markerfacecolor=point_color, markeredgecolor=point_color,
            label=label_str
        ))
     
        for j, (x, y, X_m, Y_m) in enumerate(group):
            row, col = 0, j #divmod(j, n_cols)
            ax = axes[row + total_n_rows, col]
            ax.plot(x, y, color=line_color, alpha=0.5)
            ax.scatter(X_m, Y_m, color=point_color, marker=marker)
            ax.set_title(f'{label_str} - Sample {j}')
            ax.set_xlabel("Time")
            if col == 0:
                ax.set_ylabel("Value")
            ax.grid(True)

        # Hide any unused subplots
        for j in range(n_samples, cur_n_rows * n_cols):
            row, col = 0, j #divmod(j, n_cols)
            axes[row + total_n_rows, col].axis("off")
        total_n_rows += cur_n_rows

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.01),  # Push below the figure
        fontsize=10,
        frameon=False
    )
    fig.suptitle("Individual time series with individual inducing (informative) points from the "\
                 + split_type + " set", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
    #plt.show()
    plt.savefig(plot_path)
    plt.close()


def plot_grouped_series_separate(grouped_data, max_per_group,
                                 label_names,
                                 plot_path_prefix='out/'):
    _plot_series(grouped_data['train'], max_per_group['train'],
                 split_type='train', marker='o',
                 label_names=label_names,
                 plot_path=plot_path_prefix + '_train.png')
    _plot_series(grouped_data['test'], max_per_group['test'],
                 split_type='test', marker='x',
                 label_names=label_names,
                 plot_path=plot_path_prefix + '_test.png')


def visualize_data_by_informative_pts(X_m_list, num_motion,
                                      label_names=[],
                                      plot_path='out/cluster.png'):
    plt.figure(figsize=(8, 6))

    train_handles = []
    test_handles = []

    for k in range(num_motion):
        # Train data
        if len(X_m_list[k]) > 0:
            X_m_arr = np.array(X_m_list[k])
            U, S, _ = scipy.sparse.linalg.svds(X_m_arr, k=2)
            reduced = U @ np.diag(S)
            plt.scatter(reduced[:, 0], reduced[:, 1],
                        c=COLOR_LIST[k], marker='x', label=f'{label_names[k]} train data')
            train_handles.append(mlines.Line2D([], [], color=COLOR_LIST[k], marker='x',
                                               linestyle='None', markersize=6, label=f'{label_names[k]} train data'))

        # Test data
        if len(X_m_list[k + num_motion]) > 0:
            X_m_arr = np.array(X_m_list[k + num_motion])
            U, S, _ = scipy.sparse.linalg.svds(X_m_arr, k=2)
            reduced = U @ np.diag(S)
            plt.scatter(reduced[:, 0], reduced[:, 1],
                        c=COLOR_LIST[k], marker='o', label=f'{label_names[k]} test data')
            test_handles.append(mlines.Line2D([], [], color=COLOR_LIST[k], marker='o',
                                              linestyle='None', markersize=6, label=f'{label_names[k]} test data'))

    # Combine handles: train first, then test
    handles = train_handles + test_handles
    by_label = dict((h.get_label(), h) for h in handles)
    plt.legend(by_label.values(), by_label.keys())

    plt.title("2D Representation of Informative Points (X_m)")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(plot_path)
    plt.close()


# Train classifiers
def train_classifier_on_Xm(grouped_data, clf='logistic'):
    # Get train/test from X_m's in grouped_data
    X_train, y_train, X_test, y_test = [], [], [], []
    for i, group in enumerate(grouped_data['train']):
        for (_, _, X_m, _) in group:
            X_train.append(X_m)
            y_train.append(i)
    for i, group in enumerate(grouped_data['test']):
        for (_, _, X_m, _) in group:
            X_test.append(X_m)
            y_test.append(i)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Choose classifier
    if clf == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif clf == 'gb':
        model = GradientBoostingClassifier()
    elif clf == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    elif clf == 'svm':
        model = SVC(kernel='rbf', C=1.0, gamma='scale')
    else:
        raise ValueError(f"Unknown classifier: {clf}")

    # Fit classifer
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc, model
