import argparse, os
import numpy as np
import pandas as pd
from data_processing import get_train_test_data_classify
import ablation_utils


def run_ablation(datasets, load_existing_data, output_path,
                 max_subplots_per_row, truncate_input,
                 num_truncate, visualize_cluster):
    acc_dict = {}
    for name in datasets:
        # Load data
        if not load_existing_data:
            add_noise=False
        else:
            add_noise=True
        _, data = get_train_test_data_classify(name, load_existing_data, add_noise)
        _, Y_train, labels_train, _, Y_test, _ = data
        num_motion = len(np.unique(labels_train))
        print(f'\n\nDataset: {name}, Train size: {len(Y_train)}, '
              f'Test size: {len(Y_test)}, Num classes: {num_motion}')
        if truncate_input:
            data = ablation_utils.truncate_data(data, max_num=num_truncate)
        
        # Assign label names
        if name == 'Pronunciation Audio':
            label_names = ['Absorptivity', 'Anything']
        elif name == 'PD setting 1':
            label_names = ['Normal', 'Light Tremor']
        elif name == 'PD setting 2':
            label_names = ['Normal', 'Light Tremor', 'Noticeable Tremor']
        elif name == 'Synthetic':
            label_names = ['Motion 1', 'Motion 2', 'Motion 3']
        elif name == 'MoteStrain':
            label_names = ['Humidity', 'Temperature']
        elif name == 'FreezerSmallTrain':
            label_names = ['Kitchen', 'Garage']
        elif name == 'PowerCons':
            label_names = ['Warm', 'Cold']
        elif name == 'ItalyPowerDemand':
            label_names = ['October to March', 'April to September']
        elif name == 'SonyAIBORobotSurface2':
            label_names = ['Cement', 'Carpet']
        elif name == 'FreezerSmallTrain':
            label_names = ['Kitchen', 'Garage']
        elif name == 'Chinatown':
            label_names = ['Weekend', 'Weekday']
        elif name == 'InsectEPGRegularTrain':
            label_names = ['Class 1', 'Class 2', 'Class 3']
        else:
            label_names = [str(i) for i in range(num_motion)]

        # Get individual informative points
        if name == 'PD setting 1':
            m, Q = 6, 2
        elif name == 'PD setting 2':
            m, Q = 12, 2
        else:
            m, Q = 10, 1
        X_m_list = ablation_utils.get_informative_pts_for_individual_series(data, num_motion, m, Q)

        # Plotting
        grouped_data = ablation_utils.group_series(data, X_m_list, num_motion)
        max_per_group = {
            'train': max_subplots_per_row,
            'test': max_subplots_per_row
        }
        save_path = os.path.join(output_path, 'plot', name)
        ablation_utils.plot_grouped_series_separate(grouped_data, max_per_group,
                                                    label_names=label_names,
                                                    plot_path_prefix=save_path)
        
        # Classifier
        acc_dict[name] = {}
        for clf in ['logistic', 'svm', 'rf', 'gb']:
            acc, _ = ablation_utils.train_classifier_on_Xm(grouped_data, clf=clf)
            print(f"Data: {name}, classifier: {clf.upper()}, test accuracy: {acc:.4f}")
            acc_dict[name][clf] = round(acc, 4)

        # Cluster
        if visualize_cluster:
            save_path = os.path.join(output_path, 'cluster', name + '.png')
            ablation_utils.visualize_data_by_informative_pts(X_m_list, num_motion=2,
                                label_names=label_names,
                                plot_path=save_path)
    
    # Save accuracy results
    results_df = pd.DataFrame.from_dict(acc_dict, orient='index')
    results_df.to_csv(os.path.join(output_path, "ablation_accuracy_results.csv"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLI arguments')
    parser.add_argument('--load_existing_data', type=int, default=1, help='Load existing data')
    parser.add_argument('--output_path', type=str, default='out/ablation', help='Output path')
    parser.add_argument('--max_subplots_per_row', type=int, default=4, help='number of subplots per row')
    parser.add_argument('--truncate_input', type=bool, default=False, help='Whether to truncate input')
    parser.add_argument('--num_truncate', type=int, default=50, help='Number of test samples for truncation')
    parser.add_argument('--visualize_cluster', type=bool, default=False, help='Visualize cluster')
    args = parser.parse_args()
    load_existing_data = bool(args.load_existing_data)
    output_path = args.output_path
    max_subplots_per_row = args.max_subplots_per_row
    truncate_input = args.truncate_input
    num_truncate = args.num_truncate
    visualize_cluster = args.visualize_cluster

    # Run ablation analysis
    datasets = ['Pronunciation Audio', 'PD setting 1', 'PD setting 2', 
                'ECGFiveDays', 'FreezerSmallTrain', 'HouseTwenty',
                'InsectEPGRegularTrain', 'ItalyPowerDemand', 'Lightning7',
                'MoteStrain', 'PowerCons', 'SonyAIBORobotSurface2',
                'UWaveGestureLibraryAll']
    #datasets = ['PD setting 2']
    #datasets = ['Pronunciation Audio']
    run_ablation(datasets, load_existing_data, output_path,
                 max_subplots_per_row, truncate_input,
                 num_truncate, visualize_cluster)