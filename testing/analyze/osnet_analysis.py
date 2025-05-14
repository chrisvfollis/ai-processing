# standard dependencies
import os
import argparse
import argparse

# 3rd-party dependencies
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# internal dependencies
from testing.execute import osnet_tests
from utilities import io_utils


def analyze_embedding_data(
        dataset: str, embedding_distances: pd.DataFrame
    ) -> dict:
    def _calculate_summary_stats(same, different, all) -> dict:
        datasets = [
            'same_employee',
            'different_employee',
            'overall',
        ]
        summary_stats = []

        for distances in [same, different, all]:
            summary_stats.append({
                'count': len(distances),
                'mean': np.mean(distances),
                'median': np.median(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances)
            })

        return dict(zip(datasets, summary_stats))
    
    print('Calculating summary stats...')

    project_root = io_utils.get_project_root()

    summary = {}

    id_col_1a = embedding_distances['person_id1']
    id_col_2a = embedding_distances['person_id2']

    same_employee = embedding_distances[id_col_1a == id_col_2a]
    different_employee = embedding_distances[id_col_1a != id_col_2a]

    summary['global'] = _calculate_summary_stats(
        same_employee['distance'], different_employee['distance'],
        embedding_distances['distance'],
    )

    per_employee_stats = {}
    for employee_id in pd.concat([id_col_1a, id_col_2a]).unique():
        employee_distances = embedding_distances[
            (id_col_1a == employee_id) |
            (id_col_2a == employee_id)
        ]

        id_col_1b = employee_distances['person_id1']
        id_col_2b = employee_distances['person_id2']

        same = employee_distances[id_col_1b == id_col_2b]
        different = employee_distances[id_col_1b != id_col_2b]

        per_employee_stats[employee_id] = _calculate_summary_stats(
            same['distance'],
            different['distance'],
            employee_distances['distance'],
        )

    summary['per_employee'] = per_employee_stats

    global_df = pd.DataFrame(summary['global']).T

    per_emp_df = (
        pd.DataFrame.from_dict(summary['per_employee'], orient='index')
        .stack()
        .apply(pd.Series)
        .reset_index()
        .rename(columns={'level_0': 'person_id', 'level_1': 'category'})
    )

    try:
        spreadsheet_path = os.path.join(
            project_root, 'files/output/', f'{dataset}_data.xlsx'
        )
        mode = 'a' if os.path.exists(spreadsheet_path) else 'w'
        with pd.ExcelWriter(
            spreadsheet_path, engine='openpyxl', mode=mode, if_sheet_exists='replace'
        ) as writer:
            global_df.to_excel(writer, sheet_name='GlobalStats', index=True)
            per_emp_df.to_excel(writer, sheet_name='EmployeeStats', index=False)
    except ValueError:
        print('Too many rows to save spreadsheet')

    return summary


def plot_distance_histograms(dataset: str, embedding_distances: pd.DataFrame):
    print('Plotting distance histograms...')

    id_col_1 = embedding_distances['person_id1']
    id_col_2 = embedding_distances['person_id2']

    same = embedding_distances[id_col_1 == id_col_2]['distance']
    different = embedding_distances[id_col_1 != id_col_2]['distance']

    plt.figure(figsize=(10, 5))
    plt.hist(same, bins=50, alpha=0.6, label='Same Person')
    plt.hist(different, bins=50, alpha=0.6, label='Different Person')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.title(f'Cosine Distance Distributions - {dataset}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    project_root = io_utils.get_project_root()
    plot_path = os.path.join(project_root, 'files/output/', f'{dataset}_distance_histograms.png')
    plt.savefig(plot_path)
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Which data to analyze')
    parser.add_argument('--weights-file', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    weights_file = args.weights_file
    
    if dataset == 'market1501':
        embeddings_filepath, img_data_df = osnet_tests.market1501_extraction(
            weights_file=weights_file
        )
        distances_df = osnet_tests.market1501_embedding_distances(
            embeddings_filepath, img_data_df
        )
    elif dataset == 'event_imgs':
        embeddings_filepath, img_data_df = osnet_tests.event_img_extraction(
            weights_file=weights_file
        )
        distances_df = osnet_tests.event_img_embedding_distances(
            embeddings_filepath, img_data_df
        )

    analyze_embedding_data(dataset, distances_df)
    plot_distance_histograms(dataset, distances_df)
