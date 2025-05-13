# standard dependencies
import os
import argparse
import sys
from datetime import datetime
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# internal dependencies
from testing import test_fxs
from utilities import io_utils


def analyze_embedding_data(embedding_distances: pd.DataFrame) -> dict:
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
    
    project_root = io_utils.get_project_root()
    stats_spreadsheet_path = os.path.join(
        project_root, 'files/output/', 'cos_distances_stats.xlsx'
    )

    summary = {}

    id_col_1a = embedding_distances['employee_id1']
    id_col_2a = embedding_distances['employee_id2']

    same_employee = embedding_distances[id_col_1a == id_col_2a]
    different_employee = embedding_distances[id_col_1a != id_col_2a]

    summary['global'] = _calculate_summary_stats(
        same_employee['distance'], different_employee['distance'],
        embedding_distances['distance'],
    )

    per_employee_stats = {}
    for employee_id in pd.concat([id_col_1a,id_col_2a]).unique():
        employee_distances = embedding_distances[
            (id_col_1a == employee_id) |
            (id_col_2a == employee_id)
        ]

        id_col_1b = employee_distances['employee_id1']
        id_col_2b = employee_distances['employee_id2']

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
        .rename(columns={'level_0': 'employee_id', 'level_1': 'category'})
    )

    with pd.ExcelWriter(stats_spreadsheet_path, engine='xlsxwriter') as writer:
        global_df.to_excel(writer, sheet_name='Global Summary', index=True)
        per_emp_df.to_excel(writer, sheet_name='Per Employee Summary', index=False)

    return summary


if __name__ == '__main__':
    pass
