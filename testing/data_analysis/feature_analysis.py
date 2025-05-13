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
from testing import test_fxs, test_utils


def analyze_embedding_data(embedding_distances: pd.DataFrame) -> dict:
    summary = {}

    # Separate distances by same vs different employee IDs
    same_employee = embedding_distances[
        embedding_distances['employee_id1'] == embedding_distances['employee_id2']
    ]
    different_employee = embedding_distances[
        embedding_distances['employee_id1'] != embedding_distances['employee_id2']
    ]

    # Helper function to calculate basic statistics
    def calculate_stats(distances):
        return {
            'count': len(distances),
            'mean': np.mean(distances),
            'median': np.median(distances),
            'std': np.std(distances),
            'min': np.min(distances),
            'max': np.max(distances)
        }

    # Global statistics
    summary['global'] = {
        'same_employee': calculate_stats(same_employee['distance']),
        'different_employee': calculate_stats(different_employee['distance']),
        'overall': calculate_stats(embedding_distances['distance'])
    }

    # Per-employee statistics
    per_employee_stats = {}
    for employee_id in pd.concat([embedding_distances['employee_id1'], embedding_distances['employee_id2']]).unique():
        # Filter distances involving this employee
        employee_distances = embedding_distances[
            (embedding_distances['employee_id1'] == employee_id) |
            (embedding_distances['employee_id2'] == employee_id)
        ]
        same = employee_distances[
            employee_distances['employee_id1'] == employee_distances['employee_id2']
        ]
        different = employee_distances[
            employee_distances['employee_id1'] != employee_distances['employee_id2']
        ]

        per_employee_stats[employee_id] = {
            'same_employee': calculate_stats(same['distance']),
            'different_employee': calculate_stats(different['distance']),
            'overall': calculate_stats(employee_distances['distance'])
        }

    summary['per_employee'] = per_employee_stats

    summary_df = pd.DataFrame.from_dict(summary, orient='index')

    return summary_df
