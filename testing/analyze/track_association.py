# standard dependencies
import os
import argparse
import ast
from typing import Optional

# 3rd-party dependencies
import numpy as np
import pandas as pd

# internal dependencies
from utilities import io_utils


def parse_dissimilarity_data(
    filename: str,
    sheet_name: str = 'Association Data',
    column: str = 'dissimilarity_costs_raw'
) -> pd.DataFrame:
    '''
    Loads an Excel file and parses a column of stringified lists representing
    matching cost data.
    '''
    project_root = io_utils.get_project_root()
    file_path = os.path.join(project_root, 'files/output/runtime_data/', filename)

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    parsed_costs = []
    for val in df[column]:
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and isinstance(parsed[0], list):
                parsed_costs.append(parsed)
            else:
                parsed_costs.append([parsed])
        except Exception:
            parsed_costs.append([])

    parsed_df = pd.DataFrame({
        'frame': df['frame'],
        'track_id': df['track_id'],
        f'{column}_parsed': parsed_costs
    })

    return parsed_df


def compute_dissimilarity_spread(parsed_df, column='dissimilarity_costs_raw'):
    spreads = []

    for cost_matrix in parsed_df[f'{column}_parsed']:
        if not cost_matrix:
            continue
        try:
            matrix = np.array(cost_matrix)
            for col in matrix.T:
                spreads.append(np.max(col) - np.min(col))
        except Exception:
            continue

    if len(spreads) == 0:
        return float('nan')
    
    return sum(spreads) / len(spreads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str)

    parser.add_argument('--features', action='store_true')
    parser.add_argument('--cost-val-type', type=str)

    args = parser.parse_args()

    file = args.file
    if args.features:
        cost_val_type = args.cost_val_type or 'raw'

        if cost_val_type == 'raw':
            column = 'dissimilarity_costs_raw'
        elif cost_val_type == 'normalized':
            column = 'dissimilarity_costs'

        parsed_df = parse_dissimilarity_data(file, column=column)
        avg_spread = compute_dissimilarity_spread(parsed_df, column=column)

        print(avg_spread)
