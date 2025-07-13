import os
from pathlib import Path

import pandas as pd

from utilities import utils, io_utils
from modules.identification import identify


if __name__ == '__main__':
    project_root = io_utils.get_project_root()
    output_dir = os.path.join(project_root, 'files/output/')

    time_segment = '2025-06-10_09-00-00'

    face_files = sorted(Path(output_dir).glob(f'{time_segment}_*_faces.parquet'))
    face_data = pd.concat([pd.read_parquet(f) for f in face_files], ignore_index=True)
    
    presence_df = pd.read_csv(
        os.path.join(output_dir, f'{time_segment}_presence_summary.csv')
    )

    identity_presence_params = utils.package_id_presence_params(
        match_cutoff          = 0.25,
        mismatch_threshold    = 0.90,
        distance_score_weight = 0.55,
        confidence_weight     = 0.45,
        n_matches             = 1,
        min_score             = 0.45,
        reliability_scale     = 0.75,
        fp_rate               = 0.20,
        presence_prior        = 0.05,
        bias_score_boundary   = 0.70,
        penalty_biases        = (0.5, 1.25),
        decay_window          = 0.9,
        boost_range           = (3.0, 5.0),
        max_decay             = 0.6,
        max_boost             = 0.8,
        boost_per_neighbor    = 0.075,
        fallback_recall_est   = 0.60,
        presence_thresh       = 0.55,
    )

    subsegment_results = identify.subsegment_identity_sweep(
        face_data, presence_df, identity_presence_params,
        full_duration=300, sub_duration=60
    )

    for i, results in subsegment_results.items():
        id_results_paths = [
            f'{time_segment}_{suffix}_{i}.csv'
            for suffix in [
                'presence', 'filtered_faces'
            ]
        ]
        subsegment_presence_df = results[0]
        subsegment_filtered_faces = results[1]

        subsegment_presence_df.to_csv(id_results_paths[0], index=False)
        subsegment_filtered_faces.to_csv(id_results_paths[1], index=False)