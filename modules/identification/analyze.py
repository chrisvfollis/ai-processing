# standard dependencies
from pathlib import Path
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd

# internal dependencies
from utilities import io_utils, log_utils
from pipelines import IdentificationPipeline


logger = log_utils.get_logger(__name__)


def identify_local_tracks(
    face_data, active_trks, inactive_trks, trk_detections, filename, credentials
):
    all_trks = [active_trks, inactive_trks]
    identification = IdentificationPipeline(filename, face_data, *all_trks, trk_detections)

    trk_identity_df = identification.run()

    for _, row in trk_identity_df.iterrows():
        trk_id = row['trk_id']
        for trk_set in all_trks:
            if trk_id in trk_set:
                trk_set[trk_id].identity = row['identity']

    identification.save_id_event_images(
        overlap_threshold=0.5, credentials=credentials
    )
    return active_trks, inactive_trks


def global_identification(
    time_prefix: str,
    output_dir: str,
    # ----- feature-engineering knobs ----------------------------------
    min_match_distance: float = 0.35,
    max_mismatch_distance: float = 0.90,
    confidence_weight: float = 0.40,
    distance_weight: float = 0.50,
    # ----- fusion / filtering knobs -----------------------------------
    n_matches: int = 5,                   # max matches per face detection
    min_score: float = 0.60,
    reliability_scale: float = 0.65,      # α – scales score→success-prob
    fp_rate: float = 0.10,                # β – per-detection false-pos rate
    prior_presence: float = 0.05,         # π – prior P(identity present)
    recall_est: float = 0.65,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    face_files = sorted(Path(output_dir).glob(f'{time_prefix}_*_faces.parquet'))
    trk_files  = sorted(Path(output_dir).glob(f'{time_prefix}_*_trk_dets.parquet'))

    if not face_files:
        raise FileNotFoundError(
            f'No face_data parquet files for prefix {time_prefix} in {output_dir}'
        )

    face_data = pd.concat([pd.read_parquet(f) for f in face_files], ignore_index=True)
    trk_dets  = pd.concat([pd.read_parquet(f) for f in trk_files],  ignore_index=True)

    face_data['confidence'] = face_data['confidence'].clip(0.0, 1.0)

    distance_span = max_mismatch_distance - min_match_distance
    face_data['dist_score'] = np.clip(
        (max_mismatch_distance - face_data['distance']) / distance_span,
        0.0, 1.0,
    )

    weight_values_sum = confidence_weight + distance_weight
    face_data['score'] = (
        confidence_weight * face_data['confidence'] +
        distance_weight * face_data['dist_score']
    ) / weight_values_sum

    face_data = face_data.loc[face_data['score'] >= min_score].copy()
    face_data = face_data[~face_data['identity'].isna() & (face_data['identity'] != '')]

    face_data = (
        face_data.sort_values('distance', ascending=True)
        .groupby(['x', 'y', 'w', 'h', 'f', 'cam_id'], group_keys=False)
        .head(n_matches)
    )

    if face_data.empty:
        logger.info('No valid face detections above score threshold.')
        presence_df = pd.DataFrame(columns=[
            'identity',
            'name',
            'n_detected',
            'max_score',
            'posterior',
            'present_flag',
        ])
        return presence_df, face_data, trk_dets

    face_data['vote_prob'] = reliability_scale * face_data['score']
    track_votes = (
        face_data
        .groupby(['identity'], as_index=False)
        .agg(
            vote_prob=('vote_prob', list),
            max_score=('score', 'max')
        )
    )

    records = []
    log_prior = math.log(prior_presence) - math.log1p(-prior_presence)
    log_beta = math.log(fp_rate)

    n_id_dets = face_data.groupby('identity')['f'].nunique()

    for _, row in track_votes.iterrows():
        ident = row['identity']

        first_name, last_name = io_utils.lookup_name(ident)
        full_name = f'{first_name}_{last_name}'

        p_list = np.clip(row['vote_prob'], 0, 1)            
        n_p = len(p_list)

        if n_p:
            log_lik_absent  = n_p * log_beta       # Σ log β
            fail_probs      = 1.0 - p_list
            log_lik_present = math.log1p(-np.prod(fail_probs))
            log_odds        = log_prior + (log_lik_present - log_lik_absent)
            posterior       = 1.0 / (1.0 + math.exp(-log_odds))
        else:
            # zero detections → prior × recall miss-probability:
            posterior = prior_presence * (1.0 - recall_est)
        
        records.append({
            'identity'     : ident,
            'name'         : full_name,
            'n_detected'   : n_id_dets.get(ident, 0),
            'max_score'    : row['max_score'] if n_p else 0.0,
            'posterior'    : posterior,
            'present_flag' : posterior >= 0.5,
        })

    presence_df = pd.DataFrame(records).sort_values('posterior', ascending=False)

    presence_path = Path(output_dir) / f'{time_prefix}_presence_summary.parquet'
    presence_df.to_parquet(presence_path)

    return presence_df, face_data, trk_dets
