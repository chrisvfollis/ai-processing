# standard dependencies
import os
from typing import Optional
from pathlib import Path
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd

# internal dependencies
from utilities import io_utils, log_utils


logger = log_utils.get_logger(__name__)


def assess_present_identities(
    time_segment: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    # ----- feature-engineering knobs ----------------------------------
    match_cutoff: float = 0.25,
    mismatch_threshold: float = 0.90,
    distance_score_weight: float = 0.55,
    confidence_weight: float = 0.45,
    # ----- fusion / filtering knobs -----------------------------------
    n_matches: int = 1,                 # max ID matches per face detection
    min_score: float = 0.45,
    reliability_scale: float = 0.75,    # α – scales `score` -> success probability
    fp_rate: float = 0.20,              # β – per-detection false positive rate
    presence_prior: float = 0.05,       # π – assumed prior for P(identity present)
    bias_score_boundary: float = 0.70,
    penalty_biases: tuple[float, float] = (0.50, 1.25),
    # ----- temporal weighting knobs -----------------------------------
    decay_window: float = 0.9,                      # seconds
    boost_range: tuple[float, float] = (3.0, 5.0),  # seconds (range)
    max_decay: float = 0.6,
    max_boost: float = 0.8,
    boost_per_neighbor: float = 0.075,
    # ----- final results knobs ----------------------------------------
    fallback_recall_est: float = 0.60,
    presence_thresh: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Determines which identities were present in the work zone within a given
    time segment by assessing the combined inference/tracking/etc data from
    all available camera footage for that time.
    '''
    def _empty_presence_df():
        presence_df = pd.DataFrame(
            columns=[
                'identity',
                'name',
                'n_detections',
                'max_score',
                'posterior',
                'present_flag',
            ]
        )
        return presence_df

    project_root = io_utils.get_project_root()
    output_dir = os.path.join(project_root, 'files/output/')

    face_files = sorted(Path(output_dir).glob(f'{time_segment}_*_faces.parquet'))
    trk_files  = sorted(Path(output_dir).glob(f'{time_segment}_*_trk_dets.parquet'))

    if not face_files:
        raise FileNotFoundError(
            f'No face data files for {time_segment} in {output_dir}'
        )

    face_data = pd.concat([pd.read_parquet(f) for f in face_files], ignore_index=True)
    trk_dets  = pd.concat([pd.read_parquet(f) for f in trk_files],  ignore_index=True)

    # retain only the top `n_matches` row(s) per detection:
    face_data = (
        face_data.sort_values('distance', ascending=True)
        .groupby(['x', 'y', 'w', 'h', 'f', 'cam_id'], group_keys=False)
        .head(n_matches)
    )

    # feature engineer the `dist_score` and `score` columns:
    distance_span = mismatch_threshold - match_cutoff
    face_data['dist_score'] = (
        mismatch_threshold - face_data['distance']
    ) / distance_span

    face_data['dist_score'] = face_data['dist_score'].clip(0.0, 1.0)
    face_data['confidence'] = face_data['confidence'].clip(0.0, 1.0)

    weight_values_sum = confidence_weight + distance_score_weight
    face_data['score'] = (
        (face_data['dist_score'] * distance_score_weight) +
        (face_data['confidence'] * confidence_weight)
    ) / weight_values_sum
    
    # filter face detections below `min_score`:
    face_data = face_data.loc[face_data['score'] >= min_score].copy()
    face_data = face_data[
        (~face_data['identity'].isna()) & (face_data['identity'] != '')
    ]
    if face_data.empty:
        presence_df = _empty_presence_df()
        logger.info('No face dets above `min_score` threshold')
        return presence_df, face_data, trk_dets
    else:
        id_det_counts = (
            face_data.drop_duplicates(
                subset=['identity', 'cam_id', 'f'], keep='first'
            )
            .groupby('identity')
            .size()
        )

    def _summarize_vote_signals(face_data):
        vote_signal_summary = (
            face_data['vote_signal'].describe()
            .apply(
                lambda x: round(float(x), 3)
            )
        )
        return vote_signal_summary

    def _apply_temporal_weighting(
        face_data: pd.DataFrame,
        decay_window: float,
        boost_range: tuple[float, float],
        max_decay: float,
        max_boost: float,
        boost_per_neighbor: float,
    ) -> pd.DataFrame:
        '''
        Applies temporal weighting in order to:
            1. Decay the influence of detections with the same identity from
                short bursts of similar output across nearby frames. This is
                essentially redundant, low-signal data that distorts the final
                results.
            2. Boost the influence of detections with the same identity that
                are moderately spaced in time. Close enough that it's plausible
                they really are the same person, but far enough that their
                orientation's likely different. This helps rule out false
                positives by confirming the same identity from multiple angles/etc.
        '''
        weighted_votes = []

        for (_, cam_id), group in face_data.groupby(['identity', 'cam_id']):
            times = group['s'].values
            scores = group['vote_signal'].values
            idxs = group.index

            n = len(times)
            weighted_scores = np.zeros(n)

            for i in range(n):
                t_i = times[i]
                diffs = np.abs(times - t_i)

                # decay:
                decay_factors = np.exp(
                    -np.square(diffs) / decay_window**2     # Gaussian kernel (Radial Basis Function)
                )
                local_clump_penalty = decay_factors.sum() - 1  # exclude self
                decay_weight = 1.0 - min(max_decay, 0.4 * local_clump_penalty)

                # boost:
                in_boost_range = (diffs >= boost_range[0]) & (diffs <= boost_range[1])
                boost_neighbors = in_boost_range.sum()
                boost_weight = 1.0 + min(max_boost, boost_per_neighbor * boost_neighbors)

                temporal_weight = decay_weight * boost_weight
                weighted_scores[i] = scores[i] * temporal_weight

            weighted_votes.extend(zip(idxs, weighted_scores))

        weighted_votes.sort()
        for idx, new_val in weighted_votes:
            face_data.at[idx, 'vote_signal'] = new_val

        return face_data
    
    # compute `vote_signal` column values:
    face_data['vote_signal'] = reliability_scale * face_data['score']

    raw_vote_signal_summary = _summarize_vote_signals(face_data)
    logger.info(f'Raw `vote_signal` summary: \n{raw_vote_signal_summary}')

    face_data = _apply_temporal_weighting(
        face_data,
        decay_window,
        boost_range,
        max_decay,
        max_boost,
        boost_per_neighbor,
    )
    vote_signal_summary = _summarize_vote_signals(face_data)
    logger.info(f'Final `vote_signal` summary: {vote_signal_summary}')

    # aggregate votes per identity:
    track_votes = (
        face_data
        .groupby(['identity'], as_index=False)
        .agg(
            vote_signal=('vote_signal', list),
            max_score=('score', 'max')
        )
    )
    if track_votes.empty:
        presence_df = _empty_presence_df()
        logger.info(
            'No track votes could be formed — skipping presence estimation'
        )
        return presence_df, face_data, trk_dets

    # convert prior to log-odds space so it can combine linearly with evidence:
    log_prior = math.log(presence_prior) - math.log1p(-presence_prior)

    records = []
    for _, row in track_votes.iterrows():
        identity = row['identity']

        obs_vote_signals = np.clip(row['vote_signal'], 0.0, 1.0)
        n_obs = len(obs_vote_signals)

        if n_obs:
            total_vote_support = sum(obs_vote_signals)
            penalty = n_obs * fp_rate

            if row['max_score'] >= bias_score_boundary:
                penalty *= min(penalty_biases)          # favorable bias
            else:
                penalty *= max(penalty_biases)          # high penalty
            
            log_odds = log_prior + (total_vote_support - penalty)
            # convert `log_odds` back into a probability between 0 and 1 using
            # the logistic function:
            posterior = 1.0 / (1.0 + math.exp(-log_odds))
        else:
            # zero detections -> prior x recall miss-probability:
            posterior = presence_prior * (1.0 - fallback_recall_est)

        records.append({
            'identity'     : identity,
            'name'         : '_'.join(io_utils.lookup_name(identity)),
            'n_detections' : id_det_counts[identity],
            'max_score'    : row['max_score'] if n_obs else 0.0,
            'posterior'    : posterior,
            'present_flag' : posterior >= presence_thresh,
        })

    if not records:
        presence_df = _empty_presence_df()
        logger.info('No face detections above score threshold')
    else:
        presence_df = pd.DataFrame(records).sort_values('posterior', ascending=False)
        presence_df.to_parquet(os.path.join(
            output_dir, f'{time_segment}_presence_summary.parquet'
        ))

    return presence_df, face_data, trk_dets
