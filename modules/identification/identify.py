# standard dependencies
import os
from typing import Optional
import uuid
from pathlib import Path
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
import h5py
from sklearn.neighbors import NearestNeighbors

# internal dependencies
from utilities import io_utils, log_utils


logger = log_utils.get_logger(__name__)


class LocalID:
    def __init__(
        self,
        video_file: str,
        face_data: pd.DataFrame,
        active_trks: dict,
        inactive_trks: dict,
        trk_detections: pd.DataFrame,
        embeddings_file: Optional[str] = None,
    ):
        # INPUT DATA:
        self.face_data = face_data 

        self.active_trks = active_trks
        self.inactive_trks = inactive_trks

        self.trk_detections = trk_detections

        # PATHS/FILENAMES/ETC:
        self.project_root = io_utils.get_project_root()

        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')
        self.event_imgs_dir = os.path.join(self.output_dir, 'event_imgs/')

        self.video_file = video_file
        self.video_path = os.path.join(self.input_dir, video_file)

        try:
            self.embeddings_file = embeddings_file or (
                f'{video_file.split(".")[0]}_embeddings.hdf5'
            )
            self.embeddings_path = os.path.join(self.output_dir, self.embeddings_file)
        except FileNotFoundError:
            self.embeddings_file = None
            self.embeddings_path = None

        # STATS/OUTPUT DATA:
        self.embedding_dists = None
        self.face_overlaps = None
        self.track_identities = None

    def run(self, min_overlap=0.3) -> pd.DataFrame:
        if self.embedding_dists is None:
            self.embedding_dists = self.embedding_cos_dists()

        if (
            self.face_data is None or self.face_data.empty or
            'identity' not in self.face_data.columns or
            self.face_data['identity'].isna().all()
        ):
            direct_identifications = pd.DataFrame(
                columns=['trk_id', 'identity', 'assignment_type', 'assignment_cost']
            )
        else:
            face_overlap_df = self.face_overlap_ratios()
            face_match_candidates = self._collect_face_match_candidates(face_overlap_df)
            face_match_candidates = face_match_candidates[
                face_match_candidates['overlap_ratio'] >= min_overlap
            ]
            direct_identifications = self.assign_identities(face_match_candidates)

        if (self.embedding_dists is None) or self.embedding_dists.empty:
            self.track_identities = direct_identifications
        else:
            indirect_identifications = self.reassociate(direct_identifications)

            self.track_identities = pd.concat(
                [direct_identifications, indirect_identifications],
                ignore_index=True
            ).drop_duplicates('trk_id')
            
        return self.track_identities

    def save_id_event_images(
            self,
            face_overlap_df: pd.DataFrame = None,
            overlap_threshold: float = 0.3,
            credentials: tuple[str] = None,
    ):
        logger.info('Saving ID event images...')

        if face_overlap_df is None:
            face_overlap_df = self.face_overlaps

        has_overlap_data = (face_overlap_df is not None) and (not face_overlap_df.empty)
        high_overlaps = face_overlap_df[face_overlap_df['overlap_ratio'] >= overlap_threshold] if has_overlap_data else pd.DataFrame()

        all_trks = self.active_trks | self.inactive_trks
        if self.trk_detections is None or self.trk_detections.empty:
            return 
        trks_df = self.trk_detections.groupby('trk_id')

        frames = []
        for trk_id, grp in trks_df:
            trk_faces = high_overlaps[high_overlaps['trk_id'] == trk_id] if has_overlap_data else pd.DataFrame()

            if not trk_faces.empty:
                use_both = len(trk_faces) >= 2
                f_first = trk_faces['f'].min() if use_both else grp['f'].min()
                f_last = trk_faces['f'].max()
            else:
                frame_width = 3840
                frame_height = 2160
                valid_boxes = grp[
                    (grp['x'] + grp['w'] > 0) &
                    (grp['y'] + grp['h'] > 0) &
                    (grp['x'] < frame_width) &
                    (grp['y'] < frame_height)
                ]
                if not valid_boxes.empty:
                    f_first = valid_boxes['f'].min()
                    f_last = valid_boxes['f'].max()
                else:
                    continue

            for i, frame in enumerate([f_first, f_last]):
                row = grp[grp['f'] == frame].iloc[0]
                frames.append({
                    'trk_id': trk_id,
                    'f': int(frame),
                    'box': (int(row['x']), int(row['y']), int(row['w']), int(row['h'])),
                    'image': all_trks[trk_id].id_event_images[i],
                })

        img_df = pd.DataFrame(frames).sort_values(by='f')

        try:
            cap = cv2.VideoCapture(self.video_path)
            img_w = 3840
            img_h = 2160
            f_prev = None

            for _, row in img_df.iterrows():
                f_num, img_name = row[['f', 'image']]
                if f_num != f_prev:
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
                        ret, frame = cap.read()
                        if not ret:
                            logger.warning(f'Frame {f_num} unreadable')
                            continue
                    finally:
                        f_prev = f_num

                if frame is None or frame.size == 0:
                    logger.warning(f'Empty frame {f_num}')
                    continue

                x, y, w, h = row['box']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, img_w)
                y2 = min(y + h, img_h)
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    logger.warning(f'Empty crop at frame {f_num} for box {row["box"]}')
                    continue

                io_utils.save_event_image(
                    img=crop,
                    object_key=img_name,
                    credentials=credentials,
                    event_imgs_dir=self.event_imgs_dir
                )
        finally:
            cap.release()

    def assign_identities(self, face_match_candidates: pd.DataFrame) -> pd.DataFrame:
        '''Direct identification via facial data'''
        scores = self._track_identity_scores(face_match_candidates)
        top_scores = (
            scores.sort_values(['trk_id', 'overlap_weighted_avg_sim'], ascending=[True, False])
            .groupby('trk_id')
            .first()
            .reset_index()
        )

        top_scores['assignment_type'] = 'direct'
        top_scores['assignment_cost'] = 1 - top_scores['overlap_weighted_avg_sim']

        keep_cols = [
            'trk_id',
            'identity',
            'assignment_type',
            'assignment_cost',
        ]
        identities_df = top_scores[keep_cols]

        return identities_df

    def reassociate(self, initial_identities: pd.DataFrame, k=5) -> pd.DataFrame:
        '''
        Indirect identification by matching unidentified tracks to high-affinity
        tracks that have been identified, excluding temporally overlapping tracks.
        Also links together unknown tracks with similar embeddings using synthetic UUIDs.
        '''
        if self.trk_detections.empty:
            return pd.DataFrame()

        knn_df = self.track_similarity_knn(k=k)
        if knn_df.empty:
            return pd.DataFrame()

        identity_map = dict(initial_identities[['trk_id', 'identity']].values)

        trk_to_frames = (
            self.trk_detections.groupby('trk_id')['f']
            .apply(set)
            .to_dict()
        )

        reassigned = []
        synthetic_id_map = {}

        for trk_id in knn_df['trk_id'].unique():
            if trk_id in identity_map:
                continue  # already has identity

            neighbors = knn_df[knn_df['trk_id'] == trk_id]
            trk_frames = trk_to_frames.get(trk_id, set())

            valid_neighbors = [
                row for _, row in neighbors.iterrows()
                if trk_frames.isdisjoint(trk_to_frames.get(row['neighbor_trk_id'], set()))
            ]

            neighbor_identities = [
                identity_map.get(row['neighbor_trk_id']) for row in valid_neighbors
            ]
            votes = pd.Series(neighbor_identities).dropna()

            if not votes.empty:
                identity = votes.mode().iloc[0]

                for row in valid_neighbors:
                    if identity_map.get(row['neighbor_trk_id']) == identity:
                        reassigned.append({
                            'trk_id': trk_id,
                            'identity': identity,
                            'assignment_type': 'indirect',
                            'assignment_cost': row['distance'],
                        })
                        break
            else:
                # no neighbors with identity; try synthetic linking:
                linked_ids = []
                for row in valid_neighbors:
                    nid = row['neighbor_trk_id']
                    if nid in synthetic_id_map:
                        linked_ids.append(synthetic_id_map[nid])

                if linked_ids:
                    synthetic_identity = linked_ids[0]
                else:
                    synthetic_identity = str(uuid.uuid4())

                synthetic_id_map[trk_id] = synthetic_identity
                reassigned.append({
                    'trk_id': trk_id,
                    'identity': synthetic_identity,
                    'assignment_type': 'synthetic',
                    'assignment_cost': valid_neighbors[0]['distance'] if valid_neighbors else 1.0,
                })

                for row in valid_neighbors:
                    nid = row['neighbor_trk_id']
                    if nid not in identity_map and nid not in synthetic_id_map:
                        synthetic_id_map[nid] = synthetic_identity
                        reassigned.append({
                            'trk_id': nid,
                            'identity': synthetic_identity,
                            'assignment_type': 'synthetic',
                            'assignment_cost': row['distance'],
                        })
        return pd.DataFrame(reassigned)

    def track_similarity_knn(
            self,
            k: int = 5,
            embedding_dists: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if embedding_dists is None:
            dists_df = self.embedding_dists
        else:
            dists_df = embedding_dists
        
        if 'trk_id1' not in dists_df.columns or 'trk_id2' not in dists_df.columns:
            logger.error(f'Missing `trk_id1` & `trk_id2` cols in embedding distances df. Columns found: {dists_df.columns.tolist()}')
            return pd.DataFrame() 

        dists_df = dists_df.dropna(subset=['trk_id1', 'trk_id2'])
        dists_df = dists_df[dists_df['trk_id1'] != dists_df['trk_id2']]

        avg_dists = (
            dists_df.groupby(['trk_id1', 'trk_id2'])['distance']
            .mean()
            .reset_index()
        )

        nan_rows = avg_dists[avg_dists['distance'].isna()]
        if not nan_rows.empty:
            logger.info(f'Found {len(nan_rows)} NaN distances in avg_dists. ' +
                        'These may indicate missing metadata or embeddings.')
            logger.debug(nan_rows)

        avg_dists['distance'] = avg_dists['distance'].fillna(1.0)

        all_ids = pd.unique(avg_dists[['trk_id1', 'trk_id2']].values.ravel())
        full_index = pd.MultiIndex.from_product([all_ids, all_ids], names=['trk_id1', 'trk_id2'])
        avg_dists = avg_dists.set_index(['trk_id1', 'trk_id2']).reindex(full_index).fillna(1.0).reset_index()

        pivot = avg_dists.pivot(index='trk_id1', columns='trk_id2', values='distance')
        pivot = pivot.reindex(index=all_ids, columns=all_ids, fill_value=1.0)

        dist_matrix = pivot.to_numpy()
        trk_ids = pivot.index.to_numpy()

        if np.isnan(dist_matrix).any():
            raise ValueError('Distance matrix still contains NaNs after cleanup.')

        dist_matrix = pivot.to_numpy()
        n_samples = dist_matrix.shape[0]

        k = min(k, n_samples)
        if k == 0:
            return pd.DataFrame()
        
        nn = NearestNeighbors(n_neighbors=k, metric='precomputed')
        nn.fit(dist_matrix)
        distances, indices = nn.kneighbors(dist_matrix)

        rows = []
        for i, trk_id in enumerate(trk_ids):
            for j in range(k):
                neighbor_trk_id = trk_ids[indices[i][j]]
                distance = distances[i][j]
                rows.append({
                    'trk_id': trk_id,
                    'neighbor_rank': j + 1,
                    'neighbor_trk_id': neighbor_trk_id,
                    'distance': distance,
                })
        knn_df = pd.DataFrame(rows)

        return knn_df

    def embedding_cos_dists(
            self,
            hdf5_file: Optional[str] = None,
            detections: Optional[pd.DataFrame] = None,
            chunk_size: int = 100,
    ) -> pd.DataFrame:
        distance_data = []
        
        try:
            hdf5_file = hdf5_file or h5py.File(self.embeddings_path, 'r')
            logger.info('Generating embedding distance matrix...')
        except FileNotFoundError:
            return pd.DataFrame()
        
        if detections is None:
            detections = self.trk_detections

        num_embeddings = hdf5_file['embeddings'].shape[0]
        frames = hdf5_file['frames'][:]
        box_indices = hdf5_file['box_indices'][:]

        # create a mapping from (frame, box_idx) to metadata:
        valid_keys = set(zip(frames.astype(int), box_indices.astype(int)))
        obs_df = detections.copy()
        if obs_df.empty or 'f' not in obs_df.columns or 'box_idx' not in obs_df.columns:
            logger.warning(
                f'Track observations dataframe missing required columns: {obs_df.columns}'
            )
            return pd.DataFrame() 
        obs_df = obs_df.astype({'f': int, 'box_idx': int})
        obs_df['key'] = list(zip(obs_df['f'], obs_df['box_idx']))
        obs_df = obs_df[obs_df['key'].isin(valid_keys)]
        metadata_map = obs_df.set_index('key').to_dict('index')

        logger.info(f'Calculating cosine distances...')
        for start_idx in range(0, num_embeddings, chunk_size):
            end_idx = min(start_idx + chunk_size, num_embeddings)

            current_chunk_np = hdf5_file['embeddings'][start_idx:end_idx]
            current_chunk = torch.from_numpy(current_chunk_np).float()

            for i in range(len(current_chunk)):
                for j in range(i + 1, len(current_chunk)):
                    sim = F.cosine_similarity(
                        current_chunk[i], current_chunk[j], dim=0
                    ).item()
                    distance = 1 - sim

                    key_i = (frames[i], box_indices[i])
                    key_j = (frames[j], box_indices[j])

                    meta_i = metadata_map.get(key_i, {})
                    meta_j = metadata_map.get(key_j, {})

                    trk_id1 = meta_i.get('trk_id')
                    trk_id2 = meta_j.get('trk_id')

                    if (trk_id1 is None) or (trk_id2 is None):
                        continue

                    distance_data.append({
                        'frame1': frames[i],
                        'box_idx1': box_indices[i],
                        'trk_id1': trk_id1,
                        'frame2': frames[j],
                        'box_idx2': box_indices[j],
                        'trk_id2': trk_id2,
                        'distance': distance,
                    })

            for next_idx in range(end_idx, num_embeddings):
                next_embedding_np = hdf5_file['embeddings'][next_idx]
                next_embedding = (
                    torch.from_numpy(next_embedding_np).float()
                    .unsqueeze(0)
                )

                sims = F.cosine_similarity(current_chunk, next_embedding, dim=1)
                distances = 1 - sims

                for i in range(len(current_chunk)):
                    distance = distances[i].item()

                    key_i = (frames[start_idx + i], box_indices[start_idx + i])
                    key_j = (frames[next_idx], box_indices[next_idx])

                    meta_i = metadata_map.get(key_i, {})
                    meta_j = metadata_map.get(key_j, {})

                    trk_id1 = meta_i.get('trk_id')
                    trk_id2 = meta_j.get('trk_id')

                    if (trk_id1 is None) or (trk_id2 is None):
                        continue

                    distance_data.append({
                        'frame1': frames[start_idx + i],
                        'box_idx1': box_indices[start_idx + i],
                        'trk_id1': trk_id1,
                        'frame2': frames[next_idx],
                        'box_idx2': box_indices[next_idx],
                        'trk_id2': trk_id2,
                        'distance': distance,
                    })

        hdf5_file.flush()
        hdf5_file.close()

        return pd.DataFrame(distance_data)

    def face_overlap_ratios(self) -> pd.DataFrame:
        def _overlap_ratio(row):
            xA = max(row['x1_face'], row['x1_trk'])
            yA = max(row['y1_face'], row['y1_trk'])
            xB = min(row['x2_face'], row['x2_trk'])
            yB = min(row['y2_face'], row['y2_trk'])

            inter_area = max(0, xB - xA) * max(0, yB - yA)
            face_area = (row['x2_face'] - row['x1_face']) * (row['y2_face'] - row['y1_face'])
            return inter_area / (face_area + 1e-6)

        face_df = (
            self.face_data.groupby(['f', 'face_idx'], as_index=False).first()
            .assign(x1_face=lambda df: df['x'],
                    y1_face=lambda df: df['y'],
                    x2_face=lambda df: df['x'] + df['w'],
                    y2_face=lambda df: df['y'] + df['h'])
        )
        trk = (
            self.trk_detections.assign(
                x1_trk=lambda df: df['x'],
                y1_trk=lambda df: df['y'],
                x2_trk=lambda df: df['x'] + df['w'],
                y2_trk=lambda df: df['y'] + df['h'])
        )
        merged = face_df.merge(trk, on='f')
        merged['overlap_ratio'] = merged.apply(_overlap_ratio, axis=1)
        keep_cols = [
            'f',
            'face_idx',
            'trk_id',
            'overlap_ratio',
        ]
        self.face_overlaps = merged[keep_cols]

        return self.face_overlaps

    def _collect_face_match_candidates(
            self, face_overlap_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        '''
        Collects candidate face-to-track associations for recognized faces.
        
        For each detected face with one or more matching identities, this method
        identifies all track detections in the same frame and computes their
        bounding box overlap. The result is a set of candidate associations, not
        hard assignments.

        Each row in the output represents:
        - One candidate identity for a recognized face
        - One overlapping track detection in the same frame
        - Their associated overlap and face-identity cosine distance

        Returns:
            pd.DataFrame with columns:
                - f: frame number
                - trk_id: track ID
                - identity: candidate identity for the face detection
                - distance: identity cosine distance
                - overlap: spatial overlap between face detection and track box
        '''
        if face_overlap_df is None:
            face_overlap_df = self.face_overlaps

        merged = self.face_data.merge(
            face_overlap_df,
            on=['f', 'face_idx'],
            how='inner',
        )
        merged = merged.dropna(subset=['identity'])
        keep_cols = [
            'f',
            'trk_id',
            'identity',
            'distance',
            'overlap_ratio',
        ]
        face_match_candidates = merged[keep_cols]
        return face_match_candidates

    def _track_identity_scores(
            self, face_match_candidates: pd.DataFrame
    ) -> pd.DataFrame:
        df = face_match_candidates.copy()
        df['similarity'] = 1 - df['distance']
        df['weighted_sim'] = df['similarity'] * df['overlap_ratio']

        grouped = (
            df.groupby(['trk_id', 'identity'])
            .agg(
                weighted_score=('weighted_sim', 'sum'),
                total_overlap=('overlap_ratio', 'sum'),
                count=('identity', 'count'),
            )
            .reset_index()
        )
        
        grouped['overlap_weighted_avg_sim'] = grouped['weighted_score'] / (grouped['total_overlap'] + 1e-6)
        return grouped.sort_values(by=['trk_id', 'overlap_weighted_avg_sim'], ascending=[True, False])


def identify_local_tracks(
    face_data, active_trks, inactive_trks, trk_detections, filename, credentials
):
    all_trks = [active_trks, inactive_trks]
    identification = LocalID(filename, face_data, *all_trks, trk_detections)

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

    face_files = sorted(Path(output_dir).glob(f'{time_prefix}_*_faces.parquet'))
    trk_files  = sorted(Path(output_dir).glob(f'{time_prefix}_*_trk_dets.parquet'))

    if not face_files:
        raise FileNotFoundError(
            f'No face_data parquet files for prefix {time_prefix} in {output_dir}'
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
            output_dir, f'{time_prefix}_presence_summary.parquet'
        ))

    return presence_df, face_data, trk_dets
