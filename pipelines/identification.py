# standard dependencies
import os
from typing import Optional
import uuid

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


class IdentificationPipeline:
    def __init__(
            self,
            video_file: str,
            face_data: pd.DataFrame,
            active_trks: dict,
            inactive_trks: dict,
            embeddings_file: Optional[str] = None,
    ):
        # INPUT DATA:
        self.face_data = face_data 

        self.active_trks = active_trks
        self.inactive_trks = inactive_trks

        obs_df, _ = self._format_track_data(active_trks, inactive_trks)
        self.trk_detections = obs_df

        # PATHS/FILENAMES/ETC:
        self.project_root = io_utils.get_project_root()

        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')
        self.event_imgs_dir = os.path.join(self.output_dir, 'event_imgs/')

        self.video_file = video_file
        self.video_path = os.path.join(self.input_dir, video_file)

        self.embeddings_file = embeddings_file or (
            f'{video_file.split(".")[0]}_embeddings.hdf5'
        )
        self.embeddings_path = os.path.join(self.output_dir, self.embeddings_file)

        # STATS/OUTPUT DATA:
        self.embedding_dists = None
        self.face_overlaps = None
        self.track_identities = None

    def run(self, min_overlap=0.3) -> pd.DataFrame:
        if self.embedding_dists is None:
            logger.info("Generating embedding distance matrix...")
            self.embedding_dists = self.embedding_cos_dists()

        if (
            self.face_data is None or self.face_data.empty or
            'identity' not in self.face_data.columns or
            self.face_data['identity'].isna().all()
        ):
            logger.info(
                'No valid face identities found; skipping direct ID assignment.'
            )
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

        indirect_identifications = self.reassociate(direct_identifications)

        track_identities = pd.concat(
            [direct_identifications, indirect_identifications],
            ignore_index=True
        ).drop_duplicates('trk_id')

        self.track_identities = track_identities
        return track_identities

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

                if frame.size == 0:
                    logger.warning(f'Empty frame {f_num}')
                    continue

                x, y, w, h = row['box']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, img_w)
                y2 = min(y + h, img_h)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
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
        
        hdf5_file = hdf5_file or h5py.File(self.embeddings_path, 'r')
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

    def _format_track_data(
            self,
            active_trks: Optional[dict] = None,
            inactive_trks: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, ...]:

        active_trks = active_trks or self.active_trks
        inactive_trks = inactive_trks or self.inactive_trks

        obs_records = []
        state_records = []

        for trk_dict in (active_trks, inactive_trks):
            for trk_id, trk in trk_dict.items():
                # observations (detections):
                for age, bbox in trk.observations.items():
                    f_num = trk.map_offset(offset=age)
                    valid = age in trk.valid_observations
                    box_idx = trk.bbox_indices[age]

                    if len(bbox) != 5:
                        logger.info(f'Detection: {bbox}')
                        continue

                    obs_records.append({
                        'f': f_num,
                        'trk_id': trk_id,
                        'age': age,
                        'box_idx': box_idx,
                        'x': bbox[0],
                        'y': bbox[1],
                        'w': bbox[2] - bbox[0],
                        'h': bbox[3] - bbox[1],
                        'is_valid': 1 if valid else 0,
                    })

                # kalman filter states:
                for t, bbox in enumerate(trk.history):
                    bbox = bbox.flatten()
                    state_records.append({
                        'trk_id': trk_id,
                        't': t,
                        'x': bbox[0],
                        'y': bbox[1],
                        'w': bbox[2] - bbox[0],
                        'h': bbox[3] - bbox[1],
                    })

        obs_df = pd.DataFrame(obs_records)
        state_df = pd.DataFrame(state_records)

        return obs_df, state_df
