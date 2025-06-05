# standard dependencies
import os
import pickle
import math
from itertools import permutations, islice
import sys

# 3rd-party dependencies
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
from sklearn.neighbors import NearestNeighbors

# internal dependencies
from utilities import io_utils, log_utils
from utilities import general_utils as utils


logger = log_utils.get_logger(__name__)


class IdentificationPipeline:
    def __init__(
            self,
            video_file: str,
            face_data: pd.DataFrame,
            track_detections: pd.DataFrame,
            embeddings_file: str = None,
        ):
        # INPUT DATA:
        self.face_data = face_data
        self.trk_detections = track_detections

        # PATHS/FILENAMES/ETC:
        self.project_root = io_utils.get_project_root()
        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')

        self.video_file = video_file
        self.video_path = os.path.join(self.input_dir, video_file)

        self.embeddings_file = embeddings_file or (
            f'{video_file.split(".")[0]}_embeddings.hdf5'
        )
        self.embeddings_path = os.path.join(self.output_dir, self.embeddings_file)

        # STATS/OUTPUT DATA:
        self.embedding_distances = None
        self.face_ious = None


    def embedding_cos_dists(
            self,
            hdf5_file: str = None,
            detections: pd.DataFrame = None,
            chunk_size: int = 100,
        ) -> pd.DataFrame:
        distance_data = []
        
        hdf5_file = hdf5_file or h5py.File(self.embeddings_path, 'r')
        detections = detections or self.trk_detections

        num_embeddings = hdf5_file['embeddings'].shape[0]
        frames = hdf5_file['frames'][:]
        box_indices = hdf5_file['box_indices'][:]

        # create a mapping from (frame, box_idx) to metadata:
        obs_df = detections.copy()
        obs_df['key'] = list(zip(obs_df['f'], obs_df['box_idx']))
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

                    distance_data.append({
                        'frame1': frames[i],
                        'box_idx1': box_indices[i],
                        'trk_id1': meta_i.get('trk_id'),
                        'frame2': frames[j],
                        'box_idx2': box_indices[j],
                        'trk_id2': meta_j.get('trk_id'),
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

                    distance_data.append({
                        'frame1': frames[start_idx + i],
                        'box_idx1': box_indices[start_idx + i],
                        'trk_id1': meta_i.get('trk_id'),
                        'frame2': frames[next_idx],
                        'box_idx2': box_indices[next_idx],
                        'trk_id2': meta_j.get('trk_id'),
                        'distance': distance,
                    })

        hdf5_file.flush()
        hdf5_file.close()

        return pd.DataFrame(distance_data)

    def face_ious(self) -> pd.DataFrame:
        def _compute_iou(row):
            xA = max(row['x1_face'], row['x1_trk'])
            yA = max(row['y1_face'], row['y1_trk'])
            xB = min(row['x2_face'], row['x2_trk'])
            yB = min(row['y2_face'], row['y2_trk'])

            inter_area = max(0, xB - xA) * max(0, yB - yA)

            box_area_face = (row['x2_face'] - row['x1_face']) * (row['y2_face'] - row['y1_face'])
            box_area_trk = (row['x2_trk'] - row['x1_trk']) * (row['y2_trk'] - row['y1_trk'])

            iou = inter_area / float(box_area_face + box_area_trk - inter_area + 1e-6)
            return iou

        face_df = self.face_data.copy()
        # one row per face detection:
        face_df = (
            face_df.groupby(['f', 'face_idx'], as_index=False)
            .first()
        )
        face_df['x1'] = face_df['x']
        face_df['y1'] = face_df['y']
        face_df['x2'] = face_df['x'] + face_df['w']
        face_df['y2'] = face_df['y'] + face_df['h']

        merged_df = face_df.merge(
            self.trk_detections,
            on='f',
            suffixes=('_face', '_trk'),
        )
        merged_df['iou'] = merged_df.apply(_compute_iou, axis=1)

        keep_cols = [
            'f',
            'face_idx'
            'trk_id',
            'iou',
        ]
        return merged_df[keep_cols]

    def face_cos_dists(self) -> pd.DataFrame:
        face_ious_df = self.face_ious()

        merged = self.face_data.merge(
            face_ious_df,
            on=['f', 'face_idx'],
            how='inner',
        )

        merged = merged.dropna(subset=['identity'])
        keep_cols = [
            'f',
            'trk_id',
            'identity',
            'distance',
            'iou',
        ]
        return merged[keep_cols]

    def track_identity_scores(self, face_identity_df: pd.DataFrame) -> pd.DataFrame:
        df = face_identity_df.copy()
        df['similarity'] = 1 - df['distance']
        df['weighted_sim'] = df['similarity'] * df['iou']

        grouped = (
            df.groupby(['trk_id', 'identity'])
            .agg(
                weighted_score=('weighted_sim', 'sum'),
                total_iou=('iou', 'sum'),
                count=('identity', 'count'),
            )
            .reset_index()
        )
        
        grouped['iou_weighted_avg_sim'] = grouped['weighted_score'] / (grouped['total_iou'] + 1e-6)
        return grouped.sort_values(by=['trk_id', 'iou_weighted_avg_sim'], ascending=[True, False])


    def knn_track_embeddings(self, dists = None, k: int = 5) -> pd.DataFrame:
        dists_df = dists or self.embedding_distances

        dists_df = dists_df.dropna(subset=['trk_id1', 'trk_id2'])
        dists_df = dists_df[dists_df['trk_id1'] != dists_df['trk_id2']]

        avg_dists = (
            dists_df.groupby(['trk_id1', 'trk_id2'])['distance']
            .mean()
            .reset_index()
        )
        pivot = avg_dists.pivot(
            index='trk_id1', columns='trk_id2', values='distance'
        ).fillna(1.0)

        trk_ids = pivot.index.to_numpy()
        dist_matrix = pivot.to_numpy()

        # fit kNN using precomputed distances:
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

        return pd.DataFrame(rows)

    def reassociate(self):
        pass
