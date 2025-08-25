# standard dependencies
import os
import uuid
from urllib.parse import urlparse
from typing import Optional



# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import av
from scipy.optimize import linear_sum_assignment
from botocore.exceptions import EndpointConnectionError, NoCredentialsError


# internal dependencies
from modules.spatial import bboxes
from utilities import io_utils, conn_utils, log_utils


logger = log_utils.get_logger(__name__)


def save_event_image(
    img: np.ndarray,
    object_key: Optional[str] = None,
    credentials: Optional[tuple[str]] = None,
    region: str = 'us-west-1',
    bucket_name: str = 'timemanager-event-imgs',
    event_imgs_dir: str = None,
) -> str | None:
    if img is None:
        print('Event image is NoneType')
        return None
    
    object_key = object_key or f'{uuid.uuid4()}.jpg'

    credentials = credentials or conn_utils.get_aws_credentials()
    event_imgs_dir = event_imgs_dir or os.path.join(
        io_utils.get_project_root(), 'files/output/', 'event_imgs/'
    )
    file_path = os.path.join(event_imgs_dir, object_key)

    cv2.imwrite(file_path, img)

    try:
        s3_client = conn_utils.s3_connect(region, credentials)
        s3_client.upload_file(file_path, bucket_name, object_key)

        io_utils.remove_files(file_path, missing_ok=True)

        return object_key
    except (EndpointConnectionError, NoCredentialsError) as e:
        print(f'Unable to connect: {e}')
    except Exception as e:
        print(f'Unexpected error during upload: {e}')


def extract_and_save_event_images(
    event_imgs_df: pd.DataFrame,
    video_paths: dict[int, str],
    credentials: tuple[str, str],
):
    for cam_id, cam_df in event_imgs_df.groupby('cam_id'):
        logger.info(f'Extracting event images from cam_id {cam_id}')
        video_path = video_paths.get(cam_id)
        if not video_path or not os.path.exists(video_path):
            print(f'Video path does not exist: {video_path}')
            continue

        frame_crop_map = {}
        for _, row in cam_df.iterrows():
            f = row['f']
            frame_crop_map.setdefault(f, []).append(row)

        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        for idx, frame in enumerate(container.decode(stream)):
            if idx not in frame_crop_map:
                continue

            img = frame.to_ndarray(format='bgr24')
            for row in frame_crop_map[idx]:
                x1 = max(0, row['x'])
                y1 = max(0, row['y'])
                x2 = min(row['x'] + row['w'], img.shape[1])
                y2 = min(row['y'] + row['h'], img.shape[0])
                crop = img[y1:y2, x1:x2]

                save_event_image(
                    img=crop,
                    object_key=row['image'],
                    credentials=credentials,
                )
        container.close()
    return


def find_best_event_images(
    time_segment: str,
    presence_df: pd.DataFrame,
    face_data: pd.DataFrame,
    person_dets: pd.DataFrame,
    min_frame_delta: int = 60,
    min_overlap: float = 0.20,
    topk: int = 10,
    overlap_weight: float = 0.75,
    quality_weight: float = 1.00,
) -> tuple[pd.DataFrame, dict]:
    project_root = io_utils.get_project_root()

    output_df_cols = [
        'identity',
        'f',
        'cam_id',
        'x',
        'y',
        'w',
        'h',
        'event',
        'image',
        'overlap_ratio',
    ]

    present_idents = presence_df[presence_df['present_flag']]['identity'].values

    person_dets = person_dets.copy()
    if not person_dets.empty:
        person_dets['cam_id'] = person_dets['cam_id'].astype(int)
        person_dets['f'] = person_dets['f'].astype(int)

    def _robust_minmax(col: pd.Series, invert: bool = False) -> pd.Series:
        col = col.astype(float)
        x = col.to_numpy(copy=True)
        mask = np.isfinite(x)
        if not mask.any():
            return pd.Series(np.zeros_like(x), index=col.index, dtype=float)

        lo = np.nanpercentile(x[mask], 1)
        hi = np.nanpercentile(x[mask], 99)
        if hi <= lo:
            scaled = np.zeros_like(x, dtype=float)
        else:
            scaled = np.clip((x - lo) / (hi - lo), 0.0, 1.0)

        if invert:
            scaled = 1.0 - scaled

        scaled[~mask] = 0.0
        return pd.Series(scaled, index=col.index, dtype=float)

    def _add_quality_column(face_df: pd.DataFrame) -> pd.DataFrame:
        df = face_df.copy()
        parts = []
        weights = []

        parts.append(_robust_minmax(df['score'], invert=False));       weights.append(1.0)
        parts.append(_robust_minmax(df['distance'], invert=True));     weights.append(0.8)
        parts.append(_robust_minmax(df['confidence'], invert=False));  weights.append(0.6)

        if parts:
            W = np.array(weights, dtype=float)
            W = W / W.sum()
            Q = np.average(np.vstack([p.values for p in parts]), axis=0, weights=W)
            df['quality'] = Q
        else:
            df['quality'] = 0.5  # fallback

        return df

    def _topk_candidates(
        face_df: pd.DataFrame, present_idents: np.ndarray, topk: int
    ) -> pd.DataFrame:
        cand = face_df[face_df['identity'].isin(present_idents)].copy()
        if cand.empty:
            return cand

        if 'cam_id' in cand.columns:
            cand['cam_id'] = cand['cam_id'].astype(int)
        if 'f' in cand.columns:
            cand['f'] = cand['f'].astype(int)

        cand = _add_quality_column(cand)

        sort_cols, sort_asc = ['quality'], [False]
        for col, asc in [
            ('score', False), ('distance', True), ('confidence', False)
        ]:
            if col in cand.columns:
                sort_cols.append(col); sort_asc.append(asc)

        cand = cand.sort_values(sort_cols, ascending=sort_asc)

        cand['rank'] = cand.groupby('identity')['quality'].rank(
            method='first', ascending=False
        )
        cand = cand[cand['rank'] <= float(topk)].drop(columns=['rank'])

        required = {'identity', 'cam_id', 'f', 'x', 'y', 'w', 'h', 'quality'}
        missing = required - set(cand.columns)
        if missing:
            raise ValueError(f'Candidates missing required columns: {missing}')

        return cand

    try:
        candidates_df = _topk_candidates(face_data, present_idents, topk=topk)
    except ValueError as e:
        logger.error(str(e))
        print('No event crops to save')
        return pd.DataFrame(columns=output_df_cols), {}

    if candidates_df.empty:
        print('No event crops to save')
        return pd.DataFrame(columns=output_df_cols), {}

    def _assign_per_frame(
        candidates_df: pd.DataFrame,
        person_dets: pd.DataFrame,
        min_overlap: float,
        overlap_weight: float = 0.75,
        quality_weight: float = 1.00,
    ) -> list[dict]:
        if candidates_df.empty:
            return []

        dets = person_dets.copy()
        if dets.empty:
            return []
        dets['cam_id'] = dets['cam_id'].astype(int)
        dets['f'] = dets['f'].astype(int)

        INF = 1e6
        results = []

        for (cam_id, f_num), faces in candidates_df.groupby(['cam_id', 'f']):
            faces = faces.reset_index(drop=True)
            det_rows = (
                dets[(dets['cam_id'] == cam_id) & (dets['f'] == f_num)]
                .reset_index(drop=True)
            )
            if faces.empty or det_rows.empty:
                continue

            n_i = len(faces)
            n_j = len(det_rows)
            cost = np.full((n_i, n_j), INF, dtype=np.float32)

            for i, face in faces.iterrows():
                fx, fy, fw, fh = face[['x','y','w','h']].astype(float).tolist()
                fbox = (fx, fy, fw, fh)
                fqual = float(face['quality'])

                for j, det in det_rows.iterrows():
                    dx, dy, dw, dh = det[['x','y','w','h']].astype(float).tolist()
                    dbox = (dx, dy, dw, dh)
                    ov = bboxes.compute_overlap_ratio(fbox, dbox)

                    if ov < min_overlap:
                        continue

                    utility = (overlap_weight * ov) + (quality_weight * fqual)
                    cost[i, j] = -utility

            row_ind, col_ind = linear_sum_assignment(cost)

            # collect only feasible matches
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= INF:
                    continue
                face = faces.iloc[r]
                det  = det_rows.iloc[c]

                fx, fy, fw, fh = face[['x','y','w','h']].astype(float).tolist()
                dx, dy, dw, dh = det[['x','y','w','h']].astype(float).tolist()
                ov = bboxes.compute_overlap_ratio((fx,fy,fw,fh), (dx,dy,dw,dh))

                results.append({
                    'identity': face['identity'],
                    'cam_id': int(cam_id),
                    'f': int(f_num),
                    'x': int(round(dx)),
                    'y': int(round(dy)),
                    'w': int(round(dw)),
                    'h': int(round(dh)),
                    'overlap_ratio': float(ov),
                    'quality': float(face['quality']),
                })

        return results

    matches = _assign_per_frame(
        candidates_df=candidates_df,
        person_dets=person_dets,
        min_overlap=min_overlap,
        overlap_weight=overlap_weight,
        quality_weight=quality_weight,
    )

    if not matches:
        logger.warning('No feasible identity↔person assignments found with current gating.')
        print('No event crops to save')
        return pd.DataFrame(columns=output_df_cols), {}

    def _pick_imgs_per_identity(
        matches: list[dict], min_frame_delta: int
    ) -> list[dict]:
        if not matches:
            return []

        by_id = {}
        for m in matches:
            by_id.setdefault(m['identity'], []).append(m)

        out = []
        for ident, rows in by_id.items():
            rows = sorted(
                rows, key=lambda r: (r['quality'], r['overlap_ratio']),
                reverse=True
            )

            chosen = []
            for r in rows:
                if not chosen:
                    chosen.append(r)
                else:
                    if abs(r['f'] - chosen[0]['f']) >= min_frame_delta:
                        chosen.append(r)
                if len(chosen) == 2:
                    break

            if len(chosen) == 1:
                chosen.append(dict(chosen[0]))

            out.extend(chosen)

        return out

    chosen = _pick_imgs_per_identity(matches, min_frame_delta=min_frame_delta)
    if not chosen:
        logger.warning('No spaced event pairs per identity after selection.')
        print('No event crops to save')
        return pd.DataFrame(columns=output_df_cols), {}

    rows = []
    for ident, group in pd.DataFrame(chosen).groupby('identity'):
        group = group.sort_values('f', ascending=True).reset_index(drop=True)
        for i, rec in group.iterrows():
            event = f'face{i+1}'
            img_object_key = f'{uuid.uuid4()}.jpg'
            rows.append({
                'identity'      : ident,
                'f'             : int(rec['f']),
                'cam_id'        : int(rec['cam_id']),
                'x'             : int(rec['x']),
                'y'             : int(rec['y']),
                'w'             : int(rec['w']),
                'h'             : int(rec['h']),
                'event'         : event,
                'image'         : img_object_key,
                'overlap_ratio' : float(rec['overlap_ratio']),
            })

            full_name = '_'.join(io_utils.lookup_name(ident))
            logger.info(f'Name: {full_name}, Image: {img_object_key}')

    event_imgs_df = pd.DataFrame(rows)

    video_paths = {
        cam_id: os.path.join(project_root, 'files/input', f'{time_segment}_{cam_id}.mp4')
        for cam_id in event_imgs_df['cam_id'].unique()
    }

    logger.info(f'Found {len(event_imgs_df)} global ID crops across {len(video_paths)} cameras')
    return event_imgs_df, video_paths


def global_id_event_imgs(
    time_segment: str,
    presence_df: pd.DataFrame,
    filtered_face_data: pd.DataFrame,
    person_dets: pd.DataFrame,
    credentials: tuple,
    min_frame_delta: int = 20,
) -> pd.DataFrame:
    event_imgs_df, video_paths = find_best_event_images(
        time_segment,
        presence_df,
        filtered_face_data,
        person_dets,
        min_frame_delta   = min_frame_delta,
        min_overlap       = 0.15,
        topk              = 15,
        overlap_weight    = 0.50,
        quality_weight    = 1.0,
    )
    extract_and_save_event_images(event_imgs_df, video_paths, credentials)

    return event_imgs_df
