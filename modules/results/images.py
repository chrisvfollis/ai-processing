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
from botocore.exceptions import EndpointConnectionError, NoCredentialsError


# internal dependencies
from modules.spatial import bboxes
from utilities import utils, io_utils, conn_utils, log_utils


logger = log_utils.get_logger(__name__)


def find_best_event_images(
    time_segment: str,
    presence_df: pd.DataFrame,
    face_data: pd.DataFrame,
    trk_dets: pd.DataFrame,
    min_frame_delta: int = 100,
    min_overlap: float = 0.4,
) -> tuple[pd.DataFrame, dict]:
    project_root = io_utils.get_project_root()

    present_idents = presence_df[presence_df['present_flag']]['identity'].values
    face_data = face_data[face_data['identity'].isin(present_idents)].copy()

    face_data['cam_id'] = face_data['cam_id'].astype(int)
    trk_dets['cam_id'] = trk_dets['cam_id'].astype(int)

    output = []
    for identity, id_faces in face_data.groupby('identity'):
        sort_cols, sort_asc = [], []
        if 'vote_signal' in id_faces.columns:
            sort_cols += ['vote_signal']; sort_asc += [False]
        if 'score' in id_faces.columns:
            sort_cols += ['score'];       sort_asc += [False]
        if 'confidence' in id_faces.columns:
            sort_cols += ['confidence'];  sort_asc += [False]
        if 'distance' in id_faces.columns:
            sort_cols += ['distance'];    sort_asc += [True]

        sorted_faces = (
            id_faces.sort_values(sort_cols, ascending=sort_asc)
            .reset_index(drop=True)
        )

        selected_faces = []

        for i, row in sorted_faces.iterrows():
            if len(selected_faces) == 0:
                selected_faces.append(row)
            elif len(selected_faces) == 1:
                prev = selected_faces[0]

                same_cam = row['cam_id'] == prev['cam_id']
                insufficient_delta = abs(row['f'] - prev['f']) < min_frame_delta

                if not (same_cam and insufficient_delta):
                    selected_faces.append(row)

            if len(selected_faces) == 2:
                break

        if len(selected_faces) == 1:
            selected_faces.append(selected_faces[0].copy())
        elif len(selected_faces) == 0:
            logger.warning(f'No faces found for identity {identity}')
            continue
        
        selected_faces = sorted(selected_faces, key=lambda x: x['f'])

        for i, face in enumerate(selected_faces):
            event      = f'face{i+1}'
            cam_id     = face['cam_id']
            f_num      = face['f']
            face_box   = tuple(face[['x', 'y', 'w', 'h']].tolist())

            best_overlap, best_trk = 0.0, None

            trk_candidates = trk_dets[
                (trk_dets['cam_id'] == cam_id) & (trk_dets['f'] == f_num)
            ]
            for _, trk in trk_candidates.iterrows():
                trk_box = tuple(trk[['x', 'y', 'w', 'h']].tolist())
                overlap = bboxes.compute_overlap_ratio(face_box, trk_box)
                if (overlap >= min_overlap) and (overlap > best_overlap):
                    best_overlap, best_trk = overlap, trk

            if best_trk is not None:
                x, y, w, h = best_trk[['x', 'y', 'w', 'h']]
            else:
                x, y, w, h = face_box

            img_object_key = f'{uuid.uuid4()}.jpg'

            output.append({
                'identity'      : identity,
                'f'             : int(f_num),
                'cam_id'        : int(cam_id),
                'x'             : int(x),
                'y'             : int(y),
                'w'             : int(w),
                'h'             : int(h),
                'event'         : event,
                'image'         : img_object_key,
                'overlap_ratio' : best_overlap,
            })

            full_name = '_'.join(io_utils.lookup_name(identity))
            logger.info(f'Name: {full_name}, Image: {img_object_key}')

    event_imgs_df = pd.DataFrame(output)
    if event_imgs_df.empty:
        print('No event crops to save')
        event_imgs_df = pd.DataFrame(columns=[
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
        ])
        video_paths = {}
        return event_imgs_df, video_paths

    video_paths = {
        cam_id: os.path.join(
            project_root, 'files/input', f'{time_segment}_{cam_id}.mp4'
        )
        for cam_id in event_imgs_df['cam_id'].unique()
    }
    logger.info(f'Found {len(event_imgs_df)} global ID crops across {len(video_paths)} cameras')

    return event_imgs_df, video_paths


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


def global_id_event_imgs(
    time_segment: str,
    presence_df: pd.DataFrame,
    face_data: pd.DataFrame,
    trk_dets: pd.DataFrame,
    credentials: tuple,
    min_frame_delta: int = 100,
) -> pd.DataFrame:
    event_imgs_df, video_paths = find_best_event_images(
        time_segment, presence_df, face_data, trk_dets, min_frame_delta
    )
    extract_and_save_event_images(event_imgs_df, video_paths, credentials)

    return event_imgs_df
