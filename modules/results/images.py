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
    time_segment: str, presence_df: pd.DataFrame, face_data, trk_dets,
    min_frame_delta: int = 100
) -> tuple[pd.DataFrame, dict]:
    project_root = io_utils.get_project_root()

    output = []

    present_idents = presence_df[presence_df['present_flag']]['identity'].values
    face_data = face_data[face_data['identity'].isin(present_idents)].copy()

    face_data['cam_id'] = face_data['cam_id'].astype(int)
    trk_dets['cam_id'] = trk_dets['cam_id'].astype(int)

    for ident, id_faces in face_data.groupby('identity'):
        sorted_faces = id_faces.sort_values('distance').reset_index(drop=True)

        selected_faces = []

        for i, row in sorted_faces.iterrows():
            if len(selected_faces) == 0:
                selected_faces.append(row)
            elif len(selected_faces) == 1:
                prev = selected_faces[0]
                same_cam = row['cam_id'] == prev['cam_id']
                frame_far_enough = abs(row['f'] - prev['f']) >= min_frame_delta

                if not same_cam or frame_far_enough:
                    selected_faces.append(row)
            if len(selected_faces) == 2:
                break

        if len(selected_faces) < 2:
            print(f'Warning: Only found {len(selected_faces)} face(s) for identity {ident}')

        for face_idx, face_row in enumerate(selected_faces):
            event      = f'face{face_idx+1}'
            cam        = face_row['cam_id']
            fnum       = face_row['f']
            x, y, w, h = face_row[['x', 'y', 'w', 'h']]
            face_box   = (x, y, w, h)

            candidates = trk_dets[(trk_dets['f'] == fnum) & (trk_dets['cam_id'] == cam)]
            if candidates.empty:
                print(f'No candidates for {ident} at frame {fnum} on cam {cam}')
                continue

            best_overlap, best_trk = 0.0, None
            for _, trk_row in candidates.iterrows():
                trk_box = (trk_row['x'], trk_row['y'], trk_row['w'], trk_row['h'])
                overlap = bboxes.compute_overlap_ratio(face_box, trk_box)
                if overlap > best_overlap:
                    best_overlap, best_trk = overlap, trk_row

            print(f'{ident} [{event}] → max_overlap={best_overlap:.2f}, trk_found={best_trk is not None}, frame={fnum}, cam={cam}, candidates={len(candidates)}')

            if best_trk is not None:
                full_name = '_'.join(io_utils.lookup_name(ident))
                event_image = f'{uuid.uuid4()}.jpg'

                logger.info(f'Name: {full_name}, Image: {event_image}')
                output.append({
                    'identity'      : ident,
                    'f'             : int(best_trk['f']),
                    'cam_id'        : int(best_trk['cam_id']),
                    'x'             : int(best_trk['x']),
                    'y'             : int(best_trk['y']),
                    'w'             : int(best_trk['w']),
                    'h'             : int(best_trk['h']),
                    'event'         : event,
                    'image'         : event_image,
                    'overlap_ratio' : best_overlap,
                })

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
    time_segment, presence_df, face_data, trk_dets, credentials,
    min_frame_delta: int = 100
) -> pd.DataFrame:
    event_imgs_df, video_paths = find_best_event_images(
        time_segment, presence_df, face_data, trk_dets, min_frame_delta
    )
    extract_and_save_event_images(event_imgs_df, video_paths, credentials)

    return event_imgs_df
