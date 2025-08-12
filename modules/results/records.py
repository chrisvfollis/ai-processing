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
from utilities.conn_utils import APIClient


logger = log_utils.get_logger(__name__)


# =============================================================================
#                        - PREPARATION/FORMATTING -
# -----------------------------------------------------------------------------


def get_event_records(time_segment: str, designation: Optional[str] = None,
                   db_name: str = 'data.db') -> list[tuple]:
    db_path = os.path.join(io_utils.get_project_root(), 'files/', db_name)
    conn, cursor = conn_utils.sqlite_db_connect(db_path)
    
    query = '''
        SELECT track_info.*, people.designation
        FROM track_info
        LEFT JOIN people ON track_info.identity = people.identity_uuid
        WHERE track_info.time_prefix = ?
    '''
    params = [time_segment]

    if designation is not None:
        query += ' AND (people.designation = ? OR people.designation IS NULL)'
        params.append(designation)
    params = tuple(params)

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn_utils.close_sqlite_db(conn, cursor)
    return results


def convert_records_to_df(event_records: list[tuple]) -> pd.DataFrame | None:
    columns = [
        'id', 'track_id', 'camera', 'time_prefix', 'identity', 'id_method',
        'id_cost', 'start_img', 'end_img', 'id_img',  'start_time', 'end_time',
        'entry', 'exit', 'designation'
    ]
    track_df = pd.DataFrame(event_records, columns=columns)

    track_df['start_time'] = pd.to_datetime(track_df['start_time'], format='mixed')
    track_df['end_time'] = pd.to_datetime(track_df['end_time'], format='mixed')

    return track_df


def merge_records_by_time(
    event_records: pd.DataFrame, max_gap: int = 75
) -> pd.DataFrame:
    merged = []
    for identity, group in event_records.groupby('identity'):
        if identity == '':
            merged.extend(group.to_dict(orient='records'))
            continue

        group = group.sort_values('start_time').reset_index(drop=True)
        current = group.iloc[0].to_dict()
        for _, row in group.iloc[1:].iterrows():
            gap = (row['start_time'] - current['end_time']).total_seconds()

            if gap <= max_gap:
                current['end_time'] = max(current['end_time'], row['end_time'])
                current['end_img'] = row['end_img']
            else:
                merged.append(current)
                current = row.to_dict()

        merged.append(current)

    return pd.DataFrame(merged)


# =============================================================================
#                             - SAVING -
# -----------------------------------------------------------------------------


def save_tracks(time_segment: str, target_trks: dict, fps: int = 30,
                    db_name='data.db') -> None:
    conn, cursor = conn_utils.sqlite_db_connect(os.path.join(
        io_utils.get_project_root(), 'files/', db_name
    ))
    columns = [
        'time_prefix',
        'identity',
        'start_img',
        'end_img',
        'start_time',
        'end_time',
    ]
    query_columns = utils.query_columns_string(columns)
    param_placeholders = utils.query_param_placeholders(columns)
    query = f'''
        INSERT INTO track_info {query_columns}
        VALUES {param_placeholders}
    '''
   
    for trk in target_trks.values():
        identity = trk.identity or str(uuid.uuid4())

        start_img = trk.id_event_images[0]
        end_img = trk.id_event_images[-1]
        
        start_frame = trk.start
        end_frame = trk.map_offset(trk.start, trk.age)

        start_time = utils.frame_timestamp(time_segment, start_frame, fps)
        end_time = utils.frame_timestamp(time_segment, end_frame, fps)

        values = (
            time_segment,
            identity,
            start_img,
            end_img,
            start_time,
            end_time,
        )
        cursor.execute(query, values)

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


def save_attendance(
        time_segment: str,
        presence_df: pd.DataFrame,
        event_imgs_df: pd.DataFrame,
        segment_length: float = 5.0,
        db_name: str = 'data.db',
) -> None:
    conn, cursor = conn_utils.sqlite_db_connect(os.path.join(
        io_utils.get_project_root(), 'files/', db_name
    ))
    columns = [
        'time_prefix',
        'identity',
        'start_img',
        'end_img',
        'start_time',
        'end_time',
    ]
    query_columns = utils.query_columns_string(columns)
    param_placeholders = utils.query_param_placeholders(columns)
    query = f'''
        INSERT INTO track_info {query_columns}
        VALUES {param_placeholders}
    '''

    total_seconds = 60 * segment_length
    fps = 15
    end_frame = fps * total_seconds

    start_time = utils.frame_timestamp(time_segment)
    end_time = utils.frame_timestamp(time_segment, end_frame, fps)

    for _, row in presence_df.iterrows():
        if not row['present_flag']:
            continue

        identity = row['identity']

        imgs = event_imgs_df[event_imgs_df['identity'] == identity]
        if not imgs.empty:
            entry_ = imgs[imgs['event'] == 'face1']
            exit_ = imgs[imgs['event'] == 'face2']

            start_img = entry_.iloc[0]['image'] if not entry_.empty else ''
            end_img = exit_.iloc[0]['image'] if not exit_.empty else ''
        else:
            start_img = end_img = ''

        values = (
            time_segment,
            identity,
            start_img,
            end_img,
            start_time,
            end_time,
        )
        cursor.execute(query, values)

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


def post_event_records(shop_id: str, time_segment: str) -> bool:
    successful_post = False
    num_results = 0

    event_records = get_event_records(time_segment, designation='tracked_employee')

    if (not event_records) or (len(event_records) == 0):
        logger.info('No event records found to post')
        return successful_post

    event_records_df = convert_records_to_df(event_records)
    event_records_df = merge_records_by_time(event_records_df)

    event_data = {
        'shop_id': [],
        'employee_id': [],
        'event': [],
        'start_time': [],
        'duration': [],
        'image': []
    }

    for _, row in event_records_df.iterrows():
        identity = row['identity']

        start_time, end_time = [str(row[t]) for t in ('start_time', 'end_time')]
        start_image, end_image = row['start_img'], row['end_img']

        num_results += 1

        # start event data:
        event_data['shop_id'].append(shop_id)
        event_data['employee_id'].append(identity)
        event_data['event'].append('workspace_entry')
        event_data['start_time'].append(start_time)
        event_data['duration'].append(0)
        event_data['image'].append(start_image)

        # end event data:
        event_data['shop_id'].append(shop_id)
        event_data['employee_id'].append(identity)
        event_data['event'].append('workspace_exit')
        event_data['start_time'].append(end_time)
        event_data['duration'].append(0)
        event_data['image'].append(end_image)

    webapp_api = APIClient(var_prefix='WEBAPP_API')
    response = webapp_api.post('save_employee_event_logs/', json=event_data)

    if response.status_code == 200:
        logger.info(f'Success: posted {num_results} employee results')            
        
        successful_post = True
    else:
        logger.error(f'Failed posting to webapp: {response.text}')
        logger.error(response.status_code)

    return successful_post
