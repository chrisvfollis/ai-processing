# standard dependencies
import os
import sqlite3
import re
import getpass
import subprocess
import gc
import uuid
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Union, Optional

# 3rd-party dependencies
import numpy as np
import pandas as pd
import h5py
import cv2
import torch
from dotenv import load_dotenv
import boto3
from botocore.exceptions import EndpointConnectionError, NoCredentialsError
import requests
import psutil
import psycopg2

# internal dependencies
from utilities import general_utils as utils
from utilities import conn_utils
from utilities.conn_utils import APIClient


# =============================================================================
#                           - MEMORY MANAGEMENT -
# -----------------------------------------------------------------------------


def clear_memory():
    import tensorflow as tf
    K = tf.keras.backend
    K.clear_session()
    torch.cuda.empty_cache()
    gc.collect()


def cleanup_semaphores(logger):
    '''
    Removes unused (stale or leaked) semaphores:

    - POSIX-named semaphores from /dev/shm/
    - SysV IPC semaphores using ipcs -s 
    '''

    try:
        posix_semaphores = [f for f in os.listdir('/dev/shm/')
                            if f.startswith('sem.')]

        if posix_semaphores:
            for sem in posix_semaphores:
                sem_path = f'/dev/shm/{sem}'

                if any(
                    sem in p.open_files() for p in
                    psutil.process_iter(['open_files'])
                ):
                    logger.info(f'Skipping active semaphore: {sem_path}')
                    continue
    
                try:
                    os.unlink(sem_path)
                    logger.info(f'Removed unused POSIX semaphore: {sem_path}')
                except FileNotFoundError:
                    logger.error(f'Skipped: {sem_path} already removed.')
                except Exception as e:
                    logger.error(f'Error removing {sem_path}: {e}')
        else:
            logger.info('No unused POSIX semaphores found.')

    except Exception as e:
        logger.error(f'Error checking POSIX semaphores: {e}')


    user = getpass.getuser()
    try:
        output = subprocess.check_output(['ipcs', '-s']).decode('utf-8')

        sysv_semaphores = [
            line.split()[1] for line in output.split('\n') if user in line
        ]

        if sysv_semaphores:
            for sem_id in sysv_semaphores:
                os.system(f'ipcrm -s {sem_id}')
                logger.info(f'Removing unused SysV semaphore: {sem_id}')
        else:
            logger.info('No unused SysV semaphores found.')

    except Exception as e:
        logger.error(f'Error checking SysV semaphores: {e}')


# =============================================================================
#                        - ENVIRONMENT VARIABLES -
# -----------------------------------------------------------------------------


def get_aws_credentials():
    load_dotenv()
    access_key = os.environ.get('AWS_ACCESS_KEY')
    secret_key = os.environ.get('AWS_SECRET_KEY')

    return access_key, secret_key


# =============================================================================
#                           - LOCAL FILES -
# -----------------------------------------------------------------------------


def get_unique_filename(dir_path, base_name):
    '''Append a number to the filename if it already exists.'''

    filename, ext = os.path.splitext(base_name)
    counter = 1
    file_path = os.path.join(dir_path, base_name)

    new_name = base_name
    while os.path.exists(file_path):
        new_name = f"{filename}_{counter}{ext}"
        file_path = os.path.join(dir_path, new_name)
        counter += 1

    return new_name


def get_unique_path(dir_path, base_name):
    base_name = base_name.strip('/')
    counter = 1
    candidate = os.path.join(dir_path, base_name)
    while os.path.exists(candidate):
        candidate = os.path.join(dir_path, f'{base_name}_{counter}')
        counter += 1
    return candidate


def get_latest_file(dir_path, base_name):
    '''Find the latest version of a file by checking for appended digits.'''

    filename, ext = os.path.splitext(base_name)
    pattern = re.compile(re.escape(filename) + r'(?:_(\d+))?' + re.escape(ext) + '$')

    latest_version = -1
    latest_file = None

    for f in os.listdir(dir_path):
        match = pattern.match(f)
        if match:
            version = int(match.group(1)) if match.group(1) is not None else 0
            if version > latest_version:
                latest_version = version
                latest_file = f

    return latest_file


def delete_local_files(identifier, file_types='any',
                 paths=['../files/input', '../files/output',
                        '../files/output/event_imgs']):
    def _parse_name_and_extension(file):
        file_parts = [x for x in file.rsplit('.', 1)]

        name = file_parts[0]
        extension = (
            file_parts[-1] if len(file_parts) == 2 else ''
        )
    
        return name, extension

    n_deleted = 0
    for path in paths:
        if not os.path.exists(path):
            print(f'Skipping non-existent path: {path}')
            continue
        for result in os.listdir(path):
            full_path = os.path.join(path, result)
            if not os.path.isfile(full_path):
                continue
            elif (
                    (full_path.endswith('.pkl')) and
                    (not full_path.endswith('inference_pipeline.pkl'))
                ):
                continue
            
            file_name, file_extension = _parse_name_and_extension(result)
            if (
                ((identifier == 'all') or (file_name.startswith(identifier))) and
                ((file_types == 'any') or (file_extension in file_types))
            ):
                try:
                    os.remove(full_path)
                    n_deleted += 1
                except Exception as e:
                    print(f'Error deleting {full_path}: {e}')

    print(f'Deleted {n_deleted} files')
    return True


def read_embeddings(hdf5_file, target_frame, device):
    with h5py.File(hdf5_file, 'r') as file:
        frames = file['frames'][:]

        indices = np.where(frames == target_frame)[0]
        if len(indices) == 0:
            print(f"Found 0 indices for frame {target_frame}")

        target_embeddings = file['embeddings'][sorted(indices)]
        target_embeddings = (
            torch.from_numpy(target_embeddings)
            .to(device)
            .detach()
        )

        return target_embeddings


def save_event_image(img, credentials, img_dir='../files/output/event_imgs/'):
    if img is None:
        return None
    object_key = f'{uuid.uuid4()}.jpg'
    file_path = os.path.join(img_dir, object_key)
    cv2.imwrite(file_path, img)
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=credentials[0],
            aws_secret_access_key=credentials[1],
            region_name='us-west-1'
        )
        bucket_name = 'timemanager-event-imgs'
        s3_client.upload_file(file_path, bucket_name, object_key)
        if os.path.exists(file_path):
            os.remove(file_path)
    except (EndpointConnectionError, NoCredentialsError) as e:
        pass

    return object_key


def upload_file(s3_client, bucket_name, file_path, object_key):
    try:
        s3_client.upload_file(file_path, bucket_name, object_key)
    except Exception as e:
        print(f'Failed to upload {file_path}: {e}')


def upload_data(credentials, dir='../files/output/', max_workers=8):
    try:
        session = boto3.session.Session()
        s3_client = session.client(
            's3',
            aws_access_key_id=credentials[0],
            aws_secret_access_key=credentials[1],
            region_name='us-west-1'
        )
        bucket_name = 'visionservice-data'

        upload_tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for root, _, files in os.walk(dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    object_key = file
                    upload_tasks.append(executor.submit(
                        upload_file, s3_client, bucket_name, file_path, object_key
                    ))

            for future in as_completed(upload_tasks):
                future.result()

    except (EndpointConnectionError, NoCredentialsError) as e:
        print(f'S3 client error: {e}')


# =============================================================================
#                           - REMOTE FILES -
# -----------------------------------------------------------------------------


def download_s3_footage(
        object_key,
        credentials=None,
        bucket_name='ivakt-footage'
    ) -> bool:
    s3_client = conn_utils.s3_connect(region='us-west-1', credentials=credentials)
    video_file = object_key.split('/')[-1]
    local_path = os.path.join('../files/input', video_file)

    try:
        s3_client.download_file(bucket_name, object_key, local_path)
        print(f'Downloaded {object_key}')
        return True
    except Exception as e:
        print(f"Failed to download {object_key}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def delete_s3_footage(
        object_key,
        credentials=None,
        bucket_name='ivakt-footage'
    ) -> bool:
    s3_client = conn_utils.s3_connect(
        region='us-west-1', credentials=credentials
    )

    try:
        s3_client.delete_object(Bucket=bucket_name, Key=object_key)
        print(f'Deleted {object_key} from S3')
        return True
    except Exception as e:
        print(f"Failed to delete {object_key} from S3: {e}")
        return False


def download_s3_image(
        object_key,
        credentials=None,
        filename=None,
        img_dir='../files/input',
        bucket_name='ivakt-employee-photos'
    ) -> bool:
    s3_client = conn_utils.s3_connect(
        region='us-west-1', credentials=credentials
    )
    if not filename:
        filename = object_key.split('/')[-1]
    output_path = os.path.join(img_dir, filename)

    try:
        if os.path.exists(output_path):
            print('Image already saved')
            return False
        
        s3_client.download_file(bucket_name, object_key, output_path)
        print(f'Downloaded {object_key}')
        return True
    except Exception as e:
        print(f"Failed to download {object_key}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


# =============================================================================
#                           - LOCAL DATABASE -
# -----------------------------------------------------------------------------


def build_database(db_path='../files/data.db') -> None:
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_uuid TEXT UNIQUE NOT NULL,
            shop_uuid TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            designation TEXT NOT NULL
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person INTEGER NOT NULL,
            file TEXT UNIQUE NOT NULL,
            FOREIGN KEY (person) REFERENCES people (id) ON DELETE CASCADE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS track_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT, camera TEXT, time_prefix TEXT,
            identity TEXT, id_method TEXT, id_cost FLOAT,
            start_img TEXT, end_img TEXT, id_img TEXT,
            start_time DATETIME, end_time DATETIME,
            entry INTEGER, exit INTEGER
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shop (
            uuid TEXT PRIMARY KEY,
            shop_name TEXT NOT NULL
        );    
    ''')

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


def get_shop(db_path='../files/data.db') -> tuple:
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    cursor.execute('''
        SELECT * FROM shop
        LIMIT 1
    ''')
    results = cursor.fetchone()

    conn_utils.close_sqlite_db(conn, cursor)
    return results


def lookup_identities(image_paths, db_path='../files/data.db') -> list[tuple]:
    '''
    Returns:
        results (List[Tuple]):
            A list of tuples corresponding to rows in the database. The order
            is as follows:
            - id: integer primary key
            - identity_uuid: the person's unique identifier
            - shop_uuid: the unique identifier of the shop
            - first_name
            - last_name
            - designation: determines whether the person's data is reported 
    '''
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    filenames = tuple(
        [img_path.split('/')[-1] for img_path in image_paths]
    )
    param_placeholders = utils.query_param_placeholders(filenames)

    query = f'''
        SELECT people.*, faces.file
        FROM people
        JOIN faces ON people.id = faces.person
        WHERE faces.file IN {param_placeholders};
    '''
    cursor.execute(query, filenames)

    results = cursor.fetchall()
    results_map = {row[-1]: row[:-1] for row in results}

    conn_utils.close_sqlite_db(conn, cursor)

    return [results_map.get(filename) for filename in filenames]


def lookup_name(identity_uuid, db_path='../files/data.db') -> tuple:
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    identity_uuid = (identity_uuid,)
    param_placeholder = utils.query_param_placeholders(identity_uuid)

    query = f'''
        SELECT first_name, last_name FROM people
        WHERE identity_uuid = {param_placeholder}
        LIMIT 1;
    '''
    cursor.execute(query, identity_uuid)
    result = cursor.fetchone() or ('', '')

    conn_utils.close_sqlite_db(conn, cursor)
    return result


def get_designation(identity_uuid, db_path='../files/data.db') -> Union[str, None]:
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    identity_uuid = (identity_uuid,)
    param_placeholder = utils.query_param_placeholders(identity_uuid)

    query = f'''
        SELECT designation FROM people
        WHERE identity_uuid = {param_placeholder}
        LIMIT 1;
    '''
    cursor.execute(query, identity_uuid)
    result = cursor.fetchone()
    designation = result[0] if result else None

    conn_utils.close_sqlite_db(conn, cursor)
    return designation


def save_track_info(time_prefix: str, camera: str, target_trks: dict,
                    fps: int = 30, db_path='../files/data.db') -> None:
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    columns = [
        'time_prefix',
        'camera',
        'track_id',
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
   
    for trk_id, trk in target_trks.items():
        identity = trk.identity or str(uuid.uuid4())

        start_img = trk.start_img or ''
        end_img = trk.end_img or ''

        if not start_img and not end_img:
            continue    # skip tracks with no images
        
        start_frame, end_frame = trk.span[0], trk.span[-1]

        start_time = utils.frame_timestamp(time_prefix, start_frame, fps)
        end_time = utils.frame_timestamp(time_prefix, end_frame, fps)

        values = (
            time_prefix, camera, trk_id, identity, start_img, end_img,
            start_time, end_time,
        )

        cursor.execute(query, values)

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


def get_track_info(time_prefix: str, designation: Optional[str] = None,
                   db_path: str = '../files/data.db') -> list[tuple]:
    conn, cursor = conn_utils.sqlite_db_connect(db_path)
    
    query = '''
        SELECT track_info.*, people.designation
        FROM track_info
        LEFT JOIN people ON track_info.identity = people.identity_uuid
        WHERE track_info.time_prefix = ?
    '''
    params = [time_prefix]

    if designation is not None:
        query += ' AND (people.designation = ? OR people.designation IS NULL)'
        params.append(designation)
    params = tuple(params)

    cursor.execute(query, params)
    results = cursor.fetchall()

    conn_utils.close_sqlite_db(conn, cursor)
    return results


def update_track_info(time_prefix, updates, db_path='../files/data.db') -> None:
    conn, cursor = conn_utils.sqlite_db_connect(db_path)
    
    for track_id, data in updates.items():
        camera, id = track_id.split('_')[0], track_id.split('_')[1].strip('trk')
        columns = sorted(data.keys())
        set_clause = ", ".join(f"{col} = ?" for col in columns)
        values = [data[col] for col in columns]

        condition = f"time_prefix = ? AND camera = ? AND track_id = ?"
        values.extend([time_prefix, camera, id])
        query = f"UPDATE track_info SET {set_clause} WHERE {condition}"

        cursor.execute(query, values)

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


def clear_track_info(identifier, db_path='../files/data.db') -> None:
    conn, cursor = conn_utils.sqlite_db_connect(db_path)
    try:
        if identifier == 'all':
            cursor.execute('DELETE FROM track_info')
        else:
            cursor.execute('''
                DELETE FROM track_info
                WHERE time_prefix = ?
            ''', (identifier,))
    except Exception as e:
        print(f'Unable to clear track_info: {e}')
    finally:
        conn_utils.close_sqlite_db(conn, cursor, commit=True)


def save_person_data(
        person_data, db_path='../files/data.db', img_dir='../files/input/faces'
    ) -> None:
    def _format_filename(img_url) -> str:
        filename = img_url.rsplit('/', 1)[1]    # remove bucket/folder info
        return '.'.join(filename.rsplit('_', 1)[:2])    # format file extension

    credentials = get_aws_credentials()
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    column_names_people = [
        'first_name',
        'last_name',
        'designation',
        'identity_uuid',
        'shop_uuid',
    ]
    query_columns_people = utils.query_columns_string(column_names_people)
    param_placeholders_people = utils.query_param_placeholders(
        column_names_people
    )
    update_clause_people = ", ".join([
        f"{col}=excluded.{col}" for col in column_names_people
        if col != 'identity_uuid'
    ])
    insert_query_people = f'''
        INSERT INTO people {query_columns_people}
        VALUES {param_placeholders_people}
        ON CONFLICT(identity_uuid) DO UPDATE SET
            {update_clause_people}
        RETURNING id
    '''

    column_names_faces = [
        'person',
        'file',
    ]
    query_columns_faces = utils.query_columns_string(column_names_faces)
    param_placeholders_faces = utils.query_param_placeholders(
        column_names_faces
    )
    insert_query_faces = f'''
        INSERT OR IGNORE INTO faces {query_columns_faces}
        VALUES {param_placeholders_faces}
    '''

    for person in person_data:
        if 'is_active' in person:
            designation = 'tracked_employee' if person['is_active'] else 'untracked'
        else:
            designation = 'tracked_employee'
        
        values_people = (
            person['first_name'],
            person['last_name'],
            designation,
            person['uuid'],
            person['shop_uuid'],
        )
        cursor.execute(insert_query_people, values_people)
        person_id = cursor.fetchone()[0]

        img_urls = [
            person['front_image'],
            person['left_image'],
            person['right_image'],
        ]
        for img_url in img_urls:
            parsed, filename = urlparse(img_url), _format_filename(img_url)
            object_key = parsed.path.lstrip('/')

            values_faces = (
                person_id,
                filename
            )
            cursor.execute(insert_query_faces, values_faces)
            download_s3_image(
                object_key, credentials, filename=filename, img_dir=img_dir
            )

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


# =============================================================================
#                         - API/REMOTE DATABASE -
# -----------------------------------------------------------------------------


def get_api_tokens(credentials: dict = None) -> Union[tuple[str], tuple[None]]:
    if not credentials:
        credentials = {
            'email': input('Enter account email: '),
            'password': input('Enter account password: ')
        }

    webapp_api = APIClient(var_prefix='WEBAPP_API')
    response = webapp_api.post('accounts/login/', json=credentials)

    if response.status_code == 200:
        access_token = response.json().get('access')
        refresh_token = response.cookies.get('refresh_token')
        
        api_tokens = (access_token, refresh_token)
    else:
        api_tokens = (None, None)
        print(f'Error: {response.status_code}: {response.json()}')
    
    return api_tokens


def fetch_person_data(
        shop_uuid: str = None, access_token: str = None, save_data: bool = True,
        db_path: str = '../files/data.db', img_dir: str = '../files/input/faces'
    ) -> list:
    shop_uuid = shop_uuid or get_shop(db_path=db_path)[0]
    access_token = access_token or get_api_tokens()[0]
    
    webapp_api = APIClient(var_prefix='WEBAPP_API')

    params = {'shop_uuid': shop_uuid}
    headers = {
        'X-Custom-API-Key': webapp_api.api_key,
        'Authorization': f'Bearer {access_token}'
    }
    response = webapp_api.get('employees-json/', headers=headers, params=params)

    if response.status_code == 200:
        person_data = response.json().get('employees', [])
        if save_data:
            save_person_data(person_data, db_path=db_path, img_dir=img_dir)
    else:
        person_data = []
        print(f'Error: {response.status_code}: {response.text}')
    
    return person_data


def get_queue_block(shop_id: str, start_from: Union[list, datetime] = None,
                    priority_camera: str = None) -> Union[list[list], None, False]:
    '''
    Returns:
        queue_block (List[List]):
            A list of lists, where each sublist contains the information for
            one of the multiple video files recorded concurrently by the shop's
            cameras over a given period of time. The order is as follows:
            - vid_object_key
            - timestamp
            - camera
    '''
    internal_api = APIClient(var_prefix='INTERNAL_API')

    params = {'shop_id': shop_id, 'priority_camera': priority_camera}
    if start_from:
        try:
            if isinstance(start_from, list):
                start_from = datetime(*start_from)

            params['start_from'] = start_from.isoformat(timespec='seconds')

        except Exception as e:
            print(f'Invalid start time input: {start_from} — {e}')
            return False

    try:
        response = internal_api.get('get_queue_block/', params=params)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            print(f"Invalid JSON response: {response.text}")
            return False

        queue_block = data.get('results')
        if not queue_block:
            print('No clips in the queue')
            return None
        return queue_block

    except requests.exceptions.RequestException as e:
        print(f'Error making request: {e}')
        return False


def clear_queue_block(shop_id, timestamp) -> None:
    internal_api = APIClient(var_prefix='INTERNAL_API')

    payload = {
        'action': 'clear_section',
        'shop_id': shop_id,
        'timestamp': timestamp.isoformat()
    }
    response = internal_api.post('update_queue/', json=payload)

    if response.status_code == 200:
        print('Successfully cleared queue block')
    else:
        print(f'Failed posting to internal API: {response.text}')
        print(response.status_code) 


def post_events_to_webapp(
        time_prefix, db_path='../files/data.db'
    ) -> Union[bool, None]:

    webapp_api = APIClient(var_prefix='WEBAPP_API')

    df = utils.create_track_df(time_prefix)
    df = utils.merge_track_records(df)

    shop_uuid = get_shop(db_path)[0]

    data = {
        'shop_id': [],
        'employee_id': [],
        'event': [],
        'start_time': [],
        'duration': [],
        'image': []
    }

    for _, row in df.iterrows():

        # Entry event
        data['shop_id'].append(shop_uuid)
        data['employee_id'].append(row['identity'])
        data['event'].append('workspace_entry')
        data['start_time'].append(str(row['start_time']))
        data['duration'].append(0)
        data['image'].append(row['start_img'])

        # Exit event
        data['shop_id'].append(shop_uuid)
        data['employee_id'].append(row['identity'])
        data['event'].append('workspace_exit')
        data['start_time'].append(str(row['end_time']))
        data['duration'].append(0)
        data['image'].append(row['end_img'])

    response = webapp_api.post('save_employee_event_logs/', json=data)
    
    if response.status_code == 200:
        print(f"Success: posted {len(data['event']) / 2} tracks")
        clear_track_info(time_prefix)
        return True
    else:
        print(f"Failed posting to webapp: {response.text}")
        print(response.status_code)
        return False
