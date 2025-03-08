import h5py
import os
import sqlite3
import numpy as np
import uuid
import cv2
import boto3
from botocore.exceptions import EndpointConnectionError, NoCredentialsError
from dotenv import load_dotenv
import requests
import pandas as pd
from utilities import utilities as utils
import torch
import re
import getpass
import subprocess
import gc


def clear_memory():
    import tensorflow as tf
    K = tf.keras.backend
    K.clear_session()
    torch.cuda.empty_cache()
    gc.collect()


# ----------------------------------------------------------------------------


# File Read/Write Functions:


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


def save_event_image(img, credentials, img_dir='../files/output/event_imgs/'):
    if img is None:
        return None
    file_name = f'{uuid.uuid4()}.jpg'
    file_path = os.path.join(img_dir, file_name)
    cv2.imwrite(file_path, img)
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=credentials[0],
            aws_secret_access_key=credentials[1],
            region_name='us-west-1'
        )
        bucket_name = 'timemanager-event-imgs'
        s3_client.upload_file(file_path, bucket_name, file_name)
    except (EndpointConnectionError, NoCredentialsError) as e:
        pass

    return file_name


def write_embeddings(hdf5_file, embeddings, frames, box_indices):
    embeddings_array = np.stack(embeddings)
    embeddings_dataset = hdf5_file['embeddings']

    frames = np.array(frames)
    frames_dataset = hdf5_file['frames']

    box_indices = np.array(box_indices)
    box_indices_dataset = hdf5_file['box_indices']

    new_size = embeddings_dataset.shape[0] + embeddings_array.shape[0]

    embeddings_dataset.resize(new_size, axis=0)
    frames_dataset.resize(new_size, axis=0)
    box_indices_dataset.resize(new_size, axis=0)

    embeddings_dataset[-embeddings_array.shape[0]:] = embeddings_array
    frames_dataset[-frames.shape[0]:] = frames
    box_indices_dataset[-box_indices.shape[0]:] = box_indices


def read_embeddings(hdf5_file, target_frame, device):
    with h5py.File(hdf5_file, 'r') as file:
        frames = file['frames'][:]

        indices = np.where(frames == target_frame)[0]
        if len(indices) == 0:
            print(f"Found 0 indices for frame {target_frame}")

        target_embeddings = file['embeddings'][sorted(indices)]
        target_embeddings = torch.from_numpy(target_embeddings).to(device)

        return target_embeddings


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
            elif full_path.endswith('.pkl'):
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


def cleanup_semaphores():
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
                try:
                    os.unlink(sem_path)
                    print(f'Removed unused POSIX semaphore: {sem_path}')
                except FileNotFoundError:
                    print(f'Skipped: {sem_path} already removed.')
                except Exception as e:
                    print(f'Error removing {sem_path}: {e}')
        else:
            print('No unused POSIX semaphores found.')

    except Exception as e:
        print(f'Error checking POSIX semaphores: {e}')


    user = getpass.getuser()
    try:
        output = subprocess.check_output(['ipcs', '-s']).decode('utf-8')

        sysv_semaphores = [
            line.split()[1] for line in output.split('\n') if user in line
        ]

        if sysv_semaphores:
            for sem_id in sysv_semaphores:
                os.system(f'ipcrm -s {sem_id}')
                print(f'Removing unused SysV semaphore: {sem_id}')
        else:
            print('No unused SysV semaphores found.')

    except Exception as e:
        print(f'Error checking SysV semaphores: {e}')


def download_s3_footage(object_key, credentials, bucket_name='ivakt-footage'):
    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials[0],
        aws_secret_access_key=credentials[1],
        region_name='us-west-1'
    )
    local_path = os.path.join('../files/input', object_key)

    try:
        s3.download_file(bucket_name, object_key, local_path)
        print(f'Downloaded {object_key}')
        return True
    except Exception as e:
        print(f"Failed to download {object_key}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def delete_s3_footage(object_key, credentials, bucket_name='ivakt-footage'):
    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials[0],
        aws_secret_access_key=credentials[1],
        region_name='us-west-1'
    )

    try:
        s3.delete_object(Bucket=bucket_name, Key=object_key)
        print(f'Deleted {object_key} from S3')
        return True
    except Exception as e:
        print(f"Failed to delete {object_key} from S3: {e}")
        return False


def get_aws_creds():
    load_dotenv()
    access_key = os.environ.get('AWS_ACCESS_KEY')
    secret_key = os.environ.get('AWS_SECRET_KEY')
    return [access_key, secret_key]


# ----------------------------------------------------------------------------


# Local Database Functions:


def get_shop(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM shop
        LIMIT 1
    ''')
    results = cursor.fetchall()[0]
    conn.close()
    return results



def get_employee(image_path, db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    image = image_path.split('/')[-1]

    cursor.execute('''
            SELECT uuid FROM employees
            WHERE front_image = ?
            OR left_image = ?
            OR right_image = ?
            LIMIT 1
        ''', (image, image, image))

    result = cursor.fetchone()
    conn.close()
    if len(result) > 0:
        return result[0]
    else:
        return None


def lookup_identities(image_paths, db_path='../files/data.db'):
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
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    filenames = [img_path.split('/')[-1] for img_path in image_paths]
    placeholders = ', '.join(['?'] * len(filenames))

    query = f'''
        SELECT people.*, faces.file
        FROM people
        JOIN facs ON people.id = faces.person
        WHERE faces.file = {placeholders};
    '''
    cursor.execute(query, filenames)

    results = cursor.fetchall(); conn.close()
    results_map = {row[-1]: row[:-1] for row in results}

    return [results_map.get(filename) for filename in filenames]


def build_db_schema(db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person INTEGER NOT NULL,
            file TEXT NOT NULL,
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
    conn.commit()
    conn.close()


def save_track_info(time_prefix, camera, target_trks, fps=30,
                    db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
   
    for trk_id, trk in target_trks.items():
        identity = trk.identity if trk.identity is not None else str(uuid.uuid4())
        
        start_img = trk.start_img if trk.start_img is not None else ""
        start_time = utils.frame_timestamp(
            time_prefix, frame=trk.span[0], fps=fps
        )
        end_img = trk.end_img if trk.end_img is not None else ""
        end_time = utils.frame_timestamp(
            time_prefix, frame=trk.span[1], fps=fps
        )

        cursor.execute('''
            INSERT INTO track_info (
                track_id, camera, time_prefix, identity,
                start_img, end_img, start_time, end_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trk_id, camera, time_prefix, identity,
              start_img, end_img, start_time, end_time))

    conn.commit()
    conn.close()


def get_track_info(time_prefix, db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM track_info
        WHERE time_prefix = ?
    ''', (time_prefix,))
    results = cursor.fetchall()
    conn.close()
    return results


def update_track_info(time_prefix, updates, db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for track_id, data in updates.items():
        camera, id = track_id.split('_')[0], track_id.split('_')[1].strip('trk')
        columns = sorted(data.keys())
        set_clause = ", ".join(f"{col} = ?" for col in columns)
        values = [data[col] for col in columns]

        condition = f"time_prefix = ? AND camera = ? AND track_id = ?"
        values.extend([time_prefix, camera, id])
        query = f"UPDATE track_info SET {set_clause} WHERE {condition}"

        cursor.execute(query, values)

    conn.commit()
    conn.close()


def clear_track_info(identifier, db_path='../files/data.db'):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if identifier == 'all':
            cursor.execute('DELETE FROM track_info')
        else:
            cursor.execute('''
                DELETE FROM track_info
                WHERE time_prefix = ?
            ''', (identifier,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'Unable to clear track_info: {e}')


# ----------------------------------------------------------------------------


# Remote Database Functions:


def get_queue_block():
    base_url = 'https://ivaktvision-fe27c015e5ff.herokuapp.com/'
    
    load_dotenv()
    headers = {
        'X-Custom-Api-Key': os.environ.get('INTERNAL_API_KEY'),
        'Content-Type': 'application/json'
    }

    get_queue_url = base_url + 'api/service/get_queue_block/'

    try:
        response = requests.get(get_queue_url, headers=headers)
        data = response.json()

        queue_block = data.get('results', [])
        if len(queue_block) == 0:
            print('No clips in the queue')
            return None
        else:
            return queue_block

    except requests.exceptions.RequestException as e:
        print(f'Error making request: {e}')
        return False
    except Exception as e:
        print(f'Unexpected error: {e}')
        return False


def clear_queue_block(timestamp):
    base_url = 'https://ivaktvision-fe27c015e5ff.herokuapp.com/'
    
    load_dotenv()
    headers = {
        'X-Custom-Api-Key': os.environ.get('INTERNAL_API_KEY'),
        'Content-Type': 'application/json'
    }

    update_queue_url = base_url + 'api/service/update_queue/'

    response = requests.post(
        update_queue_url, json={
            'action': 'clear_section', 'timestamp': timestamp.isoformat()},
        headers=headers
    )

    if response.status_code == 200:
        print("Success")
    else:
        print(f"Failed posting to internal API: {response.text}")
        print(response.status_code) 


def post_events_to_webapp(time_prefix, db_path='../files/data.db'):
    def _merge_tracks(df, max_continuation_gap=75):
        merged = []
        for identity, group in df.groupby('identity'):
            if identity == "":
                merged.extend(group.to_dict(orient="records"))
                continue

            group = group.sort_values('start_frame').reset_index(drop=True)
            current = group.iloc[0].to_dict()
            for _, row in group.iloc[1:].iterrows():
                gap = (row['start_time'] - current['end_time']).total_seconds()

                if gap <= max_continuation_gap:
                    current['end_frame'] = max(current['end_frame'], row['end_frame'])
                    current['end_time'] = max(current['end_time'], row['end_time'])
                    current['end_img'] = row['end_img']
                else:
                    merged.append(current)
                    current = row.to_dict()

            merged.append(current)

        return pd.DataFrame(merged)

    load_dotenv()
    WEBAPP_API_KEY = os.environ.get('WEBAPP_API_KEY')
    url = 'https://timemanager-api-dev-b944386035a1.herokuapp.com/save_employee_event_logs/'

    results = get_track_info(time_prefix)
    if (not results) or (len(results) == 0):
        return None
    
    columns = [
        'id', 'track_id', 'camera', 'time_prefix', 'identity', 'id_method', 
        'id_cost', 'start_img', 'end_img', 'id_img', 'start_frame', 
        'start_time', 'end_frame', 'end_time', 'entry', 'exit'
    ]

    df = pd.DataFrame(results, columns=columns)
    df['start_time'] = pd.to_datetime(df['start_time'], format='mixed')
    df['end_time'] = pd.to_datetime(df['end_time'], format='mixed')
    df = _merge_tracks(df)

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

    headers = {
        'x-custom-api-key': WEBAPP_API_KEY,
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Success")
        clear_track_info(time_prefix)
        return True
    else:
        print(f"Failed posting to webapp: {response.text}")
        print(response.status_code)
        return False
