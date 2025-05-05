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


def get_aws_creds():
    load_dotenv()
    access_key = os.environ.get('AWS_ACCESS_KEY')
    secret_key = os.environ.get('AWS_SECRET_KEY')
    return [access_key, secret_key]


def download_s3_footage(object_key, credentials, bucket_name='ivakt-footage'):
    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials[0],
        aws_secret_access_key=credentials[1],
        region_name='us-west-1'
    )

    video_file = object_key.split('/')[-1]
    local_path = os.path.join('../files/input', video_file)

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


def download_s3_image(object_key, credentials, filename=None, img_dir='../files/input',
                      bucket_name='ivakt-employee-photos'):
    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials[0],
        aws_secret_access_key=credentials[1],
        region_name='us-west-1'
    )

    if not filename:
        filename = object_key.split('/')[-1]
    output_path = os.path.join(img_dir, filename)

    try:
        if os.path.exists(output_path):
            print('Image already saved')
            return False
        
        s3.download_file(bucket_name, object_key, output_path)
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


def build_database(db_path='../files/data.db'):
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

    conn.commit()
    conn.close()


def get_shop(db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM shop
        LIMIT 1
    ''')
    results = cursor.fetchall()[0]
    conn.close()
    return results


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
        JOIN faces ON people.id = faces.person
        WHERE faces.file IN ({placeholders});
    '''
    cursor.execute(query, tuple(filenames))
    results = cursor.fetchall()
    conn.close()
    results_map = {row[-1]: row[:-1] for row in results}

    return [results_map.get(filename) for filename in filenames]


def lookup_name(identity_uuid, db_path='../files/data.db'):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f'''
        SELECT first_name, last_name FROM people
        WHERE identity_uuid = ?;
    '''
    cursor.execute(query, (identity_uuid,))
    results = cursor.fetchone() or [identity_uuid, identity_uuid]
    conn.close()

    return results


def get_designation(identity, db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = '''
        SELECT designation FROM people
        WHERE identity_uuid = ?
        LIMIT 1;
    '''
    cursor.execute(query, (identity,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def save_track_info(time_prefix, camera, target_trks, fps=30,
                    db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
   
    for trk_id, trk in target_trks.items():
        identity = trk.identity or str(uuid.uuid4())

        start_img = trk.start_img or ''
        end_img = trk.end_img or ''

        if not start_img and not end_img:
            continue    # skip tracks with no images
        
        start_frame = trk.span[0]
        end_frame = trk.span[-1]

        start_time = utils.frame_timestamp(time_prefix, start_frame, fps)
        end_time = utils.frame_timestamp(time_prefix, end_frame, fps)

        query = '''
            INSERT INTO track_info (
                time_prefix, camera,
                track_id, identity,
                start_img, end_img,
                start_time, end_time
            )
            VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?
            )
        '''

        values = (
            time_prefix, camera,
            trk_id, identity,
            start_img, end_img,
            start_time, end_time,
        )

        cursor.execute(query, values)

    conn.commit()
    conn.close()


def get_track_info(time_prefix, designation=None, db_path='../files/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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

    cursor.execute(query, tuple(params))
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


def save_person_data(
        person_data, db_path='../files/data.db', img_dir='../files/input/faces'
    ):
    def _format_filename(img_url):
        filename = img_url.rsplit('/', 1)[1]    # remove bucket/folder info
        return '.'.join(filename.rsplit('_', 1)[:2])    # format file extension
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    column_names__people = [
        'first_name',
        'last_name',
        'designation',
        'identity_uuid',
        'shop_uuid',
    ]
    update_clause__people = ", ".join([
        f"{col}=excluded.{col}" for col in column_names__people
        if col != 'identity_uuid'
    ])
    insert_query__people = f'''
        INSERT INTO people ({", ".join(column_names__people)})
        VALUES ({", ".join(["?"] * len(column_names__people))})
        ON CONFLICT(identity_uuid) DO UPDATE SET
            {update_clause__people}
        RETURNING id
    '''
    column_names__faces = [
        'person',
        'file',
    ]
    insert_query__faces = f'''
        INSERT OR IGNORE INTO faces ({", ".join(column_names__faces)})
        VALUES ({", ".join(["?"] * len(column_names__faces))})
    '''

    for person in person_data:
        if 'is_active' in person:
            designation = 'tracked_employee' if person['is_active'] else 'untracked'
        else:
            designation = 'tracked_employee'
        cursor.execute(insert_query__people, (
            person['first_name'], person['last_name'],
            designation,
            person['uuid'], person['shop_uuid'],
        ))
        person_id = cursor.fetchone()[0]

        img_urls = [
            person['front_image'],
            person['left_image'],
            person['right_image'],
        ]
        for img_url in img_urls:
            credentials = get_aws_creds()

            parsed = urlparse(img_url)
            object_key = parsed.path.lstrip('/')
            filename = _format_filename(img_url)
            
            cursor.execute(insert_query__faces, (person_id, filename))
            download_s3_image(
                object_key, credentials, filename=filename, img_dir=img_dir
            )

    conn.commit()
    conn.close()


# =============================================================================
#                         - API/REMOTE DATABASE -
# -----------------------------------------------------------------------------


def get_api_tokens(credentials=None):
    if not credentials:
        email = input('Enter account email: ')
        password = input('Enter account password: ')

        credentials = {
            'email': email,
            'password': password
        }

    load_dotenv()
    WEBAPP_API_KEY = os.environ.get('WEBAPP_API_KEY')
    headers = {
        'x-custom-api-key': WEBAPP_API_KEY,
        'Content-Type': 'application/json'
    }

    base_url = 'https://timemanager-api-dev-b944386035a1.herokuapp.com/'
    endpoint = 'accounts/login/'

    endpoint_url = base_url + endpoint

    r = requests.post(endpoint_url, json=credentials, headers=headers)

    if r.status_code == 200:
        access_token = r.json().get('access')
        refresh_token = r.cookies.get('refresh_token')
        
        api_tokens = (access_token, refresh_token)
    else:
        api_tokens = (None, None)
        print(f'Error: {r.status_code}: {r.json()}')
    
    return api_tokens


def fetch_person_data(
        shop_uuid: str = None, access_token: str = None, save_data: bool = True,
        db_path: str = '../files/data.db', img_dir: str = '../files/input/faces'
    ) -> list:

    if not shop_uuid:
        shop_uuid, _ = get_shop(db_path=db_path)
    if not access_token:
        access_token, _ = get_api_tokens()
    
    load_dotenv()
    WEBAPP_API_KEY = os.environ.get('WEBAPP_API_KEY')

    base_url = 'https://timemanager-api-dev-b944386035a1.herokuapp.com/'
    endpoint = 'employees-json/'

    endpoint_url = f"{base_url}{endpoint}?shop_uuid={shop_uuid}"
    headers = {
        'X-Custom-API-Key': WEBAPP_API_KEY,
        'Authorization': f'Bearer {access_token}'
    }
    r = requests.get(endpoint_url, headers=headers)

    if r.status_code == 200:
        person_data = r.json().get('employees', [])
        if save_data:
            save_person_data(person_data, db_path=db_path, img_dir=img_dir)
    else:
        person_data = []
        print(f'Error: {r.status_code}: {r.text}')
    
    return person_data


def get_queue_block(shop_id, start_from=None, priority_camera=None):
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
    load_dotenv()

    base_url = 'https://ivaktvision-fe27c015e5ff.herokuapp.com/'
    endpoint = 'api/service/get_queue_block/'

    endpoint_url = base_url + endpoint

    headers = {
        'X-Custom-Api-Key': os.environ.get('INTERNAL_API_KEY'),
        'Content-Type': 'application/json'
    }

    params = {
        'shop_id': shop_id,
        'priority_camera': priority_camera
    }
    if start_from:
        try:
            if isinstance(start_from, list):
                start_from = datetime(*start_from)
            params['start_from'] = start_from.isoformat(timespec='seconds')
        except Exception as e:
            print(f'Invalid start time input: {start_from} — {e}')
            return False

    try:
        response = requests.get(endpoint_url, headers=headers, params=params)
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


def clear_queue_block(shop_id, timestamp):
    load_dotenv()

    base_url = 'https://ivaktvision-fe27c015e5ff.herokuapp.com/'
    endpoint = 'api/service/update_queue/'
    endpoint_url = base_url + endpoint
    
    headers = {
        'X-Custom-Api-Key': os.environ.get('INTERNAL_API_KEY'),
        'Content-Type': 'application/json'
    }

    payload = {
        'action': 'clear_section',
        'shop_id': shop_id,
        'timestamp': timestamp.isoformat()
    }
    response = requests.post(endpoint_url, headers=headers, json=payload)

    if response.status_code == 200:
        print('Successfully cleared queue block')
    else:
        print(f'Failed posting to internal API: {response.text}')
        print(response.status_code) 


def post_events_to_webapp(time_prefix, db_path='../files/data.db'):
    def _merge_tracks(df, max_continuation_gap=75):
        merged = []
        for identity, group in df.groupby('identity'):
            if identity == '':
                merged.extend(group.to_dict(orient='records'))
                continue

            group = group.sort_values('start_time').reset_index(drop=True)
            current = group.iloc[0].to_dict()
            for _, row in group.iloc[1:].iterrows():
                gap = (row['start_time'] - current['end_time']).total_seconds()

                if gap <= max_continuation_gap:
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

    results = get_track_info(time_prefix, designation='tracked_employee')
    if (not results) or (len(results) == 0):
        print('No tracked_employee tracks found')
        return None
    
    columns = [
        'id', 'track_id', 'camera', 'time_prefix', 'identity', 'id_method',
        'id_cost', 'start_img', 'end_img', 'id_img',  'start_time', 'end_time',
        'entry', 'exit', 'designation'
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
        print(f"Success: posted {len(data['event']) / 2} tracks")
        clear_track_info(time_prefix)
        return True
    else:
        print(f"Failed posting to webapp: {response.text}")
        print(response.status_code)
        return False


def pg_db_connect(var_prefix='PG'):
    return psycopg2.connect(
        host=os.getenv(f'{var_prefix}_HOST'),
        port=os.getenv(f'{var_prefix}_PORT'),
        user=os.getenv(f'{var_prefix}_USER'),
        password=os.getenv(f'{var_prefix}_PASSWORD'),
        dbname=os.getenv(f'{var_prefix}_NAME')
    )
