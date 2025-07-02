# standard dependencies
import os
import re
import getpass
import subprocess
import gc
import uuid
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
import errno

# 3rd-party dependencies
import numpy as np
import pandas as pd
import h5py
import cv2
import av
import torch
import boto3
from botocore.exceptions import EndpointConnectionError, NoCredentialsError
from botocore.config import Config
import requests
import psutil

# internal dependencies
from utilities import general_utils as utils
from utilities import conn_utils
from utilities.conn_utils import APIClient


# =============================================================================
#                           - MEMORY MANAGEMENT -
# -----------------------------------------------------------------------------


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def cleanup_semaphores(logger):
    '''
    Removes unused (stale or leaked) semaphores:

    - POSIX-named semaphores from /dev/shm/
    - SysV IPC semaphores using ipcs -s 
    '''
    try:
        posix_semaphores = [
            file for file in os.listdir('/dev/shm/')
            if file.startswith('sem.')
        ]
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
                    logger.info(f'Skipping {sem_path} (already removed)')
                except Exception as e:
                    logger.error(f'Error removing {sem_path}: {e}')
        else:
            logger.info('No unused POSIX semaphores found')

    except Exception as e:
        logger.error(f'Error checking POSIX semaphores: {e}')

    user = getpass.getuser()
    try:
        output = subprocess.check_output(['ipcs', '-s']).decode('utf-8')

        lines = output.strip().split('\n')
        removed = 0

        for line in lines[3:]:  # skip headers
            parts = line.split()
            if len(parts) < 5:
                continue
            sem_id, owner = parts[1], parts[2]
            if owner == user:
                os.system(f'ipcrm -s {sem_id}')
                logger.info(f'Removed unused SysV semaphore: {sem_id}')
                removed += 1
        if removed == 0:
            logger.info('No unused SysV semaphores found')

    except Exception as e:
        logger.error(f'Error checking SysV semaphores: {e}')


# =============================================================================
#                           - LOCAL FILES -
# -----------------------------------------------------------------------------


def remove_files(
        file_paths: list[str] | str,
        missing_ok: bool = True,
        verbose: bool = True
) -> int:
    file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
    total_removed = 0

    for file_path in file_paths:
        try:
            os.remove(file_path)
            total_removed += 1
        except FileNotFoundError as missing:
            if missing_ok == False:
                raise FileNotFoundError(
                    errno.ENOENT, missing.strerror, missing.filename
                )
            else:
                if verbose == True:
                    print(f'{missing.strerror}: {missing.filename}')
        except Exception as e:
            if verbose == True:
                print(f'Error removing file: {e}')

    return total_removed


def get_project_root() -> str:
    '''Gets the absolute path of the project root.'''

    current_path = os.path.abspath(os.path.dirname(__file__))
    
    while current_path != os.path.dirname(current_path):
        if os.path.exists(os.path.join(current_path, 'setup.py')):
            return current_path

        current_path = os.path.dirname(current_path)
    
    raise RuntimeError('Project root not found')


def get_common_dirs(project_root: Optional[str] = None) -> dict:
    '''
    Gets the absolute paths of frequently-used directories within the project
    that functions & pipelines regularly read/write data to.

    Args:
        project_root (str): The absolute path of the project root. If you have
            already stored this value in the current context, you may pass it
            here for efficiency to avoid running get_project_root() again
            unnecessarily.
    '''
    project_root = project_root or get_project_root()

    input_dir = os.path.join(project_root, 'files/input/')
    output_dir = os.path.join(project_root, 'files/output/')
    
    event_imgs_dir = os.path.join(output_dir, 'event_imgs/')
    runtime_data_dir = os.path.join(output_dir, 'runtime_data')

    model_weights_dir = os.path.join(project_root, 'models/weights/')

    return {
        'project_root': project_root,
        'input_dir': input_dir,
        'output_dir': output_dir,
        'event_imgs_dir': event_imgs_dir,
        'runtime_data_dir': runtime_data_dir,
        'weights_dir': model_weights_dir,
    }


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
    '''
    Returns the full path to a unique filename, appending _1, _2, etc. if needed
    to ensure uniqueness.
    '''
    filename, ext = os.path.splitext(base_name)
    counter = 1
    new_name = base_name
    file_path = os.path.join(dir_path, new_name)

    while os.path.exists(file_path):
        new_name = f"{filename}_{counter}{ext}"
        file_path = os.path.join(dir_path, new_name)
        counter += 1

    return file_path


def get_unique_subdir(dir_path, base_name):
    '''
    Returns the full path to a unique directory name, appending _1, _2, etc. if
    needed to ensure uniqueness.
    '''
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


def clear_local_files(
        target_file_prefix: Optional[str] = None,
        target_extensions: Optional[list] = None,
        target_dirs: Optional[list[str]] = None,
        skip_suffixes: Optional[list[str]] = [],
) -> int:
    if not target_dirs:
        dir_paths = get_common_dirs()
        target_dir_names = [
            'input_dir',
            'output_dir',
            'event_imgs_dir',
        ]
        target_dirs = [dir_paths[name] for name in target_dir_names]
    
    skip_suffixes += ['_tracking_pipeline.pkl']

    target_paths = []
    for dir_path in target_dirs:
        if not os.path.exists(dir_path):
            print(f'Skipping {dir_path}, no such directory')
            continue

        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                skip = False
                for suffix in skip_suffixes:
                    if file_path.endswith(suffix):
                        skip = True
                        break
                if skip:
                    continue
            else:
                continue

            if target_file_prefix:
                if not filename.startswith(target_file_prefix):
                    continue
            if target_extensions:
                file_extension = utils.parse_filename(filename)[-1]
                if file_extension not in target_extensions:
                    continue

            target_paths.append(file_path)

    total_removed = remove_files(target_paths, missing_ok=True)
    print(f'Successfully deleted {total_removed} files')

    return total_removed


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
        get_project_root(), 'files/output/', 'event_imgs/'
    )
    file_path = os.path.join(event_imgs_dir, object_key)

    cv2.imwrite(file_path, img)

    try:
        s3_client = conn_utils.s3_connect(region, credentials)
        s3_client.upload_file(file_path, bucket_name, object_key)

        remove_files(file_path, missing_ok=True)

        return object_key
    except (EndpointConnectionError, NoCredentialsError) as e:
        print(f'Unable to connect: {e}')
    except Exception as e:
        print(f'Unexpected error during upload: {e}')


def extract_and_save_crops(
    event_imgs_df: pd.DataFrame,
    video_paths: dict[int, str],
    credentials: tuple[str, str],
):
    for cam_id, cam_df in event_imgs_df.groupby('cam_id'):
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
                print('Cropping/saving...')
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


def save_global_id_event_imgs(
        time_prefix, presence_df, face_data, trk_dets, credentials
):
    project_root = get_project_root()

    print(f'Track detections: {len(trk_dets)} rows')

    output = []

    present_idents = presence_df[presence_df['present_flag']]['identity'].values
    face_data = face_data[face_data['identity'].isin(present_idents)].copy()

    face_data['cam_id'] = face_data['cam_id'].astype(int)
    trk_dets['cam_id'] = trk_dets['cam_id'].astype(int)

    print(f'{len(face_data.groupby("identity"))} identities')
    
    for ident, id_faces in face_data.groupby('identity'):
        f_min, f_max = id_faces['f'].min(), id_faces['f'].max()
        
        f_split = f_min + (f_max - f_min) // 2

        early_faces = id_faces[id_faces['f'] <= f_split]
        late_faces  = id_faces[id_faces['f'] > f_split]

        if early_faces.empty:
            early_faces = id_faces[id_faces['f'] == f_min]
        if late_faces.empty:
            late_faces = id_faces[id_faces['f'] == f_max]

        if early_faces.empty and late_faces.empty:
            print('Both early and late faces empty')
            continue

        if early_faces.empty:
            early_faces = late_faces
        if late_faces.empty:
            late_faces = early_faces

        best_early_face = early_faces.sort_values('distance').iloc[0]
        best_late_face  = late_faces.sort_values('distance').iloc[0]

        for event, face_row in zip(['entry', 'exit'], [best_early_face, best_late_face]):
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
                overlap = utils.compute_overlap_ratio(face_box, trk_box)
                if overlap > best_overlap:
                    best_overlap, best_trk = overlap, trk_row
            print(f'{ident} [{event}] → max_overlap={best_overlap:.2f}, trk_found={best_trk is not None}, frame={fnum}, cam={cam}, candidates={len(candidates)}')

            if best_trk is not None:
                output.append({
                    'identity'      : ident,
                    'f'             : int(best_trk['f']),
                    'cam_id'        : int(best_trk['cam_id']),
                    'x'             : int(best_trk['x']),
                    'y'             : int(best_trk['y']),
                    'w'             : int(best_trk['w']),
                    'h'             : int(best_trk['h']),
                    'event'         : event,
                    'image'         : f'{uuid.uuid4()}.jpg',
                    'overlap_ratio' : best_overlap,
                })
    event_imgs_df = pd.DataFrame(output)

    if event_imgs_df.empty:
        print('No event crops to save.')
        return pd.DataFrame(columns=[
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

    video_paths = {
        cam_id: os.path.join(
            project_root, 'files/input', f'{time_prefix}_{cam_id}.mp4'
        )
        for cam_id in event_imgs_df['cam_id'].unique()
    }
    extract_and_save_crops(event_imgs_df, video_paths, credentials)

    print(f'Generated {len(event_imgs_df)} global ID crops across {len(video_paths)} cameras')

    return event_imgs_df


def upload_file(
        s3_client, bucket_name: str, file_path: str, object_key: str
) -> bool:
    try:
        s3_client.upload_file(file_path, bucket_name, object_key)
        return True
    except Exception as e:
        print(f'Failed to upload {file_path}: {e}')
        return False


def upload_data(credentials, max_workers=8):
    output_dir = os.path.join(get_project_root(), 'files/output')
    try:
        config = Config(
            region_name='us-west-1',
            max_pool_connections=max_workers * 3
        )
        session = boto3.session.Session()
        s3_client = session.client(
            's3',
            aws_access_key_id=credentials[0],
            aws_secret_access_key=credentials[1],
            config=config,
        )
        bucket_name = 'visionservice-data'

        upload_tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        continue
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


class S3UploadError(Exception):
    """Generic error raised when an S3 upload fails."""
    pass


class S3DownloadError(Exception):
    """Generic error raised when an S3 upload fails."""
    pass


def download_s3_footage(
        object_keys: list[str] | str,
        credentials: Optional[tuple[str, ...]] = None,
        region: str = 'us-west-1',
        bucket_name: str = 'ivakt-footage',
) -> bool:
    s3_client = conn_utils.s3_connect(region, credentials)
    project_root = get_project_root()

    object_keys = [object_keys] if isinstance(object_keys, str) else object_keys
    successfully_downloaded = []

    for object_key in object_keys:
        download_status = False

        filename = utils.parse_obj_key(object_key)[-1]
        local_path = os.path.join(project_root, 'files/input', filename)

        try:
            s3_client.download_file(bucket_name, object_key, local_path)
            download_status = True
            print(f'Downloaded {filename}')

        except Exception as e:
            print(f'Failed to download {filename}: {e}')
            remove_files(local_path, missing_ok=True)

        successfully_downloaded.append(download_status)

    return all(successfully_downloaded)


def delete_s3_footage(
        object_keys: list[str] | str,
        credentials: Optional[tuple[str, ...]] = None,
        region: str = 'us-west-1',
        bucket_name: str = 'ivakt-footage',
) -> bool:
    object_keys = [object_keys] if isinstance(object_keys, str) else object_keys
    s3_client = conn_utils.s3_connect(region, credentials)
    
    all_successful = False
    try:
        object_keys = [{'Key': k} for k in object_keys]

        response = s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={'Objects': object_keys}
        )
        deleted = response.get('Deleted', [])
        errors = response.get('Errors', [])

        for obj in deleted:
            _, filename = utils.parse_obj_key(obj['Key'])
            print(f'Deleted {filename} from S3')
    
        if errors:
            for err in errors:
                _, filename = utils.parse_obj_key(err['Key'])
                print(f'Failed to delete {filename}: {err["Message"]}')
        else:
            all_successful = True

    except Exception as e:
        print(f'Failed to delete from S3: {e}')
    
    return all_successful


def download_s3_image(
        object_key,
        credentials=None,
        filename=None,
        img_dir='files/input/faces/',
        bucket_name='ivakt-employee-photos',
) -> bool:
    s3_client = conn_utils.s3_connect(
        region='us-west-1', credentials=credentials
    )
    if not filename:
        filename = object_key.split('/')[-1]
    output_path = os.path.join(get_project_root(), img_dir, filename)

    try:
        if os.path.exists(output_path):
            print('Image already saved')
            return False
        
        s3_client.download_file(bucket_name, object_key, output_path)
        print(f'Downloaded {object_key}')
        return True
    except Exception as e:
        print(f"Failed to download {object_key}: {e}")
        remove_files(output_path, missing_ok=True)
        return False


# =============================================================================
#                           - LOCAL DATABASE -
# -----------------------------------------------------------------------------


def build_database(db_name='data.db') -> None:
    db_path = os.path.join(get_project_root(), 'files/', db_name)

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
            time_prefix TEXT,
            identity TEXT,
            start_img TEXT,
            end_img TEXT,
            start_time DATETIME,
            end_time DATETIME
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shop (
            uuid TEXT PRIMARY KEY,
            shop_name TEXT NOT NULL
        );    
    ''')

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


def get_shop(db_name='data.db') -> tuple:
    db_path = os.path.join(get_project_root(), 'files/', db_name)
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    cursor.execute('''
        SELECT * FROM shop
        LIMIT 1
    ''')
    results = cursor.fetchone()

    conn_utils.close_sqlite_db(conn, cursor)
    return results


def lookup_identities(image_paths, db_name='data.db') -> list[tuple]:
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
    db_path = os.path.join(get_project_root(), 'files/', db_name)
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


def lookup_name(identity_uuid, db_name='data.db') -> tuple:
    db_path = os.path.join(get_project_root(), 'files/', db_name)
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


def identity_is_known(identity_uuid: str) -> bool:
    if identity_uuid is None:
        return False
    first, last = lookup_name(identity_uuid)
    return bool(first or last)


def get_designation(identity_uuid, db_name='data.db') -> str | None:
    db_path = os.path.join(get_project_root(), 'files/', db_name)
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


def save_track_info(time_prefix: str, target_trks: dict, fps: int = 30,
                    db_name='data.db') -> None:
    conn, cursor = conn_utils.sqlite_db_connect(os.path.join(
        get_project_root(), 'files/', db_name
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

        start_time = utils.frame_timestamp(time_prefix, start_frame, fps)
        end_time = utils.frame_timestamp(time_prefix, end_frame, fps)

        values = (
            time_prefix,
            identity,
            start_img,
            end_img,
            start_time,
            end_time,
        )
        cursor.execute(query, values)

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


def save_attendance_info(
        time_prefix: str,
        presence_df: pd.DataFrame,
        event_imgs_df: pd.DataFrame,
        segment_length: float = 5.0,
        db_name: str = 'data.db',
) -> None:
    conn, cursor = conn_utils.sqlite_db_connect(os.path.join(
        get_project_root(), 'files/', db_name
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

    start_time = utils.frame_timestamp(time_prefix)
    end_time = utils.frame_timestamp(time_prefix, end_frame, fps)

    for _, row in presence_df.iterrows():
        if not row['present_flag']:
            continue

        identity = row['identity']

        imgs = event_imgs_df[event_imgs_df['identity'] == identity]
        if not imgs.empty:
            entry_ = imgs[imgs['event'] == 'entry']
            exit_ = imgs[imgs['event'] == 'exit']

            start_img = entry_.iloc[0]['image'] if not entry_.empty else ''
            end_img = exit_.iloc[0]['image'] if not exit_.empty else ''
        else:
            start_img = end_img = ''

        values = (
            time_prefix,
            identity,
            start_img,
            end_img,
            start_time,
            end_time,
        )
        cursor.execute(query, values)

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


def get_track_info(time_prefix: str, designation: Optional[str] = None,
                   db_name: str = 'data.db') -> list[tuple]:
    db_path = os.path.join(get_project_root(), 'files/', db_name)
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


def update_track_info(time_prefix, updates, db_name='data.db') -> None:
    db_path = os.path.join(get_project_root(), 'files/', db_name)
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


def clear_track_info(identifier, db_name='data.db') -> None:
    db_path = os.path.join(get_project_root(), 'files/', db_name)
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


def save_person_data(person_data, db_name='data.db') -> None:
    def _format_filename(img_url) -> str:
        filename = img_url.rsplit('/', 1)[1]    # remove bucket/folder info
        return '.'.join(filename.rsplit('_', 1)[:2])    # format file extension

    credentials = conn_utils.get_aws_credentials()
    
    project_root = get_project_root()

    db_path = os.path.join(project_root, 'files/', db_name)
    conn, cursor = conn_utils.sqlite_db_connect(db_path)

    img_dir = os.path.join(project_root, 'files/input/', 'faces/')

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
                object_key, credentials, filename=filename
            )

    conn_utils.close_sqlite_db(conn, cursor, commit=True)


# =============================================================================
#                         - API/REMOTE DATABASE -
# -----------------------------------------------------------------------------


def get_api_tokens(credentials: dict = None) -> tuple[str | None, ...]:
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
        shop_uuid: str = None,
        access_token: str = None,
        save_data: bool = True,
        db_name: str = 'data.db',
) -> list:
    shop_uuid = shop_uuid or get_shop(db_name=db_name)[0]
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
            save_person_data(person_data, db_name=db_name)
    else:
        person_data = []
        print(f'Error: {response.status_code}: {response.text}')
    
    return person_data


def get_queue_block(
        shop_id: str,
        start_from: list | datetime = None,
        priority_camera: str = None,
) -> list[tuple] | None:
    '''
    Returns:
        queue_block (list[tuple] or None): A list of rows corresponding to each
            video file in the queue block. Each queue block row is ordered as
            follows: (id, shop_id, filename, timestamp, cam_id, uploaded) 
    '''
    queue_block = None

    internal_api = APIClient(var_prefix='INTERNAL_API')
    params = {'shop_id': shop_id, 'priority_camera': priority_camera}
    if start_from:
        try:
            if isinstance(start_from, list):
                start_from = datetime(*start_from)

            params['start_from'] = start_from.isoformat(timespec='seconds')

        except Exception:
            print(f'Invalid start time input: {start_from}. Using default instead.')

    try:
        response = internal_api.get('get_queue_block/', params=params)
        response.raise_for_status()
        try:
            data = response.json()
            queue_block = data.get('results', None)

        except ValueError:
            print(f'Invalid JSON response: {response.text}')
    except requests.exceptions.RequestException as e:
        print(f'Error making request: {e}')
    
    if not queue_block:
        print('No clips in the queue')
    
    return queue_block


def clear_queue_block(shop_id, timestamp) -> None:
    internal_api = APIClient(var_prefix='INTERNAL_API')

    payload = {
        'directive': 'clear_section',
        'shop_id': shop_id,
        'timestamp': timestamp.isoformat()
    }
    response = internal_api.post('update_queue/', json=payload)

    if response.status_code == 200:
        print('Successfully cleared queue block')
    else:
        print(f'Failed posting to internal API: {response.text}')
        print(response.status_code) 


def post_event_data(
        shop_id, time_prefix, delete_data: bool = True, logger=None
) -> bool:
    successful_post = False

    webapp_api = APIClient(var_prefix='WEBAPP_API')

    track_data_df = utils.create_track_df(time_prefix)
    track_data_df = utils.merge_track_records(track_data_df)

    event_data = {
        'shop_id': [],
        'employee_id': [],
        'event': [],
        'start_time': [],
        'duration': [],
        'image': []
    }
    entry_event, exit_event = 'workspace_entry', 'workspace_exit'

    num_tracks = 0
    for _, row in track_data_df.iterrows():
        identity = row['identity']

        start_time, end_time = [str(row[t]) for t in ('start_time', 'end_time')]
        start_image, end_image = row['start_img'], row['end_img']

        num_tracks += 1

        # Start event data:
        event_data['shop_id'].append(shop_id)
        event_data['employee_id'].append(identity)
        event_data['event'].append(entry_event)
        event_data['start_time'].append(start_time)
        event_data['duration'].append(0)
        event_data['image'].append(start_image)

        # End event data:
        event_data['shop_id'].append(shop_id)
        event_data['employee_id'].append(identity)
        event_data['event'].append(exit_event)
        event_data['start_time'].append(end_time)
        event_data['duration'].append(0)
        event_data['image'].append(end_image)

    response = webapp_api.post('save_employee_event_logs/', json=event_data)
    if response.status_code == 200:
        success_message = f'Success: posted {num_tracks} tracks'
        if logger is None:
            print(success_message)
        else:
            logger.info(success_message)

        if delete_data:
            clear_track_info(time_prefix)

        successful_post = True
    else:
        failure_messages = [
            f'Failed posting to webapp: {response.text}',
            response.status_code,
        ]
        if logger is None:
            print(failure_messages[0])
            print(failure_messages[1])
        else:
            logger.error(failure_messages[0])
            logger.error(failure_messages[1])

    return successful_post
