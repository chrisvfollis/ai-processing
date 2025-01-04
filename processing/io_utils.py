import h5py
import csv
import os
import sqlite3
import numpy as np
import utilities
import json
from datetime import datetime, timedelta
import uuid
import cv2
import boto3
from botocore.exceptions import EndpointConnectionError, NoCredentialsError
from dotenv import load_dotenv
from datetime import datetime
import requests
import pandas as pd


def get_config():
    config = {}
    with open('../config/primary_cameras.txt', 'r') as file:
        file = file.read()
        camera_views = [view.strip() for view in file.split(',')]
        config['primary_cameras'] = camera_views
    with open('../config/secondary_cameras.txt', 'r') as file:
        file = file.read()
        camera_views = [view.strip() for view in file.split(',')]
        config['secondary_cameras'] = camera_views
    
    config['entryways'] = {}

    all_entrances = {}
    for file in os.listdir('../config'):
        if file.endswith('entryways.csv'):
            cam = file.split('_')[0]
            entrances = {}
            with open(f'../config/{file}', 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader)

                for row in csvreader:
                    if row[0] in entrances:
                        entrances[row[0]].append(row[1:])
                    else:
                        entrances[row[0]] = [row[1:]]

            config['entryways'][cam] = entrances
    
    return config


def write_detection_csv(detection_data, video_file):
    base_path = '../intermediate_output/'
    csv_file = f'{video_file.split(".")[0]}_detections.csv'
    csv_path = os.path.join(base_path, csv_file)
    
    file = open(csv_path, 'w', newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['f', 'x', 'y', 'w', 'h', 'c'])

    for frame in sorted(detection_data.keys()):
        for detection in detection_data[frame]:
            writer.writerow([frame] + detection)

    file.close()


def read_detection_csv(csv_path):
    frame_data = {}

    file = open(csv_path, 'r')
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)

    for row in csvreader:
        f, x, y, w, h = map(int, row[0:5])
        c = round(float(row[5]), 3)

        if f not in frame_data:
            frame_data[f] = []
        frame_data[f].append([x, y, w, h, c])

    file.close()
    return frame_data


def write_keypoint_csv(keypoint_data, video_file):
    base_path = '../intermediate_output/'
    csv_file = f'{video_file.split(".")[0]}_keypoints.csv'
    csv_path = os.path.join(base_path, csv_file)

    file = open(csv_path, 'w', newline='')
    writer = csv.writer(file, delimiter=',')

    standard_cols = ['x', 'y', 'c']

    writer.writerow(['f'] + [f'{x}{i}' for i in range(17)
                             for x in standard_cols])
    
    for frame in sorted(keypoint_data.keys()):
        for detection in keypoint_data[frame]:
            writer.writerow([frame] + [x for keypoint in detection
                                       for x in keypoint])

    file.close()


def read_keypoint_csv(csv_path):
    frame_data = {}

    file = open(csv_path, 'r')
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)

    for row in csvreader:
        f = int(row[0])
        row[1::3] = list(map(lambda x: int(float(x)), row[1::3]))
        row[2::3] = list(map(lambda x: int(float(x)), row[2::3]))
        row[3::3] = list(map(float, row[3::3]))

        detection = np.array([row[i:i+3] for i in range(1,18,3)])
        frame_data.setdefault(f, []).append(detection)

    file.close()
    return frame_data


def write_face_csv(face_data, video_file):
    base_path = '../intermediate_output/'
    csv_file = f'{video_file.split(".")[0]}_faces.csv'
    csv_path = os.path.join(base_path, csv_file)

    merged_dfs = []
    for frame, dfs in face_data.items():
        valid_dfs = [df for df in dfs if not df.empty]
        if valid_dfs:
            merged_df = pd.concat(valid_dfs, ignore_index=True)
            merged_df['f'] = frame
            merged_dfs.append(merged_df)
    
    if not merged_dfs:
        return None

    full_df = pd.concat(merged_dfs, ignore_index=True)

    drop_columns = [
        'target_x', 'target_y', 'target_w', 'target_h', 'threshold'
    ]
    full_df = full_df.drop([col for col in drop_columns if col in
                            full_df.columns], axis=1)
    full_df = full_df.rename(columns={'source_x': 'x', 'source_y': 'y',
                                      'source_w': 'w', 'source_h': 'h'})
    full_df.to_csv(csv_path, index=False)


def write_embeddings(hdf5_file, embeddings, frames, box_indices):
    frames = np.array(frames)
    box_indices = np.array(box_indices)

    embeddings_array = np.stack(embeddings)

    embeddings_dataset = hdf5_file['embeddings']
    frames_dataset = hdf5_file['frames']
    box_indices_dataset = hdf5_file['box_indices']

    new_size = embeddings_dataset.shape[0] + embeddings_array.shape[0]

    embeddings_dataset.resize(new_size, axis=0)
    frames_dataset.resize(new_size, axis=0)
    box_indices_dataset.resize(new_size, axis=0)

    embeddings_dataset[-embeddings_array.shape[0]:] = embeddings_array
    frames_dataset[-frames.shape[0]:] = frames
    box_indices_dataset[-box_indices.shape[0]:] = box_indices


def read_embeddings(hdf5_file, target_frame):
    with h5py.File(hdf5_file, 'r') as file:
        frames = file['frames']
        indices = np.where(frames[:] == target_frame)[0]
        target_embeddings = file['embeddings'][sorted(indices)]
        return target_embeddings


def write_trk_data(video_file, all_trks, span):
    time_prefix, cam = utilities.parse_clip_filename(video_file)
    trk_path = ('../intermediate_output/' + f'{time_prefix}_trk_data.hdf5')

    with h5py.File(trk_path, 'a') as file:
        metadata = file.require_group('metadata')
        try:
            metadata.create_dataset('frame_span', data=span)
        except ValueError:
            prior_span = [int(item) for item in metadata['frame_span']]
            if prior_span != span:
                start = min([prior_span[0], span[0]])
                end = max([prior_span[1], span[1]])
                metadata['frame_span'][...] = [start, end]

        trks = file.require_group('tracks')
        for id, trk in all_trks.items():
            trk_group = trks.create_group(f'c{cam}_trk{id}')

            detections_group = trk_group.create_group('detections')
            det_frames, det_boxes = [], []
            for frame in sorted(trk.detections.keys()):
                det_frames.append(frame)
                det_boxes.append(trk.detections[frame])
            detections_group.create_dataset('frames', data=det_frames)
            detections_group.create_dataset('boxes', data=det_boxes)

            trk_span = [trk.first_detection_frame, trk.last_detection_frame]
            trk_group.create_dataset('trk_span', data=trk_span)


def save_track_info(time_prefix, camera, all_trks,
                    db_path='../appdata/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS track_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT,
            camera TEXT,
            time_prefix TEXT,
            identity TEXT,
            id_method TEXT,
            id_cost FLOAT,
            start_img TEXT,
            end_img TEXT,
            id_img TEXT,
            start_frame INTEGER,
            start_time DATETIME,
            end_frame INTEGER,
            end_time DATETIME,
            entry INTEGER,
            exit INTEGER
        )
    ''')
    conn.commit()
   
    for id, trk in all_trks.items():
        track_id = id
        identity = trk.identity if trk.identity is not None else str(uuid.uuid4())
        id_cost = trk.id_cost if trk.id_cost is not None else ""
        start_img = trk.start_img if trk.start_img is not None else ""
        end_img = trk.end_img if trk.end_img is not None else ""
        start_frame = trk.first_detection_frame
        start_time = utilities.frame_timestamp(time_prefix, frame=start_frame)
        end_frame = trk.last_detection_frame
        end_time = utilities.frame_timestamp(time_prefix, frame=end_frame)

        cursor.execute('''
            INSERT INTO track_info (
                track_id, camera, time_prefix, identity, id_cost, start_img,
                end_img, start_frame, start_time, end_frame, end_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (track_id, camera, time_prefix, identity, id_cost, start_img,
              end_img, start_frame, start_time, end_frame, end_time))

    conn.commit()
    conn.close()


def update_track_info(time_prefix, updates, db_path='../appdata/data.db'):
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


def get_trk_data(trk_path, cameras, min_span=0):
    metadata = {}
    all_trks = {}

    with h5py.File(trk_path, 'r') as file:
        for k in file['metadata'].keys():
            metadata[k] = [v for v in file['metadata'][k]]

        trks = file['tracks']
        for id in trks.keys():
            if id.split('_')[0] not in cameras:
                continue

            trk_span = trks[id]['trk_span'][:]
            if (trk_span[1] - trk_span[0]) >= min_span:
                all_trks[id] = {}
                all_trks[id]['trk_span'] = trk_span
                detections = dict(zip(trks[id]['detections']['frames'][:],
                                    [[int(item) for item in sublist]
                                     for sublist in
                                     trks[id]['detections']['boxes'][:]]
                                     ))
                all_trks[id]['detections'] = detections

                if trks[id].attrs.get('identity', False):
                    all_trks[id]['identity'] = trks[id].attrs['identity']
            else:
                continue

    return metadata, all_trks 


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


def save_track_continuations(video_file, last_frame, active_trks, db_path='../appdata/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS track_continuation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            track_id TEXT,
            camera TEXT,

            clip_start_time DATETIME,
            clip_end_time DATETIME,
            
            F TEXT,
            Q TEXT,
            H TEXT,
            R TEXT,
            x TEXT,
            P TEXT,
            
            last_embedding TEXT,
            last_detection_delta INTEGER 
        )
    ''')
    conn.commit()

    insert_query = '''
        INSERT INTO track_continuation (
            track_id, camera, clip_start_time, clip_end_time, 
            F, Q, H, R, x, P, last_embedding, last_detection_delta
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    time_prefix, camera = utilities.parse_clip_filename(video_file)

    clip_start_time = utilities.frame_timestamp(time_prefix)
    clip_end_time = utilities.frame_timestamp(time_prefix, frame=last_frame)

    for id, data in active_trks.items():
        F = json.dumps(data.F.tolist())
        Q = json.dumps(data.Q.tolist())
        H = json.dumps(data.H.tolist())
        R = json.dumps(data.R.tolist())
        x = json.dumps(data.x.tolist())
        P = json.dumps(data.x.tolist())
        embedding = json.dumps(data.embeddings[-1].tolist())
        last_detection_delta = data.last_detection_frame - last_frame

        cursor.execute(insert_query, (id, camera, clip_start_time,
                                      clip_end_time, F, Q, H, R, x, P,
                                      embedding, last_detection_delta))

    conn.commit()
    conn.close()


def load_track_continuations(video_file, db_path='../appdata/data.db'):
    time_prefix, camera = utilities.parse_clip_filename(video_file)
    timestamp = utilities.frame_timestamp(time_prefix)
    prev_end_cutoff = timestamp - timedelta(seconds=2.5)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = '''
        SELECT * FROM track_continuation
        WHERE clip_end_time BETWEEN ? AND ?
        AND camera = ?
    '''
    try:
        cursor.execute(query, (prev_end_cutoff, timestamp, camera))
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.OperationalError:
        return None


def save_clip_headcounts(time_prefix, sections, db_path='../appdata/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS section_headcounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            time_prefix TEXT,
            timestamp DATETIME,

            start_headcount TEXT,
            end_headcount TEXT 
        )
    ''')

    start_section = sorted(sections.keys())[0]
    end_section = sorted(sections.keys())[-1]

    start_headcount = sections[start_section]['headcount']
    end_headcount = sections[end_section]['headcount']

    timestamp = utilities.frame_timestamp(time_prefix)

    cursor.execute('''
        INSERT INTO section_headcounts (
            time_prefix, timestamp, start_headcount, end_headcount
        )
        VALUES (?, ?, ?, ?)
    ''', (time_prefix, timestamp, start_headcount, end_headcount))

    conn.commit()
    conn.close()


def get_prev_headcount(time_prefix, cutoff=60, db_path='../appdata/data.db'):
    timestamp = utilities.frame_timestamp(time_prefix)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = '''
        SELECT * FROM section_headcounts
        ORDER BY timestamp DESC
        LIMIT 1
    '''

    try:
        cursor.execute(query)
        results = cursor.fetchone()
        conn.close()
        if results:
            prev_timestamp = datetime.strptime(results[2], '%Y-%m-%d %H:%M:%S')
            if (timestamp - prev_timestamp).total_seconds() < cutoff:
                return int(results[4])
        return None
    except sqlite3.OperationalError:
        return None


def get_employee(image_path, db_path='../appdata/data.db'):
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


def save_event_image(img, img_dir='../output_files/event_imgs/'):
    if img is None:
        return None
    file_name = f'{uuid.uuid4()}.jpg'
    file_path = os.path.join(img_dir, file_name)
    cv2.imwrite(file_path, img)
    try:
        load_dotenv()
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'),
            region_name='us-west-1'
        )
        bucket_name = 'timemanager-event-imgs'
        s3_client.upload_file(file_path, bucket_name, file_name)
    except (EndpointConnectionError, NoCredentialsError) as e:
        pass

    return file_name


def get_track_events(time_prefix, db_path='../appdata/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM track_info
        WHERE time_prefix = ?
    ''', (time_prefix,))
    results = cursor.fetchall()
    conn.close()
    return results


def clear_track_info(identifier, db_path='../appdata/data.db'):
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


def post_events_to_webapp(time_prefix, db_path='../appdata/data.db'):
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

    results = get_track_events(time_prefix)
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
