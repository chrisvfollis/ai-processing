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


def write_detections(frame_data, video_file):
    base_path = '../intermediate_output/'
    csv_file = f'{video_file.split(".")[0]}_detections.csv'
    csv_path = os.path.join(base_path, csv_file)
    
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Frame', 'X', 'Y', 'W', 'H', 'Conf'])
        for frame in sorted(frame_data.keys()):
            for det in frame_data[frame]:
                writer.writerow([frame] + det)


def read_detections(csv_path):
    frame_data = {}
    with open(csv_path, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        next(csvreader)

        for row in csvreader:
            f, x, y, w, h = map(int, row[0:5])
            c = round(float(row[5]), 3)

            if f not in frame_data:
                frame_data[f] = []
            frame_data[f].append([x, y, w, h, c])

    return frame_data


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


def get_embeddings(hdf5_file, target_frame):
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


def save_track_info(time_prefix, all_trks, db_path='../appdata/data.db'):
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
   

    for id, data in all_trks.items():
        camera, track_id = id.split('_')[0], id.split('_')[1].strip('trk')
        start_frame = int(min(data['detections'].keys()))
        start_time = utilities.frame_timestamp(time_prefix, frame=start_frame)
        end_frame = int(max(data['detections'].keys()))
        end_time = utilities.frame_timestamp(time_prefix, frame=end_frame)

        cursor.execute('''
            INSERT INTO track_info (
                track_id, camera, time_prefix, start_frame,
                start_time, end_frame, end_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (track_id, camera, time_prefix, start_frame, start_time, end_frame,
              end_time))

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


def update_identities(trk_path, all_trks, reset=False):
    with h5py.File(trk_path, 'r+') as file:
        trks_group = file['tracks']
        for id, data in all_trks.items():
            if not reset:
                if data.get('identity', False):
                    trk_group = trks_group[id]
                    trk_group.attrs['identity'] = data['identity']
            else:
                trk_group = trks_group[id]
                try:
                    print(trk_group.attrs['identity'])
                    del trk_group.attrs['identity']
                    print('deleted')
                except Exception:
                    pass


def get_queue_block(designation='primary', db_path='../appdata/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT q.* FROM queue q
        JOIN cameras cd ON q.camera = cd.camera
        WHERE cd.designation = ?
        AND q.timestamp = (
            SELECT MIN(timestamp)
            FROM queue q2
            JOIN cameras cd2 on q2.camera = cd2.camera
            WHERE cd2.designation = ?
        )
    ''', (designation, designation))

    results = cursor.fetchall()
    conn.close()
    return results


def update_queue(action='add', video_file=None, datetime=None, cam=None,
                 db_path='../appdata/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_file TEXT,
            camera TEXT,
            timestamp DATETIME 
        )
    ''')
    conn.commit()

    if action == 'add':
        cursor.execute('''
            INSERT INTO queue (video_file, camera, timestamp)
            VALUES (?, ?, ?)
        ''', (video_file, cam, datetime))
    elif action == 'remove':
        cursor.execute('''
            DELETE FROM queue
            WHERE video_file = ?
        ''', (video_file,))
    elif action == 'clear_section':
        cursor.execute('''
            DELETE FROM queue
            WHERE timestamp = ?
        ''', (datetime,))

    conn.commit()
    conn.close()


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


def save_track_continuations(video_file, clip_end, active_trks, db_path='../appdata/data.db'):
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
    clip_end_time = utilities.frame_timestamp(time_prefix, frame=clip_end)

    for id, data in active_trks.items():
        F = json.dumps(data.F.tolist())
        Q = json.dumps(data.Q.tolist())
        H = json.dumps(data.H.tolist())
        R = json.dumps(data.R.tolist())
        x = json.dumps(data.x.tolist())
        P = json.dumps(data.x.tolist())
        embedding = json.dumps(data.embeddings[-1].tolist())
        last_detection_delta = data.last_detection_frame - clip_end

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

    cursor.execute('SELECT uuid FROM employees WHERE front_image = ? LIMIT 1',
                   (image_path,))

    result = cursor.fetchone()
    conn.close()
    if len(result) > 0:
        return result[0]
    else:
        return None


def save_event_image(img, img_dir='../output_files/event_imgs/'):
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
        AND (entry = 1 OR exit = 1)
    ''', (time_prefix,))
    results = cursor.fetchall()
    conn.close()
    return results


def post_events_to_webapp(time_prefix, db_path='../appdata/data.db'):
    load_dotenv()
    WEBAPP_API_KEY = os.environ.get('WEBAPP_API_KEY')
    url = 'https://timemanager-api-dev-b944386035a1.herokuapp.com/save_employee_event_logs/'

    results = get_track_events(time_prefix)
    if (not results) or (len(results) == 0):
        return None

    shop_uuid = get_shop(db_path)[0]

    data = {
        'shop_id': [],
        'employee_id': [],
        'event': [],
        'start_time': [],
        'duration': [],
        'image': []
    }

    entries = [r for r in results if (r[4]) and (r[14] == 1)]
    exits = [r for r in results if (r[4]) and (r[15] == 1)]

    for entry in entries:
        data['shop_id'].append(shop_uuid)
        data['employee_id'].append(entry[4])
        data['event'].append('workspace_entry')
        data['start_time'].append(str(entry[11]))
        data['duration'].append(0)
        data['image'].append(entry[7])
    
    for exit in exits:
        data['shop_id'].append(shop_uuid)
        data['employee_id'].append(exit[4])
        data['event'].append('workspace_exit')
        data['start_time'].append(str(exit[13]))
        data['duration'].append(0)
        data['image'].append(exit[8])

    json_data = json.dumps(data)

    headers = {
        'x-custom-api-key': WEBAPP_API_KEY,
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, data=json_data, headers=headers)
    if response.status_code == 200:
        print("Success")
    else:
        print(response.text)
        print(response.status_code)
