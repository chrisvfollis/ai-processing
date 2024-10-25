import h5py
import csv
import matplotlib.pyplot as plt
import os
import sqlite3
import torch
import numpy as np
import utilities


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


def write_detection_csv(frame_data, clip):
    new_csv_path = f'../intermediate_output/{clip.split(".")[0]}_detections.csv'
    with open(new_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Frame', 'X', 'Y', 'W', 'H', 'Conf'])
        for frame in sorted(frame_data.keys()):
            for det in frame_data[frame]:
                csvwriter.writerow([frame] + det)


def write_embeddings_hdf5(hdf5_file, embeddings, frames, box_indices):
    frames = np.array(frames)
    box_indices = np.array(box_indices)

    # Stack the embeddings into a numpy array for saving
    embeddings_array = np.stack(embeddings)

    # Retrieve existing datasets from the file
    embeddings_dataset = hdf5_file['embeddings']
    frames_dataset = hdf5_file['frames']
    box_indices_dataset = hdf5_file['box_indices']

    # Calculate new size after appending
    new_size = embeddings_dataset.shape[0] + embeddings_array.shape[0]

    # Resize the datasets to accommodate new data
    embeddings_dataset.resize(new_size, axis=0)
    frames_dataset.resize(new_size, axis=0)
    box_indices_dataset.resize(new_size, axis=0)

    # Append new data
    embeddings_dataset[-embeddings_array.shape[0]:] = embeddings_array
    frames_dataset[-frames.shape[0]:] = frames
    box_indices_dataset[-box_indices.shape[0]:] = box_indices


def read_detection_csv(csv_path):
    frame_data = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)

        for row in csvreader:
            f, x, y, w, h = map(int, row[0:5])
            c = round(float(row[5]), 3)

            if f not in frame_data:
                frame_data[f] = []

            frame_data[f].append([x, y, w, h, c])
    return frame_data


def read_segmentbox_csv(csv_path):
    bbox_data = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)

        for row in csvreader:
            f, t, x, y, w, h = map(int, row)
            if f not in bbox_data:
                bbox_data[f] = {t: [x, y, w, h]}
            else:
                bbox_data[f][t] = [x, y, w, h]
    return bbox_data


def read_entryway_event_csv(csv_path):
    entryway_events = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)

        for row in csvreader:
            f, d, e = map(int, row)
            if f not in entryway_events:
                entryway_events[f] = [[d, e]]
            else:
                entryway_events[f].append([d, e])
    return entryway_events


def get_embeddings(file_name, target_frame):
    with h5py.File(file_name, 'r') as file:
        frames = file['frames']
        indices = np.where(frames[:] == target_frame)[0]
        target_embeddings = file['embeddings'][sorted(indices)]
        return target_embeddings


def get_entryway_events(file_path):
    entryway_events = {}
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)

        for row in csvreader:
            f, e, x, y, w, h = map(int, row)
            if f not in entryway_events:
                entryway_events[f] = [[x, y, w, h]]
            else:
                entryway_events[f].append([x, y, w, h])
    return entryway_events


def write_trk_data(filename, all_trks, span):
    name = filename.rsplit('_', 1)[0]
    cam = filename.rsplit('_', 1)[1].split('.')[0]
    print(f'CAM: {cam}')
    file_path = ('../intermediate_output/' +
                 f'{name}_trk_data.hdf5')

    with h5py.File(file_path, 'a') as file:

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


def save_trk_data(filename, end, trk_data, db_path='../appdata/data.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS track_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT NOT NULL,
            camera TEXT NOT NULL,

            clip_start_time DATETIME NOT NULL,
            clip_end_frame INTEGER NOT NULL,

            identity TEXT NOT NULL,

            d_start_frame INTEGER NOT NULL,
            d_start_time DATETIME NOT NULL,
            
            kf_end_frame INTEGER NOT NULL,
            kf_end_time DATETIME NOT NULL,
            kf_end_box TEXT NOT NULL
        )
    ''')
    conn.commit()
   
    clip_timestamp = filename.rsplit('_', 1)[0]
    clip_start_time = utilities.frame_timestamp(clip_timestamp, 0)
    camera = filename.rsplit('_', 1)[1].split('.')[0]

    for id, data in trk_data.items():

        d_start_f = min(data.detections.keys())
        d_start_t = utilities.frame_timestamp(clip_timestamp, d_start_f)

        kf_end_f = max(data.states.keys())
        kf_end_t = utilities.frame_timestamp(clip_timestamp, d_start_f)
        kf_end_b = str(data.states[kf_end_f])

        cursor.execute()


    conn.close()

def get_trk_data(file_path, cameras, min_span=0):
    metadata = {}
    all_trks = {}

    with h5py.File(file_path, 'r') as file:
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


def write_track_ids(file, all_trks):
    path = f'../intermediate_output/{file}_identified.csv'
    with open(path, 'w', newline='') as file:
        csvwriter = csv.writer(file, delimiter=',')
        csvwriter.writerow(['track', 'identity'])
        for trk, data in all_trks.items():
            csvwriter.writerow([trk, data.get('identity', None)])


def write_trackspans(file, all_frames, headcounts, all_trks,
                     granularity=1):

    path = f'../intermediate_output/{file}_track_spans.csv'

    trks = [trk for trk in sorted(all_trks.keys(), key=lambda x:
                                  (int(x.split('_')[0].strip('c')) * 100)
                                  + int(x.split('_')[1].strip('trk'))
                                  )]
    frames = [f for f in all_frames if (f % granularity) == 0]

    for trk, data in all_trks.items():
        f_data = [0 for _ in frames]
        start = int(round(data['trk_span'][0] / granularity, 0) * granularity)
        end = int(round(data['trk_span'][1] / granularity, 0) * granularity)

        trk_frames = [f for f in range(start, end, granularity)]

        for f in trk_frames:
            i = frames.index(f)
            f_data[i] = 1

        if data.get('entry', None):
            i = frames.index(trk_frames[0])
            f_data[i] = 'e'
        if data.get('exit', None):
            i = frames.index(trk_frames[-1])
            f_data[i] = 'e'

        all_trks[trk]['sheet'] = f_data

    with open(path, 'w', newline='') as file:
        csvwriter = csv.writer(file, delimiter=',')
        csvwriter.writerow(['time', 'frame', 'headcount'] + trks)

        for i, f in enumerate(frames):
            mins = f // (60 * 30)
            secs = int(round((f - (mins * 60 * 30)) / 30, 0))
            t = f'{mins}m{secs}s'
            h = headcounts[f]
            trk_data = [all_trks[trk]['sheet'][i] for trk in trks]
            csvwriter.writerow([t, f, h] + trk_data)


def update_identities(file_path, all_trks, reset=False):
    with h5py.File(file_path, 'r+') as file:
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
                except Exception as e:
                    print(e)
                    pass


def get_queue_block(db_path, designation='primary'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT q.* FROM queue q
        JOIN camera_designations cd ON q.camera = cd.camera
        WHERE cd.designation = ?
        AND q.file_timestamp = (
            SELECT MIN(file_timestamp)
            FROM queue q2
            JOIN camera_designations cd2 on q2.camera = cd2.camera
            WHERE cd2.designation = ?
        )
    ''', (designation, designation))

    results = cursor.fetchall()
    conn.close()
    return results


def update_queue(db_path, file, time=None, cam=None, action='add'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            camera TEXT NOT NULL,
            file_timestamp DATETIME NOT NULL,
            record_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

    if action == 'add':
        cursor.execute('''
            INSERT INTO queue (file_name, camera, file_timestamp)
            VALUES (?, ?, ?)
        ''', (file, cam, time))
    elif action == 'remove':
        cursor.execute('''
            DELETE FROM queue
            WHERE file_name = ?
        ''', (file,))

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