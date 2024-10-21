import math
import csv
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import os
from tracker_utils import init_kf_args, Track
import input_output as io_utils
from datetime import datetime
import h5py
import torch
import torch.nn.functional as F


def get_key_given_index(i, map):
    for k, v in map.items():
        if v == i:
            return k


def centroid(coordinates):
    if (coordinates is not None):
        x = coordinates[0] + coordinates[2] / 2
        y = coordinates[1] + coordinates[3] / 2
    return x, y


def cos_sim(embedding1, embedding2):
    embedding1 = torch.tensor(embedding1)
    embedding2 = torch.tensor(embedding2)
    embedding1 = embedding1.unsqueeze(0) if embedding1.dim() == 1 else embedding1
    embedding2 = embedding2.unsqueeze(0) if embedding2.dim() == 1 else embedding2
    sim_tensor = F.cosine_similarity(embedding1, embedding2, dim=1)
    return sim_tensor.item()


def euclidean_distance(xy_centroids):
    x_1, y_1 = xy_centroids[0]
    x_2, y_2 = xy_centroids[1]
    delta_x = x_2 - x_1
    delta_y = y_2 - y_1
    return math.sqrt(delta_x**2 + delta_y**2)


def framewise_transforms(frame_boxes):
    frame_centroids = []
    for box in frame_boxes:
        frame_centroids.append(centroid(box))
    return frame_centroids


def filter_rows(adj_matrix):
    coordinates = []
    unique_cols = set()
    keep = []
    filtered_matrix = []

    for r in range(len(adj_matrix)):
        row = adj_matrix[r]
        if np.isfinite(row).sum() == 1:
            c = int(np.where(row != float('inf'))[0][0])
            coordinates.append((r, c))
            unique_cols.add(c)

    for c in unique_cols:
        rows_w_finite_vals = [rc[0] for rc in coordinates if rc[1] == c]
        min_val_row = min(rows_w_finite_vals, key=lambda r: adj_matrix[r, c])
        keep.append(min_val_row)

    all_rows = set(range(adj_matrix.shape[0]))
    used_rows = set(rc[0] for rc in coordinates)

    unused = list(all_rows - used_rows)
    keep.extend(unused)
    keep = sorted(keep)

    for i in keep:
        filtered_matrix.append(adj_matrix[i])

    return np.array(filtered_matrix), keep


def highest_similarity(trk, embedding):
    highest = -1
    for trk_embedding in trk.embeddings:
        sim = cos_sim(trk_embedding, embedding)
        if sim > highest:
            highest = sim
    return highest


def distance_costs(detections, trks):
    trk_ids = sorted(trks.keys())
    t_cntrs = []
    for id in trk_ids:
        t_cntrs.append(trks[id].x.tolist()[:2])
    d_cntrs = framewise_transforms(detections)

    diagonal = np.sqrt(1920**2 + 1080**2)

    rows = len(t_cntrs)
    cols = len(d_cntrs)
    cost_matrix = [[float('inf')] * cols for _ in range(rows)]

    cost_threshold = 1000
    for i, c2 in enumerate(t_cntrs):
        for j, c1 in enumerate(d_cntrs):
            euc = euclidean_distance((c1, c2))
            if euc < cost_threshold:
                normalized = euc / diagonal
                cost_matrix[i][j] = normalized

    return np.array(cost_matrix)


def similarity_costs(embeddings, trks):
    trk_ids = sorted(trks.keys())

    rows = len(trk_ids)
    cols = len(embeddings)
    cost_matrix = [[float('inf')] * cols for _ in range(rows)]

    for i, id in enumerate(trk_ids):
        for j, emb in enumerate(embeddings):
            cost = 1 - highest_similarity(trks[id], emb)
            cost_matrix[i][j] = cost

    return np.array(cost_matrix)


def construct_cost_matrix(detections, embeddings, trks, weights=[1, 0.1]):
    distance_matrix = distance_costs(detections, trks)
    similarity_matrix = similarity_costs(embeddings, trks)
    if (f_num >= 13738) and (f_num <= 13742):
        print(distance_matrix)
        print(similarity_matrix)
    weighted_distances = distance_matrix * weights[0]
    weighted_similarities = similarity_matrix * weights[1]

    return weighted_distances + weighted_similarities


def optimal_assignments(cost_matrix):
    assignments_dict = {}

    viable_rows = ~np.isinf(cost_matrix).all(axis=1)
    viable_cols = ~np.isinf(cost_matrix).all(axis=0)

    if viable_rows.any() and viable_cols.any():
        try:
            filtered_matrix = cost_matrix[np.ix_(viable_rows, viable_cols)]

            if filtered_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(filtered_matrix)

                orig_row_ind = np.where(viable_rows)[0]
                orig_col_ind = np.where(viable_cols)[0]

                for i, j in zip(row_ind, col_ind):
                    orig_row = orig_row_ind[i]
                    orig_col = orig_col_ind[j]
                    assignments_dict[orig_row] = orig_col

        except Exception as e:
            print(f'Exception: {e}')

            filtered_matrix, keep = filter_rows(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(filtered_matrix)

            orig_row_ind = [keep[i] for i in row_ind]
            for i, j in zip(orig_row_ind, col_ind):
                assignments_dict[i] = j

    return assignments_dict


def match_and_update(measurements, trks, f_num, min_lifespan=15):

    trk_ids = sorted(trks.keys())
    detections, embeddings = measurements

    cost_matrix = construct_cost_matrix(detections, embeddings, trks,
                                        weights=[1, .1])
    assignments = optimal_assignments(cost_matrix)

    matched = []
    unmatched_trks = []
    for trk_index, measurement_index in assignments.items():
        id = trk_ids[trk_index]
        trk = trks[id]

        unmatched_trks.append(id)

        matched.append(measurement_index)

        box = detections[measurement_index]
        measurement = np.array([centroid(box)[0], centroid(box)[1], box[2], box[3]])

        trk.update(measurement)
        trk.add_detection(box, f_num)
        trk.add_embedding(embeddings[measurement_index])

    # for id in unmatched_trks:
    #     min_lifespan = int(round(min_lifespan / (f_num - trks[id].first_detection_frame), 0)) # accounts for stride
    #     if len(trks[id].detections.keys()) < min_lifespan:
    #         del trks[id]


    unmatched_det = [detections[j] for j in range(len(detections)) if j not in matched]
    embeddings = [embeddings[j] for j in range(len(embeddings)) if j not in matched]

    return trks, unmatched_det, embeddings


def init_trks(active_trks, trk_id, detections, embeddings, f_num):
    for i, box in enumerate(detections):
        x, y = centroid(box[:4])
        w, h = box[2:4]
        kf_args = init_kf_args(cntr=[x, y], wh=[w, h], vel=[5, 5, 0, 0],
                               accvar=[50, 50, 50, 50], ewh=[5, 5],
                               mvar=[500, 500, 500, 500], evel=[5, 5, 0, 0],
                               dt=0.5)

        active_trks[trk_id] = Track(box, embeddings[i], f_num, *kf_args)
        trk_id += 1

    return active_trks, trk_id


def predict_or_cache(active_trks, track_cache, f_num):
    ks = []
    for k, v in active_trks.items():
        if (f_num - v.last_detection_frame) < 90:
            v.predict()
            v.add_state(v.x, f_num)
        else:
            track_cache[k] = v
            ks.append(k)
    for k in ks:
        del active_trks[k]


if __name__ == '__main__':
    strides = [1, 2, 3]
    cams = [2]

    location = 'CP_Sacramento'
    timestamp = '2024-08-12_08:35:57'
    # timestamp = '2024-08-12_08_35_57'
    colors = {0: (255, 0, 0), 1: (0, 255, 0)}

    for stride in strides:
        for i in cams:
            # video = f'{location}_{timestamp}_{i}'
            video = f's{stride}_{location}_{timestamp}_{i}'
            vid = f'{location}_{timestamp}_{i}'
            base_path = '../intermediate_output'
            detections = io_utils.read_detection_csv(f'{base_path}/{video}_detections.csv')
            # emb_file = f'{base_path}/{video}_embeddings.hdf5'
            emb_file = f'{base_path}/{video}_embeddings.hdf5'

            # cap = cv2.VideoCapture(f'../input_files/{video}.mp4')
            cap = cv2.VideoCapture(f'../input_files/{vid}.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            detection_frames = sorted(detections.keys())

            active_trks = {}
            trk_cache = {}
            trk_id = 0

            start = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            f_num = start

            while True:
                if ((f_num - start) % 100) == 0:
                    print(f'{f_num - start} frames processed')

                ret, frame = cap.read()
                if not ret:
                    break

                if (f_num - start) % stride == 0:
                    numtracks = len(active_trks.keys())

                    if numtracks > 0:
                        predict_or_cache(active_trks, trk_cache, f_num)

                    try:
                        d_bxs = detections[f_num]
                        embeddings = io_utils.get_embeddings(emb_file, f_num)
                    except KeyError:
                        f_num += 1
                        continue

                    numtracks = len(active_trks.keys())
                    if numtracks > 0:
                        active_trks, unmatched, embeddings = match_and_update([d_bxs, embeddings],
                                                                            active_trks, f_num)
                    else:
                        unmatched = d_bxs

                    active_trks, trk_id = init_trks(active_trks, trk_id, unmatched,
                                                    embeddings, f_num)

                f_num += 1


            cap.release()

            all_trks = {**active_trks, **trk_cache}
            del active_trks
            del trk_cache
            span = [start, f_num - 1]

            io_utils.write_trk_data(location, timestamp, i, all_trks, span, stride=stride)