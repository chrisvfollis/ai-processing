import math
import csv
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import os
import io_utils
from datetime import datetime
import torch
import torch.nn.functional as F

import cv2
import cv2.legacy as cv2l
import numpy as np
import utilities
import math


class KalmanFilter:
    def __init__(self, frame, F, Q, H, R, x_init, P_init, B=None, u=None):
        self.F = F  # State transition matrix
        self.Q = Q  # Process noise covariance
        self.H = H  # Measurement translation matrix
        self.R = R  # Measurement noise covariance
        self.x = x_init  # State vector
        self.P = P_init  # Estimate uncertainty
        self.I = np.eye(F.shape[0])  # Identity matrix

        self.states = {frame: x_init}

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x = utilities.restrain_boxes(self.x)

    def update(self, Z):
        Y = Z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ Y
        self.P = (self.I - K @ self.H) @ self.P
    
    def add_state(self, new_state, frame_number):
        self.states[frame_number] = new_state


class Track(KalmanFilter):
    def __init__(self, detection, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.detections = {args[0]: detection}
        self.embeddings = [embedding]
        self.first_detection_frame = args[0]
        self.last_detection_frame = args[0]

    def add_embedding(self, embedding):
        self.embeddings.append(embedding)
        self.embeddings = self.embeddings[-30:]

    def add_detection(self, new_detection, frame_number):
        self.detections[frame_number] = new_detection
        self.last_detection_frame = frame_number


def kf_args(cntr=[960, 540], wh=[200, 200], vel=[0, 0, 0, 0],
                 accvar=[10, 10, 20, 20], mvar=[5, 5, 5, 5],
                 evel=[25, 25, 25, 25], ewh=[30, 30], dt=1.0):
    
    '''
    -------------- MODEL PARAMETERS --------------
    accvar — higher magnitudes = greater process noise, so more weight on
             incoming measurements relative to the model predictions. This
             results in a greater impact on subsequent predictions.
    mvar — higher magnitudes = greater measurement noise (the R matrix), so more weight on
           the filter's predictions relative to the incoming measurements. mvar essentially
           represents what you expect the typical measurement error in pixels to be is, squared.
    
    --------------- INITIAL VALUES ---------------
    cntr — initial centroid
    wh — initial width and height
    vel — initial velocity
    evel — initial estimate velocity uncertainty
    ewh — initial estimate width and height uncertainty
    '''

    pos_var = (dt**4)/4
    vel_var = (dt**2)

    x_acc_var, y_acc_var, w_acc_var, h_acc_var = accvar

    # Q values:
    x_pvar = pos_var * x_acc_var
    y_pvar = pos_var * y_acc_var
    w_pvar = pos_var * w_acc_var
    h_pvar = pos_var * h_acc_var

    x_vvar = vel_var * x_acc_var
    y_vvar = vel_var * y_acc_var
    w_vvar = vel_var * w_acc_var
    h_vvar = vel_var * h_acc_var
    
    F = np.array([
        [1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dt/8, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dt/8],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])

    Q = np.array([
        [x_pvar, 0, 0, 0, 0, 0, 0, 0],
        [0, y_pvar, 0, 0, 0, 0, 0, 0],
        [0, 0, w_pvar, 0, 0, 0, 0, 0],
        [0, 0, 0, h_pvar, 0, 0, 0, 0],
        [0, 0, 0, 0, x_vvar, 0, 0, 0],
        [0, 0, 0, 0, 0, y_vvar, 0, 0],
        [0, 0, 0, 0, 0, 0, w_vvar, 0],
        [0, 0, 0, 0, 0, 0, 0, h_vvar]
        ])
    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
        ])
    R = np.diag(mvar)
    cntr_x, cntr_y = cntr
    vel_x, vel_y, vel_w, vel_h = vel
    w, h = wh
    x_init = np.array([cntr_x, cntr_y, w, h, vel_x, vel_y, vel_w, vel_h])
    P_init = np.diag([mvar[0], mvar[1], ewh[0], ewh[1], evel[0], evel[1],
                      evel[2], evel[3]])

    return F, Q, H, R, x_init, P_init


def construct_cost_matrix(detections, embeddings, trks, weights=[1, 0.1]):
    '''
    Creates a cost matrix based on a weighted sum of geometric costs and
    embedding similarity costs.
    '''

    def _distance_costs(detections, trks):
        trk_ids = sorted(trks.keys())
        t_cntrs = []
        for id in trk_ids:
            t_cntrs.append(trks[id].x.tolist()[:2])
        d_cntrs = utilities.get_centroids(detections)

        diagonal = np.sqrt(1920**2 + 1080**2)

        rows = len(t_cntrs)
        cols = len(d_cntrs)
        cost_matrix = [[float('inf')] * cols for _ in range(rows)]

        cost_threshold = 1000
        for i, c2 in enumerate(t_cntrs):
            for j, c1 in enumerate(d_cntrs):
                euc = utilities.euclidean_distance((c1, c2))
                if euc < cost_threshold:
                    normalized = euc / diagonal
                    cost_matrix[i][j] = normalized

        return np.array(cost_matrix)

    def _similarity_costs(embeddings, trks):
        def _highest_similarity(trk, embedding):
            highest = -1
            for trk_embedding in trk.embeddings:
                sim = utilities.cos_sim(trk_embedding, embedding)
                if sim > highest:
                    highest = sim
            return highest

        trk_ids = sorted(trks.keys())

        rows = len(trk_ids)
        cols = len(embeddings)
        cost_matrix = [[float('inf')] * cols for _ in range(rows)]

        for i, id in enumerate(trk_ids):
            for j, emb in enumerate(embeddings):
                cost = 1 - _highest_similarity(trks[id], emb)
                cost_matrix[i][j] = cost

        return np.array(cost_matrix)

    distance_matrix = _distance_costs(detections, trks)
    similarity_matrix = _similarity_costs(embeddings, trks)
    weighted_distances = distance_matrix * weights[0]
    weighted_similarities = similarity_matrix * weights[1]

    return weighted_distances + weighted_similarities


def optimal_assignments(cost_matrix):
    def _filter_sparse_rows(cost_matrix):
        '''
        This function helps ensure linear assignment is feasible on a cost
        matrix by whittling down problematic sets of sparse rows. These
        sets are characterized by the following properties:

        - Each row has only one column it could possibly be assigned to. The
        match cost between this row and any of the other columns is "inf" and thus
        not a real candidate.
        - Every row in the  is the same for all the rows in the set
        every row is discarded except whichever has the lowest match cost to
        the column.
        '''

        matrix_coordinates = []
        unique_cols = set()
        keep = []
        filtered_matrix = []

        # For any row containing one finite entry, store the entry's matrix
        # coordinates and add its column index to unique_cols:
        for r in range(len(cost_matrix)):
            row = cost_matrix[r]
            if np.isfinite(row).sum() == 1:
                c = int(np.where(row != float('inf'))[0][0])
                matrix_coordinates.append((r, c))
                unique_cols.add(c)

        # For each column index from the relevant entries identified above,
        # add the row index of the minimum-value entry in that column:
        for c in unique_cols:
            rows_w_finite_vals = [rc[0] for rc in matrix_coordinates if rc[1] == c]
            min_val_row = min(rows_w_finite_vals, key=lambda r: cost_matrix[r, c])
    
            keep.append(min_val_row)

        all_rows = set(range(cost_matrix.shape[0]))
        used_rows = set(rc[0] for rc in matrix_coordinates)

        unused = list(all_rows - used_rows)
        keep.extend(unused)
        keep = sorted(keep)

        for i in keep:
            filtered_matrix.append(cost_matrix[i])

        return np.array(filtered_matrix), keep

    assignments_dict = {}

    # Construct boolean arrays denoting which rows/columns from the cost
    # matrix contain at least one viable entry (finite matching cost):
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

        except ValueError:  # This block handles linear_sum_assignment's "cost
                            # matrix is infeasible" error

            filtered_matrix, keep = _filter_sparse_rows(cost_matrix)
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
        c_x, c_y = utilities.centroid(box)
        w, h = box[2:4]
        measurement = np.array([c_x, c_y, w, h])

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
        x, y = utilities.centroid(box[:4])
        w, h = box[2:4]
        kf_args = kf_args(cntr=[x, y], wh=[w, h], vel=[5, 5, 0, 0],
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


def track(file, stride, previous_data=None):
    start = 0
    end = int(cv2.VideoCapture(f'../input_files/{file}')
              .get(cv2.CAP_PROP_FRAME_COUNT)) + 1

    file = file.split('.')[0]
    base_path = '../intermediate_output'
    detections = io_utils.read_detection_csv(f'{base_path}/{file}_detections.csv')
    emb_file = f'{base_path}/{file}_embeddings.hdf5'

    active_trks = {}
    trk_cache = {}
    trk_id = 0

    for f_num in range(start, end):
        # if ((f_num - start) % 100) == 0:
        #     print(f'{f_num - start} frames processed')

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

    all_trks = {**active_trks, **trk_cache}
    del active_trks
    del trk_cache
    span = [start, end - 1]

    return all_trks, span
