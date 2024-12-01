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
import sqlite3
import json


class KalmanFilter:
    def __init__(self, frame, F, Q, H, R, x_init, P_init, B=None, u=None):
        self.F = F  # State transition matrix
        self.Q = Q  # Process noise covariance matrix
        self.H = H  # Measurement translation matrix
        self.R = R  # Measurement noise covariance matrix
        self.x = x_init  # State vector
        self.P = P_init  # Estimate uncertainty matrix
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
        self.face_data = {}

    def add_embedding(self, embedding, window=-20):
        self.embeddings.append(embedding)
        self.embeddings = self.embeddings[window:]

    def add_detection(self, new_detection, frame_number):
        self.detections[frame_number] = new_detection
        self.last_detection_frame = frame_number


class Tracker:
    def __init__(self, start, end, detection_data, face_data, embedding_path):
        self.f_num = start
        self.end = end

        self.active_trks = {}
        self.trk_cache = {}
        self.trk_id = 0

        self.detection_data = detection_data
        self.face_data = face_data
        self.embedding_path = embedding_path

        self.unmatched = []

    def _create_new_tracks(self):
        detections, embeddings = self.unmatched
        for i, box in enumerate(detections):
            x, y = utilities.centroid(box[:4])
            w, h = box[2:4]
            kf_args = construct_kf_args(cntr=[x, y], wh=[w, h],
                                        vel=[5, 5, 0, 0],
                                        accvar=[50, 50, 50, 50],
                                        ewh=[5, 5], mvar=[500, 500, 500, 500],
                                        evel=[5, 5, 0, 0], dt=0.5)
            new_track = Track(box[:4], embeddings[i], self.f_num, *kf_args)
            self.active_trks[self.trk_id] = new_track
            self.trk_id += 1

    def _predict_or_cache(self, threshold=90):
        cached = []
        for id, trk in self.active_trks.items():
            if (self.f_num - trk.last_detection_frame) < threshold:
                trk.predict()
                trk.add_state(trk.x, self.f_num)
            else:
                self.trk_cache[id] = trk
                cached.append(id)
        for id in cached:
            del self.active_trks[id]

    def _match_and_update(self, measurements, min_lifespan=15):
        def _construct_cost_matrix(detections, embeddings, weights=[1, 0.1]):
            '''
            Creates a cost matrix based on a weighted sum of geometric costs and
            embedding similarity costs.
            '''

            def _distance_costs(detections, cost_threshold=1000):
                trk_ids = sorted(self.active_trks.keys())
                t_cntrs = []
                for id in trk_ids:
                    t_cntrs.append(self.active_trks[id].x.tolist()[:2])
                d_cntrs = utilities.get_centroids(detections)

                diagonal = np.sqrt(1920**2 + 1080**2)

                rows = len(t_cntrs)
                cols = len(d_cntrs)
                cost_matrix = [[float('inf')] * cols for _ in range(rows)]

                for i, c2 in enumerate(t_cntrs):
                    for j, c1 in enumerate(d_cntrs):
                        euc = utilities.euclidean_distance((c1, c2))
                        if euc < cost_threshold:
                            normalized = euc / diagonal
                            cost_matrix[i][j] = normalized

                return np.array(cost_matrix)

            def _similarity_costs(embeddings):
                def _highest_similarity(trk, embedding):
                    highest = -1
                    for trk_embedding in trk.embeddings:
                        sim = utilities.cos_sim(trk_embedding, embedding)
                        if sim > highest:
                            highest = sim
                    return highest

                trk_ids = sorted(self.active_trks.keys())

                rows = len(trk_ids)
                cols = len(embeddings)
                cost_matrix = [[float('inf')] * cols for _ in range(rows)]

                for i, id in enumerate(trk_ids):
                    for j, emb in enumerate(embeddings):
                        cost = 1 - _highest_similarity(self.active_trks[id], emb)
                        cost_matrix[i][j] = cost

                return np.array(cost_matrix)

            distances = _distance_costs(detections)
            similarities = _similarity_costs(embeddings)
            weighted_matrix = (
                (distances * weights[0]) + (similarities * weights[1])
            )

            return weighted_matrix

        def _assign_matches(cost_matrix):
            def _filter_sparse_rows(cost_matrix):
                '''
                This function helps ensure linear assignment is feasible on a cost
                matrix by whittling down problematic sets of sparse rows. These
                sets are characterized by the following properties:

                - Each row has only one column it could possibly be assigned to. It is
                the one column with a finite cost value in that row; all the others
                contain "inf" due to exceeding a cost threshold.
                - The one viable column in a given row is the one viable column in
                the entire set. In other words, only one row from each set can
                ultimately be matched with a column. 
                
                Once all such sets have been identified, each is reduced to a single
                row (whichever has the lowest match cost to the viable volumn). The
                other rows from each set are filtered from the cost matrix, increasing
                the number of new tracks to subsequently initialize for this frame.
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

                # Handle cases where the "cost matrix is infeasible" error is
                # thrown during the linear_sum_assignment execution:
                except ValueError:
                    filtered_matrix, keep = _filter_sparse_rows(cost_matrix)
                    row_ind, col_ind = linear_sum_assignment(filtered_matrix)

                    orig_row_ind = [keep[i] for i in row_ind]
                    for i, j in zip(orig_row_ind, col_ind):
                        assignments_dict[i] = j

            return assignments_dict

        trk_ids = sorted(self.active_trks.keys())
        detections, embeddings = measurements

        cost_matrix = _construct_cost_matrix(detections, embeddings,
                                            weights=[1, .1])
        assignments = _assign_matches(cost_matrix)

        matched = []
        for trk_index, measurement_index in assignments.items():
            id = trk_ids[trk_index]
            trk = self.active_trks[id]

            matched.append(measurement_index)

            box = detections[measurement_index]
            c_x, c_y = utilities.centroid(box)
            w, h = box[2:4]
            measurement = np.array([c_x, c_y, w, h])

            trk.update(measurement)
            trk.add_detection(box[:4], self.f_num)
            trk.add_embedding(embeddings[measurement_index])

        self.unmatched_det = [detections[j] for j in range(len(detections))
                              if j not in matched]
        self.unmatched_emb = [embeddings[j] for j in range(len(embeddings))
                              if j not in matched]

    def run(self):
        while self.f_num < self.end:
            if self.active_trks:
                self._predict_or_cache()

            try:
                detections = self.detection_data[self.f_num]
                embeddings = io_utils.read_embeddings(self.embedding_path, self.f_num)
                measurements = (detections, embeddings)
            except KeyError:
                measurements = None

            if measurements and self.active_trks:
                self._match_and_update(measurements)
            elif measurements and (not self.active_trks):
                self.unmatched_det = detections
                self.unmatched_emb = embeddings
            
            self._init_trks()

            self.f_num += 1


        if measurements:
            if active_trks:
                match_output = _match_and_update([d_bxs, embeddings],
                                                 active_trks, f_num)
                active_trks, unmatched, embeddings = match_output
            else:
                unmatched = d_bxs

            active_trks, trk_id = _init_trks(active_trks, trk_id, unmatched,
                                            embeddings, f_num)


def kf_matrix_fmt(measurement, xy_vel=[0, 0], wh_vel=[0, 0],
                 accvar=[10, 10, 20, 20], mvar=[5, 5, 5, 5],
                 evel=[25, 25, 25, 25], ewh=[30, 30], dt=1.0):
    
    '''
    Formats 

    -------------- MODEL PARAMETERS --------------
    p_noise — these values are used to compute the process noise covariance.
              Higher magnitudes = greater process noise, meaning incoming
              measurements are given more weight relative to model predictions.
              The result is that new measurements have a larger impact on
              updating the trajectory of subsequent predictions.
    m_noise — the measurement noise covariance values. Higher magnitudes = greater measurement
           noise (the R matrix), so more weight on the filter's predictions
           relative to the incoming measurements. mvar essentially
           represents what you expect the typical measurement error in pixels to be is, squared.
    
    --------------- INITIAL VALUES ---------------
    measurement — the bounding box of the object, formatted as:
                  [center x, center y, width, height]
    xy_vel — the expected initial velocity of the object.
    wh_vel — the expected initial velocity of the bounding box dimensions.
    evel — initial estimate velocity uncertainty
    ewh — initial estimate width and height uncertainty
    '''
    
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

    # Q values:
    position_var = (dt**4)/4 # Position variance
    velocity_var = (dt**2) # Velocity variance

    x_var, y_var, w_var, h_var = accvar

    x_pvar = position_var * x_var
    y_pvar = position_var * y_var
    w_pvar = position_var * w_var
    h_pvar = position_var * h_var

    x_vvar = velocity_var * x_var
    y_vvar = velocity_var * y_var
    w_vvar = velocity_var * w_var
    h_vvar = velocity_var * h_var

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

    x_init = np.array(measurement + xy_vel + wh_vel)
    P_init = np.diag([mvar[0], mvar[1], ewh[0], ewh[1], evel[0], evel[1],
                      evel[2], evel[3]])

    return F, Q, H, R, x_init, P_init


def track(video_file):
    def _trk_continuation(video_file, stride, threshold=90, fps=30):
        continuations = io_utils.load_track_continuations(video_file)
        if (not continuations) or (len(continuations) == 0):
            return None
        
        prev_end_time = datetime.strptime(continuations[0][4], "%Y-%m-%d %H:%M:%S.%f")

        time_prefix = utilities.parse_clip_filename(video_file, data='time')
        clip_start_time = utilities.frame_timestamp(time_prefix)

        interim = round((clip_start_time - prev_end_time).total_seconds(), 0) * fps

        active_trks = {}
        for row in continuations:
            last_detection_delta = row[11]
            if (-1 * (last_detection_delta - interim)) >= threshold:
                continue
            trk_id = row[1]
            F = np.array(json.loads(row[5]))
            Q = np.array(json.loads(row[6]))
            H = np.array(json.loads(row[7]))
            R = np.array(json.loads(row[8]))
            x = np.array(json.loads(row[9]))
            P = np.array(json.loads(row[10]))
            embedding = np.array(json.loads(row[11]))

            detection = [None, None, None, None]
            trk = Track(detection, embedding, (0 - interim), F, Q, H, R, x, P)
            trk.last_detection_frame = last_detection_delta

            active_trks[trk_id] = trk
        
        trk_cache = {}
        for f in range(interim):
            if f % stride == 0:
                if len(active_trks.keys()) == 0:
                    return None
                else:
                    _predict_or_cache(active_trks, trk_cache, f)
                
        for trk in active_trks.values():
            trk.last_detection_frame -= interim
    
        return active_trks

    def _init_trks(active_trks, trk_id, detections, embeddings, f_num):
        for i, box in enumerate(detections):
            x, y = utilities.centroid(box[:4])
            w, h = box[2:4]
            kf_args = construct_kf_args(cntr=[x, y], wh=[w, h],
                                        vel=[5, 5, 0, 0],
                                        accvar=[50, 50, 50, 50],
                                        ewh=[5, 5], mvar=[500, 500, 500, 500],
                                        evel=[5, 5, 0, 0], dt=0.5)

            active_trks[trk_id] = Track(box[:4], embeddings[i], f_num, *kf_args)
            trk_id += 1
        return active_trks, trk_id

    start = 0
    end = int(cv2.VideoCapture(f'../input_files/{video_file}')
              .get(cv2.CAP_PROP_FRAME_COUNT)) + 1

    base_path = '../intermediate_output'
    video_name = video_file.split('.')[0]
    detection_data = io_utils.read_detection_csv(f'{base_path}/{video_name}'
                                                 + '_detections.csv')
    face_df = pd.read_csv(f'{base_path}/{video_name}_faces.csv')
    embedding_path = f'{base_path}/{video_file.split(".")[0]}_embeddings.hdf5'

    # prior_trks = _trk_continuation(video_file, stride=stride)
    active_trks = {}
    trk_cache = {}
    trk_id = 0

    for f_num in range(start, end):
        if active_trks:
            _predict_or_cache(active_trks, trk_cache, f_num)

        try:
            detections = detection_data[f_num]
            embeddings = io_utils.read_embeddings(embedding_path, f_num)
            measurements = (detections, embeddings)
        except KeyError:
            measurements = None

        if measurements:
            if active_trks:
                match_output = _match_and_update([d_bxs, embeddings],
                                                 active_trks, f_num)
                active_trks, unmatched, embeddings = match_output
            else:
                unmatched = d_bxs

            active_trks, trk_id = _init_trks(active_trks, trk_id, unmatched,
                                            embeddings, f_num)
        


    if len(active_trks.keys()) > 0:
        io_utils.save_track_continuations(video_file, end - 1, active_trks)

    all_trks = {**active_trks, **trk_cache}
    del active_trks
    del trk_cache
    span = [start, end - 1]

    return all_trks, span
