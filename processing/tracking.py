import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import os
import io_utils
from datetime import datetime
import numpy as np
import utilities as utils
import json
from itertools import permutations
import time


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
        self.x = utils.restrain_boxes(self.x)

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

        self.face_detections = {}
        self.detections = {args[0]: detection}
        self.keypoints = {}
        self.embeddings = [embedding]

        self.first_detection_frame = args[0]
        self.last_detection_frame = args[0]

        self.coincident_trks = []
        self.identity = None

    def add_embedding(self, embedding, window=-20):
        self.embeddings.append(embedding)
        self.embeddings = self.embeddings[window:]

    def add_detection(self, new_detection, frame_number):
        self.detections[frame_number] = new_detection
        self.last_detection_frame = frame_number
    
    def add_keypoints(self, new_keypoints, frame_number):
        self.keypoints[frame_number] = new_keypoints

    def add_face_detection(self, possible_matches, frame_number):
        self.face_detections[frame_number] = possible_matches

    def calc_id_match_costs(self):
        costs = {}

        all_dfs = list(self.face_detections.values())
        if not all_dfs:
            return {}

        merged_df = pd.concat(all_dfs, ignore_index=True)
        grouped = merged_df.groupby('identity')

        for identity, group in grouped:
            distances = group['distance']
            frequency = len(group)
            
            avg_distance = sum(distances)/frequency
    
            frequency_adjustment = (
                np.log10(1 + np.exp(frequency)) * (1.025**frequency)
            )
            
            cost = avg_distance / frequency_adjustment
            costs[identity] = cost
        
        return costs
    
    def find_best_id(self):
        id_costs = self.calc_id_match_costs()

        if not id_costs:
            self.identity = None
            self.id_cost = None
            return self.identity

        identities, costs = zip(*id_costs.items())
        min_idx = costs.index(min(costs))

        self.identity = identities[min_idx]
        self.id_cost = costs[min_idx]
        
        return self.identity

    def best_single_id_frame(self, target_id=None):
        min_distance = float('inf')
        identity = None
        frame = None

        if not target_id:
            for f_num, df in self.face_detections.items():
                if not df.empty:
                    min_row = df.loc[df['distance'].idxmin()]

                    if min_row['distance'] < min_distance:
                        min_distance = min_row['distance']
                        frame = f_num
                        identity = min_row['identity']
        else:
            identity = target_id
            for f_num, df in self.face_detections.items():
                df = df.loc[df['identity'] == target_id]
                if not df.empty:
                    target_row = df.loc[df['distance'].idxmin()]

                    if target_row['distance'] < min_distance:
                        min_distance = target_row['distance']
                        frame = f_num

        return (identity, min_distance, frame)

    def get_high_keypoint_frames(self, percentile=50):
        if not self.keypoints:
            return None

        frame_confidences = {
            frame: self.keypoints[frame][:, 2].sum()
            for frame in self.keypoints.keys()
        }

        confidences = np.array(list(frame_confidences.values()))
        median_confidence = np.percentile(confidences, percentile)

        qualifying_frames = [
            frame for frame, total_conf in frame_confidences.items()
            if total_conf >= median_confidence
        ]

        if not qualifying_frames:
            return None

        return qualifying_frames


class TrackingPipeline:
    def __init__(self, video_file, detection_data, keypoint_data, face_data):
        self.video_file = video_file
        self.f_num = 0

        cap = cv2.VideoCapture(os.path.join('../input_files/', video_file))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.resolution = [cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                           cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
        cap.release()

        self.all_trks = {}
        self.active_trks = {}
        self.trk_cache = {}

        self.lifespan_filtered = {}
        self.keypoint_filtered = {}
    
        self.trk_id = 0
        self.min_lifespan = self.fps * 15

        self.max_absence = self.fps * 3

        self.detection_data = detection_data
        self.keypoint_data = keypoint_data
        self.face_data = face_data
        self.embedding_path = os.path.join(
            "../intermediate_output/",
            f"{os.path.splitext(video_file)[0]}_embeddings.hdf5"
        )

        self.unmatched = []

        self.variance_scaling_factor = (self.resolution[0] / 1920) ** 2
        self.initial_uncertainty = [5 * self.variance_scaling_factor] * 8
        self.m_noise = [500 * self.variance_scaling_factor] * 4
        self.p_noise = [50 * self.variance_scaling_factor] * 4
        self.dt = 1/self.fps

        self.matching_time = 0
        self.prediction_time = 0
        self.id_assign_time = 0

    def load_prior_tracks(self, threshold=90):
        continuations = io_utils.load_track_continuations(self.video_file)
        if (not continuations) or (len(continuations) == 0):
            return None
        
        prev_end_time = datetime.strptime(continuations[0][4], "%Y-%m-%d %H:%M:%S.%f")

        time_prefix = utils.parse_clip_filename(self.video_file, data='time')
        clip_start_time = utils.frame_timestamp(time_prefix)

        interim = round((clip_start_time - prev_end_time)
                        .total_seconds(), 0) * self.fps

        for row in continuations:
            last_detection_delta = row[11]
            if (-1 * (last_detection_delta - interim)) >= threshold:
                continue

            trk_id = 'a' + str(row[1])
    
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

            self.active_trks[trk_id] = trk
        
        if self.active_trks:
            self.f_num = 0 - interim

    def run(self):
        def _create_new_tracks():
            try:
                detections, embeddings, keypoints = self.unmatched
            except ValueError:
                return None

            for i, detection in enumerate(detections):
                box = detection[:4]
                c_x, c_y = utils.centroid(box)
                measurement = [c_x, c_y] + box[2:]
        
                kf_args = utils.format_cv2D_kf(
                    measurement, self.m_noise, self.p_noise,
                    self.initial_uncertainty, dt=self.dt
                )

                new_track = Track(box, embeddings[i], self.f_num, *kf_args)
                if keypoints:
                    new_track.add_keypoints(keypoints[i], self.f_num)
    
                self.active_trks[self.trk_id] = new_track
                self.trk_id += 1

            self.unmatched = []

        def _predict_or_cache():
            start_prediction = time.perf_counter()

            cached = []
            for id, trk in self.active_trks.items():
                if (self.f_num - trk.last_detection_frame) <= self.max_absence:
                    trk.predict()
                    trk.add_state(trk.x, self.f_num)
                else:
                    self.trk_cache[id] = trk
                    cached.append(id)
            for id in cached:
                del self.active_trks[id]
            
            end_prediction = time.perf_counter()
            self.prediction_time += (end_prediction - start_prediction)

        def _match_and_update(measurements):
            def _construct_cost_matrix(detections, embeddings, weights=[1, 0.1]):
                '''
                Creates a cost matrix based on a weighted sum of geometric
                distances and embedding distances.
                
                Both of the distance metrics are normalized, such that all values
                for each are scaled within the range [0, 1]. This prevents
                significant scale differences from unintentionally biasing how
                each metric is weighted.
                '''

                def _distance_costs(detections, cutoff_multiplier=0.5):

                    d_cntrs = utils.get_centroids(detections)
                    t_cntrs = []

                    trk_ids = sorted(self.active_trks.keys())
                    for id in trk_ids:
                        t_cntrs.append(self.active_trks[id].x.tolist()[:2])
                    
                    frame_w = self.resolution[0]
                    frame_h = self.resolution[1]
                    max_distance_in_frame = np.sqrt(frame_w**2 + frame_h**2)

                    distance_cutoff = max_distance_in_frame * cutoff_multiplier

                    rows = len(t_cntrs)
                    cols = len(d_cntrs)
                    cost_matrix = [[float('inf')] * cols for _ in range(rows)]

                    for i, c2 in enumerate(t_cntrs):
                        for j, c1 in enumerate(d_cntrs):
                            distance = utils.euclidean_distance((c1, c2))
                            if distance < distance_cutoff:
                                normalized = distance / max_distance_in_frame
                                cost_matrix[i][j] = normalized

                    return np.array(cost_matrix)

                def _similarity_costs(embeddings):
                    def _lowest_distance(trk, embedding):
                        lowest = 1
                        for trk_embedding in trk.embeddings:
                            dst = utils.cos_distance(trk_embedding, embedding,
                                                    normalize=True)
                            if dst < lowest:
                                lowest = dst
                        return lowest

                    trk_ids = sorted(self.active_trks.keys())

                    rows = len(trk_ids)
                    cols = len(embeddings)
                    cost_matrix = [[float('inf')] * cols for _ in range(rows)]

                    for i, id in enumerate(trk_ids):
                        for j, emb in enumerate(embeddings):
                            cost = _lowest_distance(self.active_trks[id], emb)
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
                        filtered_matrix, keep = _filter_sparse_rows(filtered_matrix)
                        keep_to_orig_row_map = np.where(viable_rows)[0][keep]

                        if filtered_matrix.size > 0:
                            try:
                                row_ind, col_ind = linear_sum_assignment(filtered_matrix)

                                orig_row_ind = [keep[i] for i in row_ind]
                                for i, j in zip(row_ind, col_ind):
                                    orig_row = keep_to_orig_row_map[i]
                                    orig_col = np.where(viable_cols)[0][j]
                                    assignments_dict[orig_row] = orig_col
                            except ValueError:
                                print('No feasible assignments')

                return assignments_dict
            
            start_match = time.perf_counter()

            trk_ids = sorted(self.active_trks.keys())
            detections, embeddings, keypoints = measurements

            cost_matrix = _construct_cost_matrix(detections, embeddings,
                                                weights=[1, 0.1])
            assignments = _assign_matches(cost_matrix)

            matched = []
            for trk_index, measurement_index in assignments.items():
                id = trk_ids[trk_index]
                trk = self.active_trks[id]

                matched.append(measurement_index)

                box = detections[measurement_index]
                c_x, c_y = utils.centroid(box)
                w, h = box[2:4]
                measurement = np.array([c_x, c_y, w, h])

                trk.update(measurement)
                trk.add_detection(box[:4], self.f_num)
                trk.add_embedding(embeddings[measurement_index])
    
                if keypoints:
                    trk.add_keypoints(keypoints[measurement_index], self.f_num)

            unmatched_detections = [detections[j] for j in range(len(detections))
                                    if j not in matched]
            unmatched_embeddings = [embeddings[j] for j in range(len(embeddings))
                                    if j not in matched]
            if keypoints:
                unmatched_keypoints = [
                    keypoints[j] for j in range(len(keypoints)) if j not in matched
                ]
            else:
                unmatched_keypoints = []
            
            self.unmatched = [unmatched_detections, unmatched_embeddings,
                              unmatched_keypoints]
            
            end_match = time.perf_counter()
            self.matching_time += (end_match - start_match)

        def _associate_faces(cutoff=0.9):
            def _overlap_costs(face_boxes, person_boxes):
                rows = len(face_boxes)
                cols = len(person_boxes)
                cost_matrix = [[float('inf')] * cols for _ in range(rows)]

                for i, box1 in enumerate(face_boxes):
                    for j, box2 in enumerate(person_boxes):
                        overlap = utils.percent_overlap(box1, box2)
                        cost = 1 - overlap
                        cost_matrix[i][j] = cost

                return np.array(cost_matrix)
            
            if (self.face_data is None) or (self.face_data.empty):
                return None
    
            face_boxes = []
            person_boxes = []

            face_df = self.face_data.loc[self.face_data['f'] == self.f_num]
            face_boxes += (face_df[['x', 'y', 'w', 'h']].drop_duplicates()
                        .values.tolist())

            trk_ids = sorted(self.active_trks.keys())
            for id in trk_ids:
                trk = self.active_trks[id]
                try:
                    box = trk.detections[self.f_num]
                except KeyError:
                    state = trk.x[:4]
                    x, y = utils.centroid(state, reverse=True)
                    w, h = state[2:4]
                    box = [x, y, w, h]
                person_boxes.append(box)
            
            if (face_df.empty) or (not person_boxes):
                return False

            cost_matrix = _overlap_costs(face_boxes, person_boxes)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments = dict(zip(row_ind, col_ind))

            for f_idx, p_idx in assignments.items():
                if cost_matrix[f_idx][p_idx] >= cutoff:
                    continue
        
                id = trk_ids[p_idx]
                face_box = face_boxes[f_idx]
                f_matches = face_df.loc[
                    (face_df['x'] == face_box[0]) &
                    (face_df['y'] == face_box[1]) &
                    (face_df['w'] == face_box[2]) &
                    (face_df['h'] == face_box[3])
                ]

                self.active_trks[id].add_face_detection(f_matches, self.f_num)     
        
        def _wrap_up():
            def _finalize_and_filter():
                def _filter_by_lifespan():                
                    for id, trk in self.all_trks.items():
                        start = trk.first_detection_frame
                        end = trk.last_detection_frame
                        lifespan = end - start

                        if (
                            (lifespan < self.min_lifespan) and
                            (not trk.face_detections)
                        ):
                            self.lifespan_filtered[id] = trk
                    
                    for id in self.lifespan_filtered.keys():
                        del self.all_trks[id]
            
                def _filter_by_keypoints(expected_kps=3, expected_conf=.55):
                    expected_avg = (expected_kps * expected_conf) / 17

                    for id, trk in self.all_trks.items():
                        if trk.face_detections:
                            continue
                        n_frames = len(trk.keypoints.keys())
                        if n_frames == 0:
                            trk.kp_avg = 0
                            self.keypoint_filtered[id] = trk
                            continue
        
                        total_conf = sum(trk.keypoints[f][:, 2].sum()
                                        for f in trk.keypoints.keys())
                        trk.kp_avg = total_conf / (n_frames * 17)

                        if trk.kp_avg < expected_avg:
                            self.keypoint_filtered[id] = trk
                    
                    for id in self.keypoint_filtered.keys():
                        del self.all_trks[id]
                
                self.all_trks = {**self.active_trks, **self.trk_cache}
                del self.active_trks
                del self.trk_cache
                
                _filter_by_lifespan()
                _filter_by_keypoints()
            
            def _get_track_images(tracks, vid_dir='../input_files/'):
                vid_path = os.path.join(vid_dir, self.video_file)
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    return None

                for trk in tracks.values():
                    images = []

                    percentile = 75
                    clear_frames = None
                    while (not clear_frames) and (percentile >= 5):
                        clear_frames = trk.get_high_keypoint_frames(percentile=percentile)
                        percentile -= 10
                    
        
                    if clear_frames:
                        frames = [clear_frames[0], clear_frames[-1]]
                    else:
                        frames = [
                            trk.first_detection_frame,
                            trk.last_detection_frame,
                        ]

                    for f in frames:
                        x, y, w, h = trk.detections[f][:4]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                        ret, frame = cap.read()
                        if not ret:
                            images.append(None)
                            continue
                        cropped = frame[y:y+h, x:x+w]
                        images.append(cropped)

                    trk.start_img = io_utils.save_event_image(images[0])
                    trk.end_img = io_utils.save_event_image(images[1])

                cap.release()

            def _print_info():
                print(f'{len(self.all_trks.keys())} valid tracks retained')
                print(str(sum(1 for trk in self.all_trks.values()
                              if trk.face_detections)) + ' valid tracks identified')
                print(f'{len(self.keypoint_filtered.keys())} low kp average tracks filtered')
                print(f'{len(self.lifespan_filtered.keys())} low lifespan tracks filtered')

                _get_track_images(self.keypoint_filtered)
                for id, trk in self.keypoint_filtered.items():
                    print(f'TRACK {id} filtered: kp_avg = {trk.kp_avg:.2f}')
                    print(f'lifespan: {trk.last_detection_frame - trk.first_detection_frame}')
                    print(f'start img: {trk.start_img}')
                    print(f'end img: {trk.start_img}')
                
            _finalize_and_filter()
            _get_track_images(self.all_trks)
            # _print_info()
    

        while self.f_num < self.total_frames:
            if self.active_trks:
                _predict_or_cache()

            try:
                detections = self.detection_data[self.f_num]
                embeddings = io_utils.read_embeddings(self.embedding_path, self.f_num)
                keypoints = self.keypoint_data.get(self.f_num, None)
                measurements = [detections, embeddings, keypoints]
            except KeyError:
                measurements = None

            if measurements and self.active_trks:
                _match_and_update(measurements)
            elif measurements and (not self.active_trks):
                self.unmatched = measurements
            
            _create_new_tracks()
            _associate_faces()

            self.f_num += 1

        _wrap_up()

    def group_tracks(self, trk_ids: list = 'all'):
        def _construct_track_graph(trk_ids):
            track_graph = np.diag([1] * len(trk_ids)).tolist()

            for i in range(len(trk_ids)):
                trk = self.all_trks[trk_ids[i]]
                span = [trk.first_detection_frame, trk.last_detection_frame]

                for j in range(i + 1, len(trk_ids)):
                    trk2 = self.all_trks[trk_ids[j]]
                    span2 = [
                        trk2.first_detection_frame,
                        trk2.last_detection_frame
                    ]

                    if utils.is_coincident(span, span2):
                        track_graph[i][j] = 1
                        track_graph[j][i] = 1

            return np.array(track_graph)
        
        def _construct_meta_graph(trk_sets):

            set_graph = np.diag([1] * len(trk_sets)).tolist()

            for i in range(len(trk_sets)):
                for j in range(i + 1, len(trk_sets)):
                    if bool(set(track_sets[i]) & set(trk_sets[j])):
                        set_graph[i][j] = 1
                        set_graph[j][i] = 1

            return np.array(set_graph)

        def _build_sets(graph):
            num_tracks = len(graph)
            visited = [False] * num_tracks
            all_sets = []

            for track in range(num_tracks):
                if not visited[track]:
                    neighbor_set = []

                    for neighbor in range(num_tracks):
                        if graph[track][neighbor] == 1:
                            neighbor_set.append(neighbor)
                            visited[neighbor] = True

                    all_sets.append(neighbor_set)
                
            return all_sets
        
        def _isolate_groups(meta_sets):
            grouped = [False] * len(meta_sets)

            group_contents = []
            groups = []

            for i in range(len(meta_sets)):
                if grouped[i]:
                    continue
    
                group_contents.append(meta_sets[i])
                grouped[i] = True

                groups.append([i])

                for j in range(i + 1, len(meta_sets)):
                    if bool(set(group_contents[-1]) & set(meta_sets[j])):
                        
                        group_contents[-1] += meta_sets[j]

                        groups[-1].append(j)
                        grouped[j] = True

            return groups
        
        if trk_ids == 'all':
            trk_ids = sorted(self.all_trks.keys())

        track_graph = _construct_track_graph(trk_ids)
        track_sets = _build_sets(track_graph)

        meta_graph = _construct_meta_graph(track_sets)
        meta_sets = _build_sets(meta_graph)

        groups = _isolate_groups(meta_sets)

        return groups, meta_sets, track_sets

    def permute_constraint_cascades(self, groups, meta_sets):
        def _remove_duplicates(track_order):
            seen = set()
            return [x for x in track_order if not (x in seen or seen.add(x))]
        
        def _filter_redundant(groups, group_permutations, meta_sets):
            def _find_independent(group_members, meta_sets):
                independent = []

                for i in range(len(group_members)):
                    for j in range(i + 1, len(group_members)):
                        if not bool(set(meta_sets[i]) & set(meta_sets[j])):
                            independent.append(str([i, j]).strip('[]'))
                
                return independent
            
            filtered_group_permutations = []

            for i in range(len(group_permutations)):
                group_members = groups[i]
                group_member_permutations = group_permutations[i]
                if len(group_members) < 2:
                    filtered_group_permutations.append(group_member_permutations)
                    continue
                filtered_gm_permutations = []
                independent = _find_independent(group_members, meta_sets)
                for j in range(len(group_member_permutations)):
                    for sequence in independent:
                        if sequence not in str(group_member_permutations[j]):
                            filtered_gm_permutations.append(
                                group_member_permutations[j]
                            )
                filtered_group_permutations.append(filtered_gm_permutations)
            
            return filtered_group_permutations

        group_permutations = []

        for group in groups:
            group_permutations.append(list(set(permutations(group))))
        
        group_permutations = _filter_redundant(
            groups, group_permutations, meta_sets
        )
        
        all_orders = []

        for group in group_permutations:
            group_orders = []

            for meta_order in group:
                track_processing_permutations = [[]]

                for meta_idx in meta_order:
                    meta_set = meta_sets[meta_idx]
                    meta_set_permutations = list(permutations(meta_set))

                    track_processing_permutations = [
                        existing_order + list(new_order)
                        for existing_order in track_processing_permutations
                        for new_order in meta_set_permutations
                    ]

                unique_orders = set()
                for track_order in track_processing_permutations:
                    cleaned_track_order = tuple(_remove_duplicates(track_order))
                    unique_orders.add(cleaned_track_order)

                group_orders.extend(list(unique_orders))


            all_orders.append(group_orders)

        return all_orders
        
    def assign_identities(self):
        def _build_cost_matrices(trk_id_costs, track_sets):
            trk_ids = sorted(trk_id_costs.keys())
            idx_to_id = {idx: trk_id for idx, trk_id in enumerate(trk_ids)}

            matrices = []

            track_mappings = {}
            identity_mappings = {}

            for k, track_set in enumerate(track_sets):
                tracks = sorted([idx_to_id[track_index] for track_index in track_set])
                identities = sorted(list(set(
                    [identity for trk_id in tracks for identity in trk_id_costs[trk_id]]
                )))

                identity_mappings[k] = []
                track_mappings[k] = []

                for identity in identities:
                    identity_mappings[k].append(identity)
                
                rows = len(tracks)
                cols = len(identities)

                cost_matrix = [[float('inf')] * cols for _ in range(rows)]
                for i, trk_id in enumerate(tracks):
                    for j, identity in enumerate(identities):
                        if identity not in trk_id_costs[trk_id]:
                            continue

                        cost = trk_id_costs[trk_id][identity]
                        cost_matrix[i][j] = cost

                    track_mappings[k].append(trk_id)
        
                matrices.append(np.array(cost_matrix))
            
            return matrices, track_mappings, identity_mappings

        def _sync_prior_assignments(assigned, tracks, identities, matrix):
            for trk_idx, track in enumerate(tracks):
                if track in assigned:
                    try:
                        id_idx = identities.index(assigned[track])
                        matrix[trk_idx, :] = float('inf')
                        matrix[:, id_idx] = float('inf')
                        matrix[trk_idx, id_idx] = 0
                    except ValueError:
                        print("Non-overlapping identity")
                        continue
            return matrix

        def _filter_sparse_rows(cost_matrix):
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

        start_assign = time.perf_counter()

        trk_id_costs = {}
        for trk_id, trk in self.all_trks.items():
            id_costs = trk.calc_id_match_costs()

            if not id_costs:
                continue

            trk_id_costs[trk_id] = id_costs
        
        trk_ids = sorted(trk_id_costs.keys())
        groups, meta_sets, track_sets = self.group_tracks(trk_ids)

        results = _build_cost_matrices(trk_id_costs, track_sets)
        trk_set_cost_matrices, track_mappings, identity_mappings = results

        unique_cascades = self.permute_constraint_cascades(groups, meta_sets)

        all_optimal_assignments = {}
        for group in unique_cascades:
            min_cost = float('inf')
            optimal_assignments = {}
            for permutation in group:
                assigned = {}
                cost = 0
                ordered_matrices = [
                    trk_set_cost_matrices[k].copy() for k in permutation
                ]
                permutation_to_original = {k: i for k, i in enumerate(permutation)}

                for k, matrix in enumerate(ordered_matrices):
                    original_index = permutation_to_original[k]
                    tracks = track_mappings[original_index]
                    identities = identity_mappings[original_index]

                    matrix = _sync_prior_assignments(assigned, tracks,
                                                     identities, matrix)

                    viable_rows = ~np.isinf(matrix).all(axis=1)
                    viable_cols = ~np.isinf(matrix).all(axis=0)

                    if viable_rows.any() and viable_cols.any():
                        try:
                            filtered_matrix = matrix[np.ix_(viable_rows,
                                                            viable_cols)]

                            if filtered_matrix.size > 0:
                                row_ind, col_ind = linear_sum_assignment(
                                    filtered_matrix
                                )

                                orig_row_ind = np.where(viable_rows)[0]
                                orig_col_ind = np.where(viable_cols)[0]

                                for i, j in zip(row_ind, col_ind):
                                    orig_row = orig_row_ind[i]
                                    orig_col = orig_col_ind[j]
                                    cost += matrix[orig_row, orig_col]
                                    assigned[tracks[orig_row]] = (identities
                                                                  [orig_col])

                        except ValueError:
                            filtered_matrix, keep = _filter_sparse_rows(
                                filtered_matrix
                            )
                            keep_to_orig_row_map = (np.where(viable_rows)
                                                    [0][keep])

                            if filtered_matrix.size > 0:
                                try:
                                    row_ind, col_ind = linear_sum_assignment(
                                        filtered_matrix
                                    )
                                    orig_row_ind = [keep[i] for i in row_ind]
                                    for i, j in zip(row_ind, col_ind):
                                        orig_row = keep_to_orig_row_map[i]
                                        orig_col = np.where(viable_cols)[0][j]
                                        cost += matrix[orig_row, orig_col]
                                        assigned[tracks[orig_row]] = (identities
                                                                      [orig_col])
                                except ValueError:
                                    print('No feasible assignments')
    
                if cost < min_cost:
                    min_cost = cost
                    optimal_assignments = assigned
            
            for trk_id, identity in optimal_assignments.items():
                self.all_trks[trk_id].identity = identity
            
            all_optimal_assignments.update(optimal_assignments)
        
        end_assign = time.perf_counter()
        self.id_assign_time = (end_assign - start_assign)

        return all_optimal_assignments
    
    def get_stats(self):
        all_stats = {}
        all_stats['time'] = {}

        all_stats['time']['id_assignment'] = self.id_assign_time
        all_stats['time']['prediction'] = self.prediction_time
        all_stats['time']['trk_matching'] = self.matching_time

        return all_stats

    def get_parameters(self):
        all_parameters = {}
        all_parameters['KFilter'] = {}

        all_parameters['KFilter']['initial_uncertainty'] = self.initial_uncertainty
        all_parameters['KFilter']['measurement_noise'] = self.m_noise
        all_parameters['KFilter']['process_noise'] = self.p_noise
        all_parameters['KFilter']['time_step'] = self.dt

        return all_parameters