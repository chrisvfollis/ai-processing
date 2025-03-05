import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import os
import numpy as np
from itertools import permutations
import time
import pickle
import math
import torch
import torch.nn.functional as F
import gc
from collections import deque

from utilities import io_utils
from utilities import utilities as utils


class TrackingPipeline:
    def __init__(self, video_file, time_prefix, detection_data, keypoint_data,
                 face_data, device, continuity=True, conf_thresh=0.65):
        self.video_file = video_file
        self.cam = video_file.split('.')[0].split('_')[-1]
        self.f_num = 0

        cap = cv2.VideoCapture(os.path.join('../files/input/', video_file))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.resolution = [cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                           cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
        
        self.start_time = utils.frame_timestamp(time_prefix)
        self.end_time = utils.frame_timestamp(time_prefix, self.total_frames,
                                              self.fps)
        cap.release()

        self.frame_diag = math.dist([0, 0], self.resolution)

        self.device = device

        self.all_trks = {}
        self.active_trks = {}
        self.trk_cache = {}
        self.filtered_trks = {}

        self.kp_filtered = 0
        self.lifespan_filtered = 0
        self.size_filtered = 0

        self.trk_id = 0

        self.min_lifespan = self.fps * 15
        self.max_absence = self.fps * 3

        self.detection_data = detection_data
        self.keypoint_data = keypoint_data
        self.face_data = face_data
        self.embedding_path = os.path.join(
            "../files/output/",
            f"{os.path.splitext(video_file)[0]}_embeddings.hdf5"
        )

        self.unmatched = []
        self.cost_method_data = []

        self.dt = 1/self.fps
        self.variance_scaling_factor = (self.resolution[0] / 1920) ** 2
        self.timestep_scaling_factor = (0.5/self.dt)**4     # Params were originally tuned
                                                            # with an arbitrary dt of 0.5

        self.initial_uncertainty = [5] * 8
        self.m_noise = [250 * self.variance_scaling_factor] * 4
        self.p_noise = [
            (50 * self.timestep_scaling_factor) * self.variance_scaling_factor
        ] * 4

        # self.m_noise = [500] * 4
        
        self.conf_thresh = conf_thresh

        self.primary_run_time = 0
        self.persist_time = 0

        self.creation_time = 0
        self.prediction_time = 0
        self.measurement_matching_time = 0
        self.identity_matching_time = 0
        self.spatial_analysis_time = 0
        self.feature_analysis_time = 0
        
        self.tensor_conversion_time = 0
        self.pkl_io_time = 0
        self.embedding_read_time = 0

        self.continuity = continuity
        self.prior_pkl = ''
    
        if continuity:
            self.persist_prior_tracks()

    def __getstate__(self):
        'Prepare object state for pickling.'
        state = self.__dict__.copy()
        state["device"] = str(self.device)
        return state

    def __setstate__(self, state):
        'Restore object state after unpickling'
        self.__dict__.update(state)
        self.device = torch.device(state['device'])

    def persist_prior_tracks(self):
        start_persist = time.perf_counter()

        output_dir = '../files/output'
        files = [f for f in os.listdir(output_dir)
                 if f.endswith(str(self.video_file.split('.')[0].split('_')[-1])
                               + '.pkl')]
        if not files:
            return
        
        self.prior_pkl = sorted(files)[-1]
        pkl_path = os.path.join(output_dir, self.prior_pkl)

        start_pkl_load = time.perf_counter()
        with open(pkl_path, 'rb') as f:
            prior_pipeline = pickle.load(f)
        
        end_pkl_load = time.perf_counter()
        self.pkl_io_time += (end_pkl_load - start_pkl_load)
        
        interim = (self.start_time - prior_pipeline.end_time).total_seconds()
        interim_frames = int(round(interim * self.fps, 0))
        prior_pipeline.total_frames += interim_frames

        prior_pipeline.trk_cache = {}
        reset_ids = {}
        for trk in prior_pipeline.active_trks.values():
            reset_ids[self.trk_id] = trk
            self.trk_id += 1
        prior_pipeline.active_trks = reset_ids

        prior_pipeline.run(is_continuation=True)

        reset_active = {}
        for trk_id, trk in prior_pipeline.active_trks.items():
            trk.span[0] = -1 * (prior_pipeline.total_frames - trk.span[0])
            trk.span[1] = -1 * (prior_pipeline.total_frames - trk.span[1])
            
            reset_active[trk_id] = trk
        
        reset_cached = {}
        for trk_id, trk in prior_pipeline.trk_cache.items():
            trk.span[0] = -1 * (prior_pipeline.total_frames - trk.span[0])
            trk.span[1] = -1 * (prior_pipeline.total_frames - trk.span[1])
            
            reset_cached[trk_id] = trk

        self.active_trks = reset_active
        self.trk_cache = reset_cached

        end_persist = time.perf_counter()
        self.persist_time += (end_persist - start_persist)

    def run(self, is_continuation=False):
        def _create_new_tracks():
            start_creation = time.perf_counter()

            try:
                detections, embeddings, keypoints = self.unmatched
            except ValueError:
                return None

            for i, detection in enumerate(detections):
                box = detection[:4]
                c_x, c_y = utils.centroid(box)
                measurement = [c_x, c_y] + box[2:4]
        
                kf_args = utils.format_cv2D_kf(
                    measurement, self.m_noise, self.p_noise,
                    self.initial_uncertainty, dt=self.dt
                )

                new_track = Track(box, embeddings[i], self.f_num, *kf_args)
                if keypoints:
                    new_track.add_keypoints(keypoints[i], self.f_num)
    
                self.active_trks[self.trk_id] = new_track
                self.trk_id += 1
            
            n_total_tracks = (
                len(self.active_trks) + len(self.trk_cache) + len(self.filtered_trks)
            )
            print(f'{n_total_tracks} tracks currently in memory')

            self.unmatched = []

            end_creation = time.perf_counter()
            self.creation_time += (end_creation - start_creation)

        def _predict_or_cache():
            start_prediction = time.perf_counter()

            cached = []
            for id, trk in self.active_trks.items():
                if (self.f_num - trk.span[1]) <= self.max_absence:
                    if not utils.out_of_bounds(trk.x, img_dims=self.resolution):
                        trk.predict()
                        trk.x = utils.restrain_boxes(
                            trk.x, img_dims=self.resolution
                        )
                    trk.add_state(trk.x, self.f_num)
                else:
                    self.trk_cache[id] = trk
                    cached.append(id)
            for id in cached:
                del self.active_trks[id]
            
            end_prediction = time.perf_counter()
            self.prediction_time += (end_prediction - start_prediction)

        def _match_and_update(new_measurements):
            def _construct_cost_matrix(new_detections, new_embeddings):
                active_tracks = sorted(self.active_trks.keys())
                cost_list = []

                start_convert = time.perf_counter()
                new_detections = torch.tensor(
                    new_detections, dtype=torch.float32, device=self.device
                )

                end_convert = time.perf_counter()
                self.tensor_conversion_time += (end_convert - start_convert)

                video_info = [self.f_num, self.fps, self.frame_diag]

                for trk_id in active_tracks:
                    trk = self.active_trks[trk_id]

                    cost_vector = trk.calc_assn_costs(
                        new_detections, new_embeddings, *video_info
                    )
                    cost_list.append(cost_vector)
                
                start_convert = time.perf_counter()
                tensorized_costs = torch.stack(cost_list)
                cost_matrix = tensorized_costs.cpu().numpy()
                del tensorized_costs

                end_convert = time.perf_counter()
                self.tensor_conversion_time += (end_convert - start_convert)

                return cost_matrix

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
                                print('No feasible measurement assignments')

                return assignments_dict
            
            start_match = time.perf_counter()

            trk_ids = sorted(self.active_trks.keys())
            detections, embeddings, keypoints = new_measurements

            cost_matrix = _construct_cost_matrix(detections, embeddings)
            assignments = _assign_matches(cost_matrix)

            matched = []
            for trk_index, measurement_index in assignments.items():
                id = trk_ids[trk_index]
                trk = self.active_trks[id]

                matched.append(measurement_index)

                box = detections[measurement_index]
                x, y = utils.centroid(box)
                w, h = box[2:4]
                measurement = np.array([x, y, w, h])

                trk.update(measurement)
                trk.add_state(trk.x, self.f_num)
                trk.add_detection(box, self.f_num)
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
            self.measurement_matching_time += (end_match - start_match)

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
            
            start_associate = time.perf_counter()

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
                    box = trk.detections[self.f_num][:4]
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
            
            end_associate = time.perf_counter()
            self.identity_matching_time = (end_associate - start_associate)
        
        def _assign_identities():
            def _group_tracks(trk_ids):
                def _construct_track_graph(trk_ids):
                    track_graph = np.diag([1] * len(trk_ids)).tolist()

                    for i in range(len(trk_ids)):
                        trk = self.all_trks[trk_ids[i]]
                        for j in range(i + 1, len(trk_ids)):
                            trk2 = self.all_trks[trk_ids[j]]

                            if utils.is_coincident(trk.span, trk2.span):
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
                
                track_graph = _construct_track_graph(trk_ids)
                track_sets = _build_sets(track_graph)

                meta_graph = _construct_meta_graph(track_sets)
                meta_sets = _build_sets(meta_graph)

                groups = _isolate_groups(meta_sets)

                return groups, meta_sets, track_sets

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

            def _permute_constraint_cascades(groups, meta_sets):
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
            no_id_c = 0
            id_c = 0
            trk_id_costs = {}
            for trk_id, trk in self.all_trks.items():
                id_costs = trk.calc_id_costs()
                if not id_costs:
                    no_id_c += 1
                    continue
                id_c += 1
                trk_id_costs[trk_id] = id_costs
            
            trk_ids = sorted(trk_id_costs.keys())
            groups, meta_sets, track_sets = _group_tracks(trk_ids)

            results = _build_cost_matrices(trk_id_costs, track_sets)
            trk_set_cost_matrices, track_mappings, identity_mappings = results

            unique_cascades = _permute_constraint_cascades(groups, meta_sets)

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
                                        print('No feasible identity assignments')
        
                    if cost < min_cost:
                        min_cost = cost
                        optimal_assignments = assigned
                
                for trk_id, identity in optimal_assignments.items():
                    self.all_trks[trk_id].identity = identity
                
                all_optimal_assignments.update(optimal_assignments)
            
            end_assign = time.perf_counter()
            self.identity_matching_time = (end_assign - start_assign)

            return all_optimal_assignments

        if not is_continuation:
            print(f"Running tracking pipeline for {self.video_file}...")
        start_run = time.perf_counter()

        while self.f_num < self.total_frames:
            if self.active_trks:
                _predict_or_cache()

            try:
                start_convert = time.perf_counter()
                detection_tensor = torch.tensor(self.detection_data[self.f_num])

                end_convert = time.perf_counter()
                self.tensor_conversion_time += (end_convert - start_convert)

                conf_mask = detection_tensor[:, 4] > self.conf_thresh

                start_read = time.perf_counter()
                embeddings = io_utils.read_embeddings(
                    self.embedding_path, self.f_num, self.device
                )

                end_read = time.perf_counter()
                self.embedding_read_time += (end_read - start_read)

                start_convert = time.perf_counter()
                detections = detection_tensor[conf_mask].tolist()
                del detection_tensor

                end_convert = time.perf_counter()
                self.tensor_conversion_time += (end_convert - start_convert)

                if self.f_num % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                embeddings = embeddings[conf_mask]
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

        if (not is_continuation) and (self.continuity == True):
            self.all_trks = self.trk_cache
            for trk in self.all_trks.values():
                self.cost_method_data.extend(trk.cost_method_data)
            _assign_identities()
            self.handle_results()
    
        elif (not is_continuation) and (self.continuity == False):
            self.all_trks = {**self.active_trks, **self.trk_cache}
            _assign_identities()
        
        end_run = time.perf_counter()
        self.primary_run_time += (end_run - start_run)
    
    def handle_results(self):
        def _finalize_and_filter():
            def _filter_by_lifespan():
                for id, trk in self.all_trks.items():
                    lifespan = trk.span[1] - trk.span[0]

                    if (
                        (lifespan < self.min_lifespan) and
                        (not trk.face_detections)
                    ):
                        self.filtered_trks[id] = trk
                        self.lifespan_filtered += 1
                
                for id in self.filtered_trks.keys():
                    try:
                        del self.all_trks[id]
                    except KeyError:
                        continue
        
            def _filter_by_keypoints(expected_kps=3, expected_conf=.55):
                expected_avg = (expected_kps * expected_conf) / 17

                for id, trk in self.all_trks.items():
                    if trk.face_detections:
                        continue
                    n_frames = len(trk.keypoints.keys())
                    if n_frames == 0:
                        trk.kp_avg = 0
                        self.filtered_trks[id] = trk
                        continue

                    total_conf = sum(trk.keypoints[f][:, 2].sum()
                                    for f in trk.keypoints.keys())
                    trk.kp_avg = total_conf / (n_frames * 17)

                    if trk.kp_avg < expected_avg:
                        self.filtered_trks[id] = trk
                        self.kp_filtered += 1

                for id in self.filtered_trks.keys():
                    try:
                        del self.all_trks[id]
                    except KeyError:
                        continue

            def _filter_by_size():
                expected_avg = (self.resolution[0] / 24) * (self.resolution[1] / 12)
                for id, trk in self.all_trks.items():
                    if trk.identity:
                        continue
                    box_sizes = [math.prod(detection[2:4]) for detection in trk.detections.values()]
                    avg = sum(box_sizes) / len(box_sizes)

                    if avg < expected_avg:
                        self.filtered_trks[id] = trk
                        self.size_filtered += 1
                
                for id in self.filtered_trks.keys():
                    try:
                        del self.all_trks[id]
                    except KeyError:
                        continue

            _filter_by_lifespan()
            _filter_by_keypoints()
            _filter_by_size()
        
        def _get_track_images(tracks, vid_dir='../files/input/'):
            vid_path = os.path.join(vid_dir, self.video_file)
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                return None

            for trk in tracks.values():
                images = []

                percentile = 75
                clear_frames = None
                while (not clear_frames) and (percentile >= 25):
                    clear_frames = trk.get_high_keypoint_frames(percentile=percentile)
                    percentile -= 10

                if clear_frames:
                    frames = [clear_frames[0], clear_frames[-1]]
                else:
                    frames = trk.span

                for f in frames:
                    x, y, w, h = map(int, trk.detections[f][:4])
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
        
        if self.continuity:
            self.save_pipeline_state()

        _get_track_images(self.all_trks)

        self.save_runtime_data()
        _finalize_and_filter()

    def save_runtime_data(self, output_dir='../files/output/runtime_data'):
        commit_hash, commit_datetime = utils.get_git_commit_info()
        clip_identifier = self.video_file.split('.')[0] + '_' + commit_hash
        os.makedirs(output_dir, exist_ok=True)

        all_tracks = {**self.active_trks, **self.trk_cache, **self.filtered_trks}

        config_data = {
            'module': [
                *['software'] * 2,
                *['video'] * 2,
                *['kalman_filter'] * 4
            ],
            'parameter': [
                'git_commit_hash',          # Software
                'git_commit_datetime',

                'resolution',               # Video
                'fps',

                'measurement_noise',        # Kalman Filter
                'process_noise',
                'time_step',
                'initial_uncertainty'
            ],
            'value': [
                commit_hash,
                commit_datetime,

                f'{self.resolution[0]}x{self.resolution[1]}',
                f'{self.fps}',

                self.m_noise,
                self.p_noise,
                self.dt,
                self.initial_uncertainty
            ]
        }
        config_df = pd.DataFrame(config_data)

        for trk in all_tracks.values():
            self.spatial_analysis_time += trk.sp_analysis_time
            self.feature_analysis_time += trk.ft_analysis_time
            self.tensor_conversion_time += trk.tensor_conversion_time

        performance_data = {
            'module': [
                *['pipeline'] * 2,
                *['track_objects'] * 6,
                *['data_management'] * 3
            ],
            'metric': [
                'primary_run_time',                 # Pipeline
                'persist_time',

                'creation_time',                    # Track Objects
                'prediction_time',
                'measurement_matching_time',
                'identity_matching_time',
                'spatial_analysis_time',
                'feature_analysis_time',

                'tensor_conversion_time',           # Data Management
                'pkl_io_time',
                'embedding_read_time'
            ],
            'value': [
                self.primary_run_time,
                self.persist_time,

                self.creation_time,
                self.prediction_time,
                self.measurement_matching_time,
                self.identity_matching_time,
                self.spatial_analysis_time,
                self.feature_analysis_time,
                
                self.tensor_conversion_time,
                self.pkl_io_time,
                self.embedding_read_time
            ],
        }
        performance_df = pd.DataFrame(performance_data)
        
        stats_data = {
            'stat_title': [
                'n_total_tracks',
                'n_keypoint_filtered',
                'n_lifespan_filtered',
                'n_size_filtered'
            ],
            'stat_value': [
                len(all_tracks),
                self.kp_filtered,
                self.lifespan_filtered,
                self.size_filtered
            ]
        }
        stats_df = pd.DataFrame(stats_data)

        track_data = []
        for trk_id, trk in all_tracks.items():
            for frame, detection in trk.detections.items():
                detection_conf = detection[-1] if len(detection) == 5 else None
                face_detections = trk.face_detections.get(frame, None)
                cosine_distance = (
                    face_detections['distance'].min() if face_detections is not None else None
                )
                keypoints = trk.keypoints.get(frame, None)
                keypoint_conf = keypoints[:, 2].sum() if keypoints is not None else None
                num_keypoints = keypoints.shape[0] if keypoints is not None else 0

                track_data.append({
                    'track_id': trk_id,
                    'frame': frame,
                    'detection_confidence': detection_conf,
                    'facial_cos_dist': cosine_distance,
                    'keypoint_confidence': keypoint_conf,
                    'num_keypoints': num_keypoints,
                })
        track_df = pd.DataFrame(track_data)

        cost_method_df = pd.DataFrame(self.cost_method_data)

        filename = io_utils.get_unique_filename(output_dir, f'tracking_data_{clip_identifier}.xlsx')
        excel_path = os.path.join(output_dir, filename)

        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                track_df.to_excel(writer, sheet_name='Tracking Data', index=False)
                stats_df.to_excel(writer, sheet_name='Stats', index=False)
                cost_method_df.to_excel(writer, sheet_name='Association Data', index=False)
                config_df.to_excel(writer, sheet_name='Configuration', index=False)
                performance_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                print(f'Saved tracking runtime data to {excel_path}')
        except Exception as e:
            print(f"Failed to save Excel file: {e}")

    def save_pipeline_state(self, output_dir='../files/output'):
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = self.video_file.split('.')[0]

        print(f'{len(self.active_trks.keys())} tracks saved to be continued')

        save_path = os.path.join(output_dir, f'{file_prefix}.pkl')

        start_pkl_mgmt = time.perf_counter()
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        
        if self.prior_pkl:
            prior_path = os.path.join(output_dir, self.prior_pkl)
            if os.path.exists(prior_path) and os.path.isfile(prior_path):
                os.remove(prior_path)
        
        end_pkl_mgmt = time.perf_counter()
        self.pkl_io_time += (end_pkl_mgmt - start_pkl_mgmt)

        print('Tracking pipeline saved')

    def generate_output_vid(self, input_dir='../files/input/', output_dir='../files/output/videos'):
        print(f'Generating output video for {self.video_file}')

        all_trks = {**self.active_trks, **self.trk_cache, **self.filtered_trks}

        cap = cv2.VideoCapture(os.path.join(input_dir, self.video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)

        prefix = self.video_file.split('.')[0]
        filename = io_utils.get_unique_filename(output_dir, f'{prefix}_boxes.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_dir, filename),
                              fourcc, fps, (1920, 1080))
        
        color = (245, 104, 17)

        f_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for trk_id, trk in all_trks.items():
                box = trk.states.get(f_num, None)
                
                if (box is None) or (box.size == 0):
                    continue

                cx, cy, w, h = map(int, box[:4])
                x1, y1 = int(cx - (w / 2)), int(cy - (h / 2))
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'trk_{trk_id}', (x1 + 5, y1 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if trk.identity:
                    cv2.putText(frame, f'{trk.identity}', (x2 - 5, y2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                det_box = trk.detections.get(f_num, None)
                if det_box is not None:
                    x, y, w, h = det_box[:4]
                    x1, y1 = int(x), int(y)
                    x2, y2 = int(x1 + w), int(y1 + h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            frame = cv2.resize(frame, (1920, 1080))
            out.write(frame)
            f_num += 1

        out.release()
        cap.release()


# ----------------------------------------------------------------------------


# Individual Track Components and Definition:


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
        self.embedding_cache = deque(maxlen=20)
        self.span = [args[0], args[0]]

        self.coincident_trks = []
        self.identity = None

        self.cost_method_data = []

        self.sp_analysis_time = 0
        self.ft_analysis_time = 0
        self.tensor_conversion_time = 0

        self.add_embedding(embedding)
    
    def __getstate__(self):
        'Prepare object state for pickling by moving tensors off of the GPU'
        state = self.__dict__.copy()

        state['embedding_cache'] = [emb.to('cpu') for emb in state['embedding_cache']]
        state['embedding_cache_tensor'] = state['embedding_cache_tensor'].to('cpu')
        return state

    def __setstate__(self, state):
        '''
        Restore object state after unpickling by moving tensors onto the GPU
        if one is available.
        '''
        self.__dict__.update(state)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.embedding_cache = deque([emb.to(device) for emb in state['embedding_cache']], maxlen=20)
        self.embedding_cache_tensor = self.embedding_cache_tensor.to(device)

    def add_embedding(self, embedding):
        self.embedding_cache.append(embedding)

        if hasattr(self, 'embedding_cache_tensor'):
            del self.embedding_cache_tensor
        
        start_convert = time.perf_counter()
        self.embedding_cache_tensor = torch.stack(list(self.embedding_cache))

        end_convert = time.perf_counter()
        self.tensor_conversion_time += (end_convert - start_convert)

    def add_detection(self, new_detection, frame_number):
        self.detections[frame_number] = new_detection
        self.span[1] = frame_number
    
    def add_keypoints(self, new_keypoints, frame_number):
        self.keypoints[frame_number] = new_keypoints

    def add_face_detection(self, possible_matches, frame_number):
        self.face_detections[frame_number] = possible_matches
    
    def calc_cos_distances(self, new_embeddings, normalize=True):
        cached_embeddings = self.embedding_cache_tensor
        cos_sims = F.cosine_similarity(
            cached_embeddings.unsqueeze(1),
            new_embeddings.unsqueeze(0),
            dim=2
        )
        cos_distances = 1 - cos_sims

        return (cos_distances / 2) if normalize else cos_distances
    
    def calc_assn_costs(self, new_detections, new_embeddings, f_num, fps,
                        frame_diag, max_scale_ratio=1.5, max_pixel_delta=100):
        '''
        Returns the association costs between the track and each of the new
        measurements. Depending on certain factors, different methods are
        used to compute these costs.
        '''

        def _spatial_analysis(new_detections, frame_diag, distance_cutoff=0.5):
            def _normalized_euclidean(new_detections, frame_diag,
                                      distance_cutoff=0.5):
                device = new_detections.device

                if len(new_detections) == 0:
                    start_convert = time.perf_counter()
                    no_detections = torch.tensor([], dtype=torch.float32, device=device)

                    end_convert = time.perf_counter()
                    self.tensor_conversion_time += (end_convert - start_convert)
                    return no_detections

                start_convert = time.perf_counter()
                trk_centroid = torch.tensor(
                    self.x[:2], dtype=torch.float32, device=device
                ).unsqueeze(0)

                det_centroids = torch.stack(
                    [torch.tensor(utils.centroid(detection), device=device)
                     for detection in new_detections]
                )

                end_convert = time.perf_counter()
                self.tensor_conversion_time += (end_convert - start_convert)

                distances = torch.norm(det_centroids - trk_centroid, dim=1)
                normalized = distances / frame_diag

                normalized = torch.where(normalized >= distance_cutoff,
                                         torch.tensor(float('inf'),
                                                      device=device), 
                                         normalized)

                return normalized
            
            def _normalized_area():
                pass
            start_sp_analysis = time.perf_counter()

            euclidean_dists = _normalized_euclidean(
                new_detections, frame_diag, distance_cutoff=distance_cutoff
            )

            start_convert = time.perf_counter()
            self.cost_method_data[-1]['spatial_costs'] = euclidean_dists.tolist()

            end_convert = time.perf_counter()
            self.tensor_conversion_time += (end_convert - start_convert)

            end_sp_analysis = time.perf_counter()
            self.sp_analysis_time = (end_sp_analysis - start_sp_analysis)

            return euclidean_dists

        def _feature_analysis(new_embeddings, methods):
            def _weighted_moving_avg(cos_distances, mask,
                                     decay_factor=0.9):
                '''
                Best suited for:
                - Ordinary association conditions with a small number
                    of time steps between the current frame and most
                    recent measurement
                - Mild to moderate changes in lighting, bounding box
                    dimensions, etc between detections.
                '''
                indices = mask.nonzero(as_tuple=True)[0]
                masked_distances = cos_distances[:, indices]

                num_cached = cos_distances.shape[0]

                start_convert = time.perf_counter()
                weights = torch.tensor(
                    [decay_factor ** (num_cached - i - 1)
                     for i in range(num_cached)],
                    device=cos_distances.device,
                    dtype=cos_distances.dtype
                ).unsqueeze(1) # shape: (num_cached, 1)

                end_convert = time.perf_counter()
                self.tensor_conversion_time += (end_convert - start_convert)

                weighted_distances = (
                    (masked_distances * weights).sum(dim=0) / weights.sum()
                )
                
                return weighted_distances

            def _lowest_in_cache(cos_distances, mask):
                ''' 
                Best suited for:
                - Inactive track reassociation.
                - When a large number of time steps have elapsed since the last
                  associated detection.
                - Extreme changes in lighting, bounding box dimensions, etc
                  between detections.
                '''

                indices = mask.nonzero(as_tuple=True)[0]
                masked_distances = cos_distances[:, indices]

                return masked_distances.min(dim=0).values

            def _median_in_cache(cos_distances, mask):
                ''' 
                Best suited for:
                - Cases where a balance between extreme values is needed.
                - Reducing sensitivity to outliers in embedding comparisons.
                - When neither the lowest nor weighted moving average distance is ideal.
                '''

                indices = mask.nonzero(as_tuple=True)[0]
                masked_distances = cos_distances[:, indices]

                return masked_distances.median(dim=0).values
            
            start_ft_analysis = time.perf_counter()

            num_cached = len(self.embedding_cache)
            num_new = new_embeddings.shape[0]
            if num_cached == 0:
                start_convert = time.perf_counter()
                num_new_tensor = torch.full((num_new,), float('inf'))

                end_convert = time.perf_counter()
                self.tensor_conversion_time += (end_convert - start_convert)
                return num_new_tensor

            cos_distances = self.calc_cos_distances(new_embeddings)

            method_map = {'standard': 0, 'lowest': 1, 'median': 2}

            start_convert = time.perf_counter()
            methods = torch.tensor(
                [method_map[m] for m in methods], device=cos_distances.device
            )
            standard_mask = methods == 0
            lowest_mask = methods == 1
            median_mask = methods == 2

            costs = torch.full(
                (num_new,), float('inf'), device=cos_distances.device
            )

            end_convert = time.perf_counter()
            self.tensor_conversion_time += (end_convert - start_convert)

            if standard_mask.any():
                costs[standard_mask] = _weighted_moving_avg(
                    cos_distances, standard_mask
                )
            if lowest_mask.any():
                costs[lowest_mask] = _lowest_in_cache(cos_distances, lowest_mask)
            if median_mask.any():
                costs[median_mask] = _median_in_cache(cos_distances, median_mask)
            
            start_convert = time.perf_counter()
            self.cost_method_data[-1]['dissimilarity_costs'] = costs.tolist()
            self.cost_method_data[-1]['cost_methods'] = methods.tolist()

            end_convert = time.perf_counter()
            self.tensor_conversion_time += (end_convert - start_convert)

            end_ft_analysis = time.perf_counter()
            self.ft_analysis_time += (end_ft_analysis - start_ft_analysis)

            return costs

        self.cost_method_data.append({
            'frame': f_num,
            'track_id': id(self),
            'detection_count': len(new_detections),
            'spatial_costs': [],
            'dissimilarity_costs': [],
            'cost_methods': []
        })

        spatial_costs = _spatial_analysis(new_detections, frame_diag,
                                          distance_cutoff=0.5)
        
        cost_methods = []
        last_detected = self.span[1]
        if (f_num - last_detected) <= (fps * 1.5):
            weights = [1, 0.2]
    
            expected_area = math.prod(self.x[2:4])
            expected_pixel_stat = 0

            for new_det in new_detections:
                measured_area = math.prod(new_det[2:4])
                measured_pixel_stat = 0
                area_vals = [expected_area, measured_area]
                color_vals = [expected_pixel_stat, measured_pixel_stat]

                if (
                    (min(area_vals) > 0) and
                    math.sqrt(max(area_vals) / min(area_vals)) >
                    max_scale_ratio
                ):
                    cost_methods.append('lowest')
                elif (max(color_vals) - min(color_vals)) > max_pixel_delta:
                    cost_methods.append('lowest')
                else:
                    cost_methods.append('standard')     
        else:
            weights = [1, 0.5]
            cost_methods = ['lowest'] * len(new_embeddings)
        
        dissimilarity_costs = _feature_analysis(new_embeddings, cost_methods)

        weighted_costs = (
            (spatial_costs * weights[0]) + (dissimilarity_costs * weights[1])
        )

        return weighted_costs

    def calc_id_costs(self):
        costs = {}

        all_dfs = list(self.face_detections.values())
        if not all_dfs:
            return {}

        merged_df = pd.concat(all_dfs, ignore_index=True, copy=False)
        grouped = merged_df.groupby('identity')

        for identity, group in grouped:
            distances = group['distance']
            frequency = len(group)
            
            avg_distance = sum(distances)/frequency
    
            freq_weighting = (
                np.log10(1 + np.exp(frequency)) * (1.025**frequency)
            )
            
            cost = avg_distance / freq_weighting
            costs[identity] = cost
        
        return costs
    
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
