# standard dependencies
import os
import sys
import psutil
from psutil import NoSuchProcess, AccessDenied, ZombieProcess
import tracemalloc
import subprocess
import threading
import gc
from datetime import datetime, timedelta
import time
from typing import Union, Sequence

# 3rd-party dependencies
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon, box

# internal dependencies
pass


def observability_thread(target, args=None):
    """
    Initializes a thread for monitoring & logging some aspect of a process.

    Args:
        target (string): The aspect of the process to observe. This determines
            which function is assigned to the `target` parameter of the thread
            constructor. Options: 'elapsed_time', 'failed_workers', 'low_memory'

        args (tuple): Any arguments that the target function expects.
            - elapsed_time: (frequency=300, include_timestamp=False)
            - failed_workers: (pool, initial_pids, async_results)
            - low_memory: (threshold=1000, interval=1)
    """

    # To do: implement logger class with .stop() method to set stop events
    # rather than having to return the stop event separately

    if target == 'elapsed_time':
        start_time, stop_event = time.time(), threading.Event()
        frequency, timestamp = args if args else (300, False)
    
        time_logger = threading.Thread(
            target=log_elapsed_time,
            args=(start_time, stop_event, frequency, timestamp),
            daemon=True
        )
        return time_logger, stop_event

    elif target == 'failed_workers':
        worker_monitor = threading.Thread(
            target=log_failed_workers, args=args, daemon=True
        )
        return worker_monitor
    
    elif target == 'low_memory':
        stop_event = threading.Event()
        threshold, interval = args if args else (1000, 1)

        low_memory_monitor = threading.Thread(
            target=log_low_memory_warnings,
            args=(stop_event, threshold, interval),
            daemon=True
        )
        return low_memory_monitor, stop_event


def log_elapsed_time(start_time, stop_event, frequency, timestamp):
    while not stop_event.is_set():
        elapsed = (time.time() - start_time) / 60
        if timestamp == False:
            print(f'Elapsed time: {elapsed:.2f} minutes')
            time.sleep(frequency)
        elif timestamp == True:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f'[{current_time}] Elapsed time: {elapsed:.2f} minutes')
            time.sleep(frequency)

    total_elapsed =  (time.time() - start_time) / 60
    if timestamp == False:
        print(f'Total elapsed time: {total_elapsed:.2f} minutes')
    elif timestamp == True:
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f'[{current_time}] Total elapsed time: {total_elapsed:.2f} minutes')


def log_failed_workers(pool, initial_pids, async_result):
    '''
    Logs any potentially failed workers from a starmap_async() run of a
    multiprocessing.Pool
    '''

    while not async_result.ready():
        time.sleep(1)
        current_pids = {p.pid for p in pool._pool if p.is_alive()}
        disappeared = initial_pids - current_pids
        if disappeared:
            print(f"[WARNING] Workers {disappeared} disappeared (possible crash)")


def memory_usage(focus, n=5, threshold=None, log_filter_key=None):
    if focus == 'processes':
        def _log_largest_processes(process_list, n):
            if process_list:
                print(f'Largest processes:')
                for pid, name, mem in processes[:n]:
                    print(f'PID {pid} - {name}: {mem:.2f} MB')

        processes = []
        for p in psutil.process_iter(
            attrs=['pid', 'name', 'memory_info'], ad_value=None
        ):
            try:
                info = p.as_dict(attrs=['pid', 'name', 'memory_info'])
                if info['memory_info']:
                    processes.append(
                        (info['pid'], info['name'], info['memory_info'].rss / 1e6)
                    )
            except (NoSuchProcess, AccessDenied, ZombieProcess):
                continue

        processes.sort(key=lambda x: x[2], reverse=True)

        total_process_memory = sum([process[2] for process in processes])
        if (threshold is None) or (total_process_memory > threshold):
            _log_largest_processes(processes, n)

        return total_process_memory

    elif focus == 'objects':
        def _log_largest_objects(object_list, n, obj_category):
            if object_list:
                print(f'Largest {obj_category} objects:')
                for obj, size in object_list[:n]:
                    print(f'Size: {size} MB | Type: {type(obj)}')

            else:
                print(f'No {obj_category} objects found')

        def _safe_sizeof(object):
            '''
            Returns the size of object in megabytes to two decimal places while
            safely handling exceptions.
            '''
            try:
                raw_size = sys.getsizeof(object)
                return round((raw_size / 1e6), 2)
            except TypeError:
                return 0

        gc.collect()

        standard_objects = sorted(
            [(obj, _safe_sizeof(obj)) for obj in gc.get_objects()],
            key=lambda x: x[1],
            reverse=True
        )
        uncollectible_objects = sorted(
            [(obj, _safe_sizeof(obj)) for obj in gc.garbage],
            key=lambda x: x[1],
            reverse=True
        )

        cpu_obj_totals = [sum([size for _, size in obj_list]) for obj_list in
                          [standard_objects, uncollectible_objects]]
        gpu_obj_totals = [(torch.cuda.memory_allocated() / 1e6)]
        
        total_obj_memory = sum(cpu_obj_totals) + sum(gpu_obj_totals)

        if (
            (threshold is None) or
            (total_obj_memory > (threshold))
        ):

            print(f'Total standard object memory: {cpu_obj_totals[0]:.2f} MB')
            _log_largest_objects(standard_objects, n, 'standard')
    
            print(f'Total uncollectible object memory: {cpu_obj_totals[1]:.2f} MB')
            _log_largest_objects(uncollectible_objects, n, 'uncollectible')

            print(f"Total pytorch object memory: {gpu_obj_totals[0]:.2f} MB")

        return total_obj_memory

    elif focus == 'allocation_lines':
        snapshot = tracemalloc.take_snapshot()
        allocation_lines = snapshot.statistics('lineno')

        allocation_lines = [(
            (line_info.traceback[-1].filename), (line_info.traceback[-1].lineno),
            (line_info.size / 1e6)
            ) for line_info in allocation_lines
        ]

        total_alloc_memory = sum([x[2] for x in allocation_lines])

        if log_filter_key is not None:
            allocation_lines = [
                (file, line_num, memory) for file, line_num, memory
                in allocation_lines if log_filter_key(file)
            ]
        if (
            (threshold is None) or
            (total_alloc_memory > threshold)
        ):
            print(f'Total allocated memory: {round(total_alloc_memory, 2)} MB')
            print('Top allocation lines:')
            for line_info in allocation_lines[:n]:
                file, line_num, memory = line_info

                print(
                    f'File {file}, line {line_num},' +
                    f'allocated {memory:.2f} MB'
                )

        return total_alloc_memory


def log_low_memory_warnings(stop_event, threshold, interval):
    while not stop_event.is_set():
        try:
            memory_info = psutil.virtual_memory()
            free_mb = memory_info.available / 1e6

            if free_mb < threshold:
                free_mb = round(free_mb, 0) if free_mb >= 1 else round(free_mb, 2)
                print(f'\n[WARNING] MEMORY CRITICAL: {free_mb} MB free')

                memory_usage('processes')

                gc.collect()
                time.sleep(10)
            else:
                time.sleep(interval)
        except Exception as e:
            print(f'Error while monitoring memory: {e}')


def get_git_commit_info(cfg_dir_path='../config'):
    '''
    Retrieve the Git commit hash and date/time for the version of the codebase
    that is currently running.
    '''

    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode('utf-8').strip()

        commit_datetime = subprocess.check_output(
            ['git', 'log', '-1', '--format=%cd', '--date=iso-strict']
        ).decode('utf-8').strip()

        return commit_hash, commit_datetime

    except subprocess.CalledProcessError:
        vfile_path = os.path.join(cfg_dir_path, 'version.txt')
        try:
            with open(vfile_path, 'r') as vfile:
                return vfile.read().strip(), 'unknown'
        except Exception:
            return 'unknown', 'unknown'


def get_video_info(source, release=True):
    if isinstance(source, str):
        source = cv2.VideoCapture(source)
    
    resolution = (int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(source.get(cv2.CAP_PROP_FPS))

    if release:
        source.release()

    return resolution, fps, total_frames


def centroid(coordinates, reverse=False):
    '''
    Returns centroid from [x1, y1, w, h] where (x1, y1) are the
    coordinates of the bounding box's top left corner.
    '''
    if reverse == False:
        x = coordinates[0] + (coordinates[2] / 2)
        y = coordinates[1] + (coordinates[3] / 2)
    elif reverse == True:
        x = coordinates[0] - (coordinates[2] / 2)
        y = coordinates[1] - (coordinates[3] / 2)

    return x, y


def get_centroids(boxes):
    '''
    Takes a list of [x1, y1, w, h] bounding boxes and creates a list of
    centroids for each box.
    '''
    frame_centroids = []
    for box in boxes:
        frame_centroids.append(centroid(box))
    return frame_centroids


def out_of_bounds(coordinates, img_dims=[1920, 1080]):
    img_w, img_h = img_dims
    margin_x, margin_y = img_w / 32, img_h / 32

    c_x, c_y, w, h = coordinates[:4]

    x_min, x_max = c_x - w / 2, c_x + w / 2
    y_min, y_max = c_y - h / 2, c_y + h / 2

    result = (
        (x_max < -margin_x) or (x_min > img_w + margin_x) or
        (y_max < -margin_y) or (y_min > img_h + margin_y)
    )

    return result


def restrain_boxes(coordinates, img_dims=[1920, 1080]):
    img_w, img_h = img_dims

    # Restrain width and height to not exceed the dimensions of
    # the image:
    coordinates[2] = min(img_w, coordinates[2])
    coordinates[3] = min(img_h, coordinates[3])

    # Restrain centroids so that box can go no further than right
    # outside of the frame.
    half_w = coordinates[2] / 2
    half_h = coordinates[3] / 2

    coordinates[0] = min((img_w + half_w), coordinates[0])
    coordinates[0] = max((0 - half_w), coordinates[0])

    coordinates[1] = min((img_h + half_h), coordinates[1])
    coordinates[1] = max((0 - half_h), coordinates[1])

    return coordinates


def get_intersection(rectangle1, rectangle2, attr='area'):
    x1, y1, w1, h1 = rectangle1
    x2, y2, w2, h2 = rectangle2

    inter_ltx = max(x1, x2)
    inter_lty = max(y1, y2)
    inter_rbx = min(x1 + w1, x2 + w2)
    inter_rby = min(y1 + h1, y2 + h2)

    if (inter_ltx >= inter_rbx) or (inter_lty >= inter_rby):
        return None

    inter_w = inter_rbx - inter_ltx
    inter_h = inter_rby - inter_lty

    if attr == 'area':
        return inter_w * inter_h
    elif attr == 'rectangle':
        return [inter_ltx, inter_lty, inter_w, inter_h]


def i_over_u(rectangle1, rectangle2):
    '''
    Returns the intersection over union between two rectangles.

    --------------------------------------------------
    Rectangle format: [x, y, w, h]
    (x, y) = top left coordinate
    '''

    r1_area = rectangle1[2] * rectangle1[3]
    r2_area = rectangle2[2] * rectangle2[3]
    
    intersection = get_intersection(rectangle1, rectangle2, attr='area')
    if not intersection:
        return None

    union = (r1_area + r2_area) - intersection

    return intersection / union


def percent_overlap(rectangle1, rectangle2):
    '''
    Returns what percent of rectangle1's total area is overlapping with
    rectangle2, i.e. how much of it is 'in' rectangle2.
    '''

    r1_area = rectangle1[2] * rectangle1[3]
    intersection = get_intersection(rectangle1, rectangle2, attr='area')
    if not intersection:
        return 0

    return intersection / r1_area


def percent_in_entryway(bbox, entryway_points):
    '''
    Returns the percent of a bounding box's total area that is 'inside' of
    an entryway. The percent is represented as a decimal.

    --------------------------------------------------
    bbox format: [x, y, w, h]
    (x, y) — top left coordinate
    --------------------------------------------------
    entryway_points format: [(x1, y1), (x2, y2), ... (x_n, y_n)]
    Order — from top left, clockwise
    --------------------------------------------------               
    '''

    entryway = Polygon(entryway_points)
    
    bbox_polygon = box(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    
    intersection = entryway.intersection(bbox_polygon).area
    area = bbox[2] * bbox[3]
    
    if area == 0:
        return 0
    
    return intersection / area


def is_coincident(span1, span2):
    '''
    Checks whether a span is coincident to another at any point.
    '''
    return not (span1[1] < span2[0] or span2[1] < span1[0])


def frame_timestamp(clip_timestamp, frame=0, fps=15):
    if isinstance(clip_timestamp, str):
        clip_timestamp = datetime.strptime(clip_timestamp, '%Y-%m-%d_%H-%M-%S')

    seconds = frame / fps
    return clip_timestamp + timedelta(seconds=seconds)


def parse_clip_filename(video_file, data='all'):
    if not video_file.endswith('.mp4'):
        return video_file

    sections = video_file.rsplit('_', 1)
    time_prefix = sections[0]
    camera = sections[1].split('.')[0]

    if data == 'all':
        return time_prefix, camera
    elif data == 'time':
        return time_prefix
    elif data == 'camera':
        return camera


def flag_entryway_events(all_trks, entryways, threshold=.4):
    for id, trk in all_trks.items():
        cam = id.split('_')[0]
        start, end = trk['trk_span']
        detections = [trk['detections'][start][:4],
                    trk['detections'][end][:4]]
        keys = ['entry', 'exit']
        for i in range(2):
            trk[keys[i]] = None
            for points in entryways[cam].values():
                pcnt_in_entryway = percent_in_entryway(detections[i], points)
                if pcnt_in_entryway > threshold:
                    trk[keys[i]] = trk['trk_span'][i]
                    break

    return all_trks


def format_cv2D_kf(measurement, m_noise, p_noise, initial_uncertainty,
                   xy_vel=[0, 0], wh_vel=[0, 0], dt=1.0):
    
    '''
    Formats the necessary matrices for modeling constant velocity in 2D space
    with a Kalman filter.

    --------------------------------------------------------

    ARGUMENTS:

    - measurement —
    The bounding box of the object, formatted as a list with the values
    [center x, center y, width, height].

    - m_noise -
    These values go along the diagonal of the measurement noise covariance
    matrix, R. Higher magnitudes = greater measurement noise, meaning more
    weight is given to predictions relative to the incoming measurements. The
    values represent the expected variance (squared error in pixels) of
    incoming measurements.

    - p_noise —
    These values are used to create the process noise covariance matrix, Q.
    Higher magnitudes = greater process noise, meaning incoming measurements
    are given more weight relative to predictions. The result is that new
    measurements have a larger impact on updating the trajectory of subsequent
    predictions. The values represent the expected variance (squared error in
    pixels) of predictions.

    - initial_uncertainty —
    These are the initial values of the estimate uncertainty matrix, P. Each
    represents the expected variance (squared error in pixels) for the
    corresponding element in the state vector.

    - xy_vel & wh_vel—
    The expected initial velocities of the object and its dimensions,
    respectively.
    
    - dt —
    The timestep.
    
    ---------------------------------------
    '''
    
    F = np.array([
        [1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dt, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dt],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])

    # Q values:
    position_var = (dt**4) / 4  # Position variance
    velocity_var = (dt**2)      # Velocity variance
    cross_var = (dt**3) / 2     # Cross-coupling term

    x_pvar, y_pvar, w_pvar, h_pvar =  np.array(p_noise) * position_var
    x_vvar, y_vvar, w_vvar, h_vvar = np.array(p_noise) * velocity_var
    x_cvar, y_cvar, w_cvar, h_cvar = np.array(p_noise) * cross_var

    Q = np.array([
        [x_pvar, 0, 0, 0, x_cvar, 0, 0, 0],
        [0, y_pvar, 0, 0, 0, y_cvar, 0, 0],
        [0, 0, w_pvar, 0, 0, 0, w_cvar, 0],
        [0, 0, 0, h_pvar, 0, 0, 0, h_cvar],
        [x_cvar, 0, 0, 0, x_vvar, 0, 0, 0],
        [0, y_cvar, 0, 0, 0, y_vvar, 0, 0],
        [0, 0, w_cvar, 0, 0, 0, w_vvar, 0],
        [0, 0, 0, h_cvar, 0, 0, 0, h_vvar]
        ])

    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
        ])

    R = np.diag(m_noise)

    x_init = np.array(measurement + xy_vel + wh_vel)
    P_init = np.diag(initial_uncertainty)

    return F, Q, H, R, x_init, P_init


def is_grayscale(frame, threshold=10):
    b, g, r = cv2.split(frame)
    
    diff_rg = np.abs(r - g)
    diff_rb = np.abs(r - b)
    diff_gb = np.abs(g - b)
    
    mean_diff = np.mean([np.mean(diff_rg), np.mean(diff_rb), np.mean(diff_gb)])
    
    return mean_diff < threshold


def cluster_bboxes_into_regions(bboxes, img_width, img_height, max_width=1920, max_height=1080):
    '''
    Clusters bounding boxes into the minimum number of non-overlapping image regions.
    
    Parameters:
    - bboxes: List of bounding boxes in the format (x, y, w, h, c) where x, y are top-left.
    - img_width: Width of the original image.
    - img_height: Height of the original image.
    - max_width: Maximum allowable width for a region (default: 1920).
    - max_height: Maximum allowable height for a region (default: 1080).

    Returns:
    - List of region coordinates [(x1, y1, w, h)] representing the cropped regions.
    '''

    bbox_coords = np.array([(x, y, x + w, y + h) for x, y, w, h, _ in bboxes])

    # Sort bounding boxes from top to bottom, then left to right
    bbox_coords = bbox_coords[np.lexsort((bbox_coords[:, 0], bbox_coords[:, 1]))]

    regions = []
    used = set()

    for i, (x1, y1, x2, y2) in enumerate(bbox_coords):
        if i in used:
            continue

        # Start a new region
        region_x1, region_y1 = x1, y1
        region_x2, region_y2 = x2, y2
        region_bboxes = [(x1, y1, x2, y2)]

        # Try to expand region while keeping it within max limits
        for j, (bx1, by1, bx2, by2) in enumerate(bbox_coords[i+1:], start=i+1):
            if j in used:
                continue

            # Compute new potential region bounds
            new_x1, new_y1 = min(region_x1, bx1), min(region_y1, by1)
            new_x2, new_y2 = max(region_x2, bx2), max(region_y2, by2)

            # Check if adding this bbox would exceed the max size
            if (new_x2 - new_x1) > max_width or (new_y2 - new_y1) > max_height:
                continue  # Skip this bbox if it would make the region too large

            # Ensure all bounding boxes remain fully within the region
            for rx1, ry1, rx2, ry2 in region_bboxes:
                if not (new_x1 <= rx1 and new_y1 <= ry1 and new_x2 >= rx2 and new_y2 >= ry2):
                    break  # One bbox would be split, so skip this expansion
            else:
                # If all are still contained, expand the region
                region_x1, region_y1 = new_x1, new_y1
                region_x2, region_y2 = new_x2, new_y2
                region_bboxes.append((bx1, by1, bx2, by2))
                used.add(j)

        # Store the computed region
        regions.append((region_x1, region_y1, (region_x2 - region_x1), (region_y2 - region_y1)))

    return regions


def filter_sparse_rows(cost_matrix):
    '''
    This function helps ensure linear assignment is feasible on a cost matrix
    by whittling down problematic sets of sparse rows. These sets are
    characterized by the following properties:

    - Each row has only one column it could possibly be assigned to. It is the
    one column with a finite cost value in that row; all the others contain "inf".
    - The one viable column in a given row is the one viable column in the
    entire set. In other words, only one row from each set can ultimately be
    matched with a column. 
    
    Once all such sets have been identified, each is reduced to a single row
    (whichever has the lowest match cost to the viable volumn). The other rows
    from each set are filtered from the cost matrix, increasing the number of
    new tracks to subsequently initialize for this frame.
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


def apply_offset(coordinates: Union[Sequence[int], Sequence[Sequence[int]]],
                 offset: Sequence[int]):
    
    if isinstance(coordinates[0], int):
        x = coordinates[0] + offset[0]
        y = coordinates[1] + offset[1]
        return (x, y)
    else:
        offset_coordinates = []
        for xy in coordinates:
            x = xy[0] + offset[0]
            y = xy[1] + offset[1]
            offset_coordinates.append((x, y))
        return offset_coordinates

