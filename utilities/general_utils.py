# standard dependencies
import os
import subprocess
from datetime import datetime, timedelta
from typing import Union
from collections.abc import Sequence, Iterable
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
from shapely.geometry import Polygon, box
import torch

# internal dependencies
from utilities import io_utils


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
    frame_diag = math.dist([0, 0], resolution)

    total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(source.get(cv2.CAP_PROP_FPS))


    if release:
        source.release()

    return resolution, frame_diag, fps, total_frames


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


def frame_timestamp(clip_timestamp, f_num=0, fps=15):
    if isinstance(clip_timestamp, str):
        clip_timestamp = datetime.strptime(clip_timestamp, '%Y-%m-%d_%H-%M-%S')

    seconds = f_num / fps
    return clip_timestamp + timedelta(seconds=seconds)


def parse_clip_filename(video_file):
    if not video_file.endswith('.mp4'):
        return video_file
    
    sections = video_file.rsplit('_', 1)

    time_prefix = sections[0]
    cam_id = sections[1].split('.')[0]

    return time_prefix, cam_id


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


def cluster_bboxes_into_regions(
        bboxes: list,
        img_height: int,
        img_width: int,
        max_width: int = 1920,
        max_height: int = 1080,
        margin: int = 15,
    ) -> list[tuple]:
    '''
    Clusters bounding boxes into the minimum number of non-overlapping image regions.
    
    Args:
        bboxes (list): A list of bounding boxes in the format (x, y, w, h, c)
            where x, y are top-left.
        img_height (int): Height of the original image.
        img_width (int): Width of the original image.
        max_width (int): Maximum allowable width for a region.
        max_height (int): Maximum allowable height for a region.
        margin (int): Number of pixels to expand around each region.

    Returns (list): A list of region coordinates in the format (x1, y1, w, h)
        representing the cropped regions.
    '''
    
    bbox_coords = np.array([xywh_xyxy(box, out='xyxy') for box in bboxes])

    # sort bounding boxes from top to bottom, then left to right:
    bbox_coords = bbox_coords[np.lexsort((bbox_coords[:, 0], bbox_coords[:, 1]))]

    regions = []
    used = set()

    for i, (x1, y1, x2, y2) in enumerate(bbox_coords):
        if i in used:
            continue

        region_x1, region_y1 = x1, y1
        region_x2, region_y2 = x2, y2
        region_bboxes = [(x1, y1, x2, y2)]

        # try to expand region while keeping it within max limits:
        for j, (bx1, by1, bx2, by2) in enumerate(bbox_coords[i+1:], start=i+1):
            if j in used:
                continue

            new_x1, new_y1 = min(region_x1, bx1), min(region_y1, by1)
            new_x2, new_y2 = max(region_x2, bx2), max(region_y2, by2)
  
            if (
                (new_x2 - new_x1) > max_width or
                (new_y2 - new_y1) > max_height
            ):
                continue

            for rx1, ry1, rx2, ry2 in region_bboxes:
                if not (
                    (new_x1 <= rx1) and (new_y1 <= ry1) and
                    (new_x2 >= rx2) and (new_y2 >= ry2)
                    ):
                    break  # one bbox would be split, so skip this expansion
            else:
                region_x1, region_y1 = new_x1, new_y1
                region_x2, region_y2 = new_x2, new_y2
                region_bboxes.append((bx1, by1, bx2, by2)) # expand the region
                used.add(j)

        # apply margin, clipped to image bounds:
        region_x1 = max(0, region_x1 - margin)
        region_y1 = max(0, region_y1 - margin)
        region_x2 = min(img_width, region_x2 + margin)
        region_y2 = min(img_height, region_y2 + margin)

        region_w = region_x2 - region_x1
        region_h = region_y2 - region_y1
        
        regions.append((region_x1, region_y1, region_w, region_h))

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


def apply_offset(
        coordinates: Union[Iterable[int], Iterable[Iterable]],
        offset: Sequence[int]
    ):
    '''
    Recursively applies a given (x, y) offset to an arbitrarily nested
    set of coordinates.

    Args:
        coordinates (Iterable): Either an iterable for the coordinates of a
        single point, or an iterable of such iterables (with as many layers as
        needed).  

        offset (Sequence): Contains the x and y with which the coordinates
        are summed.
    '''

    if isinstance(coordinates, (np.ndarray, pd.Series)):
        coordinates = coordinates.tolist() 

    if isinstance(coordinates[0], (Iterable)):
        return [
            apply_offset(coord, offset) for coord in coordinates
        ]
    else:
        x0, y0 = offset[:2]
        x_, y_ = coordinates[:2]

        x1, y1 = int(x_ + x0), int(y_ + y0)

        return (x1, y1)


def crop_region(img, region):
    x1, y1, x2, y2 = [int(round(v)) for v in region[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def xywh_xyxy(coordinates, out='xyxy'):
    if torch.is_tensor(coordinates):
        coordinates = coordinates.detach().cpu().numpy()
    elif not isinstance(coordinates, np.ndarray):
        coordinates = np.asarray(coordinates, dtype=float)

    if out == 'xyxy':
        x1, y1, w, h = coordinates[:4]

        x2 = x1 + w
        y2 = y1 + h

        return x1, y1, x2, y2
    
    elif out == 'xywh':
        x1, y1, x2, y2 = coordinates[:4]

        w = x2 - x1
        h = y2 -y1

        return x1, y1, w, h


def c_xywh(coordinates, in_fmt='xywh'):
    if in_fmt == 'xywh':
        x, y, w, h = coordinates[:4]
    elif in_fmt == 'xyxy':
        x, y, w, h = xywh_xyxy(coordinates, out='xywh')
        
    cx = x + (w / 2)
    cy = y + (h / 2)

    return cx, cy, w, h


def logceil_round(x):
    '''
    Rounds a number up to the next "nice" number based on its order of magnitude.
    Useful for generating human-friendly intervals (e.g. for chart axes or scaling
    in general).
    
    Returns:
        int: the smallest number in the set {1, 2, 5, 10} x 10ⁿ that is greater
        than or equal to `x`, where n is the base-10 order of magnitude of `x`
    
    Examples:
        >>> logceil_round(16)
        20
        >>> logceil_round(72)
        100
        >>> logceil_round(630)
        1000
    '''

    if x == 0:
        return 0
    magnitude = 10 ** math.floor(math.log10(x))
    steps = [1, 2, 5, 10]
    for step in steps:
        rounded = step * magnitude
        if x <= rounded:
            return rounded
    return 10 * magnitude


def query_param_placeholders(items: Union[list, tuple]) -> str:
    '''
    Args:
        items (list or tuple): A collection of items such as column names.

    Returns:
        str: A string of comma-separated question marks enclosed in parentheses,
            with one question mark for each value in `items`.
    
    Example:
        >>> query_param_placeholders(['col1', 'col2', 'col3'])
        "(?, ?, ?)"
    '''
    question_marks = ['?'] * len(items)
    return f"({', '.join(question_marks)})"


def query_columns_string(columns: Union[list, tuple]) -> str:
    '''
    Converts a collection of table columns into a comma-separated string
    enclosed in parentheses, for use in query strings.
    '''
    return f"({', '.join(columns)})"


def create_track_df(time_prefix: str) -> pd.DataFrame:
    results = io_utils.get_track_info(time_prefix, designation='tracked_employee')
    if (not results) or (len(results) == 0):
        print('No tracked_employee tracks found')
        return None
    
    columns = [
        'id', 'track_id', 'camera', 'time_prefix', 'identity', 'id_method',
        'id_cost', 'start_img', 'end_img', 'id_img',  'start_time', 'end_time',
        'entry', 'exit', 'designation'
    ]
    track_df = pd.DataFrame(results, columns=columns)

    track_df['start_time'] = pd.to_datetime(track_df['start_time'], format='mixed')
    track_df['end_time'] = pd.to_datetime(track_df['end_time'], format='mixed')

    return track_df


def merge_track_records(
        track_records: pd.DataFrame, max_gap: int = 75
    ) -> pd.DataFrame:
    merged = []
    for identity, group in track_records.groupby('identity'):
        if identity == '':
            merged.extend(group.to_dict(orient='records'))
            continue

        group = group.sort_values('start_time').reset_index(drop=True)
        current = group.iloc[0].to_dict()
        for _, row in group.iloc[1:].iterrows():
            gap = (row['start_time'] - current['end_time']).total_seconds()

            if gap <= max_gap:
                current['end_time'] = max(current['end_time'], row['end_time'])
                current['end_img'] = row['end_img']
            else:
                merged.append(current)
                current = row.to_dict()

        merged.append(current)

    return pd.DataFrame(merged)


def get_default_device() -> torch.device:
    '''Returns cuda:0 if a GPU is available, otherwise the CPU'''
    return torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def calculate_progress(completed, total):
    percent_complete = completed / total * 100

    return int(round(percent_complete, 0))
