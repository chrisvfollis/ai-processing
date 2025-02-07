from shapely.geometry import Polygon, box
from datetime import datetime, timedelta
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import subprocess
import os


def get_git_commit_hash(cfg_dir_path='../config'):
    """
    Retrieve the Git commit hash for the version of the codebase that is
    currently running.
    """

    try:
        return subprocess.check_output(["git", "rev-parse", "--short",
                                        "HEAD"]).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        vfile_path = os.path.join(cfg_dir_path, 'version.txt')
        try:
            with open(vfile_path, 'r') as vfile:
                return vfile.read().strip()
        except Exception:
            return "unknown"    
        return "unknown"


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


def restrain_boxes(coordinates, image_size=[1920, 1080]):
    img_width, img_height = image_size

    # Restrain width and height to not exceed the dimensions of
    # the image:
    coordinates[2] = min(img_width, coordinates[2])
    coordinates[3] = min(img_height, coordinates[3])

    # Restrain centroids so that box can go no further than right
    # outside of the frame.
    half_width = coordinates[2] / 2
    half_height = coordinates[3] / 2

    coordinates[0] = min((img_width + half_width), coordinates[0])
    coordinates[0] = max((0 - half_width), coordinates[0])

    coordinates[1] = min((img_height + half_height), coordinates[1])
    coordinates[1] = max((0 - half_height), coordinates[1])

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
    rectangle2, i.e. how much of it is "in" rectangle2.
    '''

    r1_area = rectangle1[2] * rectangle1[3]
    intersection = get_intersection(rectangle1, rectangle2, attr='area')
    if not intersection:
        return 0

    return intersection / r1_area


def percent_in_entryway(bbox, entryway_points):
    '''
    Returns the percent of a bounding box's total area that is "inside" of
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

    x_noise, y_noise, w_noise, h_noise = p_noise

    x_pvar = position_var * x_noise
    y_pvar = position_var * y_noise
    w_pvar = position_var * w_noise
    h_pvar = position_var * h_noise

    x_vvar = velocity_var * x_noise
    y_vvar = velocity_var * y_noise
    w_vvar = velocity_var * w_noise
    h_vvar = velocity_var * h_noise

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


import numpy as np
import cv2

def cluster_bboxes_into_regions(bboxes, img_width, img_height, max_width=1920, max_height=1080):
    """
    Clusters bounding boxes into the minimum number of non-overlapping image regions.
    
    Parameters:
    - bboxes: List of bounding boxes in the format (x, y, w, h, c) where x, y are top-left.
    - img_width: Width of the original image.
    - img_height: Height of the original image.
    - max_width: Maximum allowable width for a region (default: 1920).
    - max_height: Maximum allowable height for a region (default: 1080).

    Returns:
    - List of region coordinates [(x1, y1, x2, y2)] representing the cropped regions.
    """

    # Convert bounding boxes to (x1, y1, x2, y2) format
    bbox_coords = [(x, y, x + w, y + h) for x, y, w, h, c in bboxes]
    bbox_coords = np.array(bbox_coords)

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

        # Try to expand region while keeping it within max limits
        for j, (bx1, by1, bx2, by2) in enumerate(bbox_coords[i+1:], start=i+1):
            if j in used:
                continue

            # Check if adding this bbox would exceed the max size
            new_x1, new_y1 = min(region_x1, bx1), min(region_y1, by1)
            new_x2, new_y2 = max(region_x2, bx2), max(region_y2, by2)

            if (new_x2 - new_x1) <= max_width and (new_y2 - new_y1) <= max_height:
                # Expand region to include this bbox
                region_x1, region_y1 = new_x1, new_y1
                region_x2, region_y2 = new_x2, new_y2
                used.add(j)

        # Store the computed region
        regions.append((region_x1, region_y1, region_x2, region_y2))

    return regions
