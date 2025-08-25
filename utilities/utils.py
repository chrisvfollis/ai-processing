# standard dependencies
import os
import subprocess
from datetime import datetime, timedelta
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch

# internal dependencies
from utilities import io_utils, log_utils


logger = log_utils.get_logger(__name__)


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
    '''
    Returns:
        tuple[tuple[int, ...], float, int, int]: (resolution, frame_diag, fps, total_frames)
    '''
    if isinstance(source, str):
        source = cv2.VideoCapture(source)
    
    resolution = (int(source.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_diag = math.dist([0, 0], resolution)

    total_frames = int(round(source.get(cv2.CAP_PROP_FRAME_COUNT), 0))
    fps = int(round(source.get(cv2.CAP_PROP_FPS), 0))

    if release:
        source.release()

    return resolution, frame_diag, fps, total_frames


def is_coincident(span1, span2):
    '''
    Checks whether a span is coincident to another at any point.
    '''
    return not (span1[1] < span2[0] or span2[1] < span1[0])


def frame_timestamp(clip_timestamp, f_num=0, fps=15) -> datetime:
    if isinstance(clip_timestamp, str):
        clip_timestamp = datetime.strptime(clip_timestamp, '%Y-%m-%d_%H-%M-%S')

    seconds = f_num / fps
    return clip_timestamp + timedelta(seconds=seconds)


def decode_vid_filename(video_file) -> tuple[str, int]:
    sections = video_file.rsplit('_', 1)

    time_segment = sections[0]
    cam_id = int(sections[1].split('.')[0])

    return time_segment, cam_id


def convert_to_datetime(x: datetime | str | list | None) -> datetime | None:
    '''
    Ensures the given argument is a datetime object by converting as necessary,
    or simply returns None if that is what's provided.
    Accepts:
        - datetime : datetime object
        - str      : whitespace-, comma-, or hyphen-separated datetime elements
        - list     : list of datetime elements
        - None     : None
    '''
    if isinstance(x, datetime) or (x is None):
        return x
    
    if isinstance(x, str):
        if ',' in x:
            x = x.split(',')
        elif '-' in x:
            x = x.split('-')
        else:
            x.split()
        
    if isinstance(x, list):
        x = [element.strip() for element in x if isinstance(element, str)]
        x = map(int, x)
    
    return datetime(*x)


def extract_datetime(footage_filename: str) -> datetime:
    datetime_portions = footage_filename.split('_')[:2]
    datetime_elements_str = '-'.join(datetime_portions)
    
    return convert_to_datetime(datetime_elements_str)


def parse_filename(filename):
    file_parts = filename.rsplit('.', 1)

    if len(file_parts) == 2:
        filename_stem, extension = file_parts
    else:
        filename_stem = file_parts.pop(0)
        extension = ''

    return filename_stem, extension


def parse_obj_key(s3_obj_key) -> tuple[str, ...]:
    obj_key_parts = s3_obj_key.rsplit('/', 1)

    if len(obj_key_parts) == 2:
        folder, filename = obj_key_parts
    else:
        folder = ''
        filename = obj_key_parts.pop(-1)

    return folder, filename


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


def query_param_placeholders(items: list | tuple) -> str:
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


def query_columns_string(columns: list | tuple) -> str:
    '''
    Converts a collection of table columns into a comma-separated string
    enclosed in parentheses, for use in query strings.
    '''
    return f"({', '.join(columns)})"


def get_default_device() -> torch.device:
    '''Returns cuda:0 if a GPU is available, otherwise the CPU'''
    return torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )


def calculate_progress(completed, total):
    percent_complete = completed / total * 100

    return int(round(percent_complete, 0))


def get_segment_info(segment_records: list[tuple]):
    filenames = [row[2] for row in segment_records]
    time_segment, _ = decode_vid_filename(filenames[0])

    return time_segment, filenames
