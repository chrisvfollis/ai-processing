import csv
from shapely.geometry import Polygon, box
from datetime import datetime, timedelta
import torch
import math


def centroid(coordinates):
    '''
    Returns centroid from [x1, y1, w, h] where (x1, y1) are the
    coordinates of the bounding box's top left corner.
    '''
    if (coordinates is not None):
        x = coordinates[0] + coordinates[2] / 2
        y = coordinates[1] + coordinates[3] / 2
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


def xywh_to_4corners(lst):
    lt = (lst[0], lst[1])
    rt = (lst[0] + lst[2], lst[1])

    lb = (lst[0], lst[1] + lst[3])
    rb = (lst[0] + lst[2], lst[1] + lst[3])

    return [lt, rt, lb, rb]


def intersection_box(rect1, rect2):
    lt_1, rt_1, lb_1, rb_1 = xywh_to_4corners(rect1)
    lt_2, rt_2, lb_2, rb_2 = xywh_to_4corners(rect2)

    inter_ltx, inter_lty = (max([lt_1[0], lt_2[0]]), max([lt_1[1], lt_2[1]]))

    if inter_lty >= min([rb_1[1], rb_2[1]]):
        return 0
    elif inter_ltx >= min([rt_1[0], rt_2[0]]):
        return 0

    inter_w = (min([rt_1[0], rt_2[0]])) - inter_ltx
    inter_h = (min([lb_1[1], lb_2[1]])) - inter_lty

    return [inter_ltx, inter_lty, inter_w, inter_h]


def i_over_u(rect1, rect2):
    '''
    Input format — [x1, y1, w, h] 
    '''
    try:
        i_w, i_h = intersection_box(rect1, rect2)[2:]
        intersection = i_w * i_h
    except Exception:
        intersection = intersection_box(rect1, rect2)

    union = (rect1[2] * rect1[3]) + (rect2[2] * rect2[3]) - intersection

    if union == 0:
        return 0

    return intersection / union


def percent_in_polygon(bbox, polygon_points):
    '''
    --------------------------------------------------
    bbox format:
    data — [x1, y1, w, h]
    order — from top left
    --------------------------------------------------
    polygon format:
    data — [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    order — from top left, clockwise
    --------------------------------------------------               
    '''

    polygon = Polygon(polygon_points)
    
    bbox_polygon = box(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    
    intersection = polygon.intersection(bbox_polygon).area
    area = bbox[2] * bbox[3]
    
    if area == 0:
        return 0
    
    return intersection / area


def read_detection_csv(csv_path):
    frame_data = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)

        for row in csvreader:
            frame = int(row[0])
            x, y, w, h = map(int, row[1:5])

            if frame not in frame_data:
                frame_data[frame] = []
            
            frame_data[frame].append([x, y, w, h])

    return frame_data


def is_coincident(span1, span2):
    '''
    Checks whether a span is coincident with another at any point.
    '''
    return not (span1[1] < span2[0] or span2[1] < span1[0])


def frame_timestamp(clip_timestamp, frame, fps=30):
    if isinstance(clip_timestamp, str):
        clip_timestamp = clip_timestamp.replace('_', ':', 2).replace('_', ' ')
        clip_timestamp = datetime.strptime(clip_timestamp, '%Y-%m-%d %H:%M:%S')

    seconds = frame / fps
    return clip_timestamp + timedelta(seconds=seconds)