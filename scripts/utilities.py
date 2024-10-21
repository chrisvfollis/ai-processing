import csv
from shapely.geometry import Polygon, box


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
