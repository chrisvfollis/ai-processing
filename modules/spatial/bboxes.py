# standard dependencies
from collections.abc import Sequence, Iterable

# 3rd-party dependencies
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box
import torch

# internal dependencies
pass


def expand_bbox(x1, y1, x2, y2, img_w, img_h, margin=0.3):
    '''
    Expands the bounding box by a given margin percentage.

    Args:
        x1, y1, x2, y2: Original bounding box coordinates.
        img_w, img_h: Dimensions of the original image.
        margin (float): Margin as a fraction of box size.

    Returns:
        Expanded (x1, y1, x2, y2), clipped to image bounds.
    '''
    box_w = x2 - x1
    box_h = y2 - y1

    delta_w = int(box_w * margin / 2)
    delta_h = int(box_h * margin / 2)

    new_x1 = max(0, x1 - delta_w)
    new_y1 = max(0, y1 - delta_h)
    new_x2 = min(img_w, x2 + delta_w)
    new_y2 = min(img_h, y2 + delta_h)

    return new_x1, new_y1, new_x2, new_y2


def expand_bbox_asym(x1, y1, x2, y2, img_w, img_h, top=0.05, bottom=0.15, left=0.05, right=0.05):
    w = x2 - x1
    h = y2 - y1
    new_x1 = max(int(x1 - left * w), 0)
    new_x2 = min(int(x2 + right * w), img_w)
    new_y1 = max(int(y1 - top * h), 0)
    new_y2 = min(int(y2 + bottom * h), img_h)
    return new_x1, new_y1, new_x2, new_y2


def compute_overlap_ratio(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    faceArea = boxA[2] * boxA[3]
    
    return interArea / (faceArea + 1e-6)


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


def apply_offset(
        coordinates: Iterable[int] | Iterable[Iterable[int]],
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


# =============================================================================
#                              - REGIONS -
# -----------------------------------------------------------------------------


def cluster_into_regions(
    bboxes: list,
    img_height: int,
    img_width: int,
    max_width: int = 1920,
    max_height: int = 1440,
    min_width: int = 608,
    min_height: int = 448,
    margin: int = 5,
    nms_thresh: float = 0.80,
) -> list[tuple[int, int, int, int]]:
    '''
    Clusters bounding boxes into the minimum number of non-overlapping image
    regions, within bounded dimensions and aspect ratios for optimal CenterFace
    detection performance.

    Returns a list of (x, y, w, h) region coordinate tuples.
    '''
    target_ar_min = 0.75
    target_ar_max = 1.33

    bbox_coords = np.array([xywh_xyxy(box, out='xyxy') for box in bboxes])
    bbox_coords = bbox_coords[np.lexsort((bbox_coords[:, 0], bbox_coords[:, 1]))]

    regions = []
    used = set()

    for i, (x1, y1, x2, y2) in enumerate(bbox_coords):
        if i in used:
            continue

        used.add(i)
        region_bboxes = [(x1, y1, x2, y2)]

        region_x1 = x1
        region_y1 = y1
        region_x2 = x2
        region_y2 = y2

        for j, (bx1, by1, bx2, by2) in enumerate(bbox_coords[i+1:], start=i+1):
            if j in used:
                continue
            
            new_x1 = min(region_x1, bx1)
            new_y1 = min(region_y1, by1)
            new_x2 = max(region_x2, bx2)
            new_y2 = max(region_y2, by2)

            new_w = new_x2 - new_x1
            new_h = new_y2 - new_y1
            
            ar = new_w / new_h
            if ar < target_ar_min:
                new_w = int(new_h * target_ar_min)
                new_x2 = new_x1 + new_w
                if new_x2 > img_width:
                    continue
            elif ar > target_ar_max:
                new_h = int(new_w / target_ar_max)
                new_y2 = new_y1 + new_h
                if new_y2 > img_height:
                    continue
        
            if (new_w > max_width) and (new_h > max_height):
                continue
            else:
                region_x1 = new_x1
                region_y1 = new_y1
                region_x2 = new_x2
                region_y2 = new_y2

                region_bboxes.append((bx1, by1, bx2, by2))
                used.add(j)

        region_w = region_x2 - region_x1
        region_h = region_y2 - region_y1

        if len(region_bboxes) == 1:
            ar = region_w / region_h
            if ar < target_ar_max:
                region_w = int(region_h * target_ar_max)

        # ensure minimum size
        region_w = max(region_w, min_width)
        region_h = max(region_h, min_height)

        new_x2 = region_x1 + region_w
        new_y2 = region_y1 + region_h

        adjust_x = new_x2 - img_width
        adjust_y = new_y2 - img_height

        if adjust_x > 0:
            region_x1 -= adjust_x
        else:
            region_x2 = new_x2
        if adjust_y > 0:
            region_y1 -= adjust_y
        else:
            region_y2 = new_y2
        
        # apply margin
        pixel_margin = margin * 10 if len(region_bboxes) == 1 else margin

        region_x1 = max(0, region_x1 - pixel_margin)
        region_y1 = max(0, region_y1 - pixel_margin)
        region_x2 = min(img_width, region_x2 + pixel_margin)
        region_y2 = min(img_height, region_y2 + pixel_margin)

        regions.append((
            region_x1,
            region_y1,
            region_x2 - region_x1,
            region_y2 - region_y1,
        ))

    return region_box_nms(regions, nms_thresh)


def region_box_nms(
    regions: list[tuple],
    nms_thresh: float = 0.80,
) -> list[tuple[int, int, int, int]]:
    '''
    Suppresses overlapping regions using a form of NMS based on area instead of
    confidence (keeps the largest overlapping region). 
    '''
    if not regions:
        return []

    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in regions])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)
        inds_to_keep = np.where(iou < nms_thresh)[0]

        order = order[inds_to_keep + 1]

    return [regions[i] for i in keep]


def crop_region(img, region):
    x1, y1, x2, y2 = [int(round(v)) for v in region[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()
