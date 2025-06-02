# standard dependencies
from typing import Tuple
import os

# 3rd-party dependencies
import numpy as np
import cv2
from deepface.models.Detector import FacialAreaRegion

# internal dependencies
from utilities import general_utils as utils


def adjust_and_extract(
        detection: FacialAreaRegion,
        source_img: np.ndarray,
        expand_percentage: int = 0,
        align: bool = False,
        width_border: int = 0,
        height_border: int = 0,
        save_data: bool = False,
        data_index: tuple = None
    ):
    '''
    Applies any relevant adjustments to the detection, then extracts its
    image from the source image by cropping the detected facial area.
    '''
    detection.confidence = detection.confidence or 0

    if save_data:
        index_prefix = f'{data_index[0]}_{data_index[1]}'
        starting_path = os.path.join('../files/output', index_prefix)

        unaligned_path = f'{starting_path}_unaligned_detection.png'
        if align:
            aligned_path = f'{starting_path}_aligned_detection.png'
        
    if expand_percentage > 0:
        # Expand the facial region height and width by the provided percentage
        # ensuring that the expanded region stays within source_img.shape limits:
        expanded_w = detection.w + int(detection.w * expand_percentage / 100)
        expanded_h = detection.h + int(detection.h * expand_percentage / 100)

        detection.x = max(0, detection.x - int((expanded_w - detection.w) / 2))
        detection.y = max(0, detection.y - int((expanded_h - detection.h) / 2))

        detection.w = min(source_img.shape[1] - detection.x, expanded_w)
        detection.h = min(source_img.shape[0] - detection.y, expanded_h)
    
    if align == False or save_data:
        face_img = source_img[
            int(detection.y) : int(detection.y + detection.h),
            int(detection.x) : int(detection.x + detection.w)
        ]

        if save_data:
            cv2.imwrite(unaligned_path, face_img)   # save unaligned image

    if align == True:
        sub_img, relative_x, relative_y = extract_sub_image(source_img, detection)

        sub_img_xyxy = utils.xywh_xyxy(
            (relative_x, relative_y, detection.w, detection.h)
        )

        aligned_sub_img, angle = align_img_wrt_eyes(sub_img, detection)

        rotated_x1, rotated_y1, rotated_x2, rotated_y2 = project_facial_area(
            facial_area=sub_img_xyxy, angle=angle,
            size=(sub_img.shape[0], sub_img.shape[1]),
        )

        face_img = aligned_sub_img[
            rotated_y1 : rotated_y2,
            rotated_x1 : rotated_x2
        ]
        del aligned_sub_img, sub_img

        if save_data:
            cv2.imwrite(aligned_path, face_img) # save aligned image
        
        detection = reframe_points(detection, width_border, height_border)

    return detection, face_img


def format_response(
        face_obj,
        color_face = None,
        width = None,
        height = None,
        normalize_face: bool = False,
    ) -> dict:
    '''
    Returns:
        dict: A dictionary with:
            - 'face_img' (numpy.ndarray): The (possibly color-converted and normalized) face image.
            - 'facial_area' (dict):
                - 'x' (int): Clamped X coordinate.
                - 'y' (int): Clamped Y coordinate.
                - 'w' (int): Clamped width.
                - 'h' (int): Clamped height.
                - 'left_eye' (any, optional): Included only if not None.
                - 'right_eye' (any, optional): Included only if not None.
                - 'nose' (any, optional): Included only if not None.
                - 'mouth_left' (any, optional): Included only if not None.
                - 'mouth_right' (any, optional): Included only if not None.
            - 'confidence' (float): Confidence score rounded to two decimals.

    Notes:
        - Landmark keys ('nose', 'mouth_left', 'mouth_right') are omitted if their value is None.
        - 'face_img' is ready for downstream use, respecting color and normalization settings.
    '''
    facial_area, face_img = face_obj.facial_area, face_obj.img

    face_img = convert_color(face_img, color_face)
    if normalize_face:
        face_img = face_img / 255  # normalize input in [0, 1]

    # cast to int for flask, and do final checks for borders
    x = max(0, int(facial_area.x))
    y = max(0, int(facial_area.y))
    w = min(width - x - 1, int(facial_area.w))
    h = min(height - y - 1, int(facial_area.h))

    facial_area_dict = {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'left_eye': facial_area.left_eye,
        'right_eye': facial_area.right_eye,
    }
    lower_facial_area_dict = {
        'nose': facial_area.nose,
        'mouth_left': facial_area.mouth_left,
        'mouth_right': facial_area.mouth_right,
    }
    facial_area_dict = facial_area_dict | lower_facial_area_dict
    
    for k in lower_facial_area_dict.keys():
        if facial_area_dict[k] is None:
            del facial_area_dict[k]

    resp_obj = {
        'face_img': face_img,
        'facial_area': facial_area_dict,
        'confidence': round(float(facial_area.confidence or 0), 2),
    }
    return resp_obj


def reframe_points(detection: FacialAreaRegion, width_border, height_border):
    def _subtract_borders(xy_coordinate, border_w, border_h):
        x, y = xy_coordinate
        return ((x - border_w), (y - border_h))
    
    borders = [width_border, height_border]
    if sum(borders) == 0:
        return detection
    
    # Reframe face:
    detection.x, detection.y = _subtract_borders(
        (detection.x, detection.y), *borders
    )

    # Reframe eyes:
    left_eye, right_eye = detection.left_eye, detection.right_eye
    if left_eye is not None:
        left_eye = _subtract_borders(left_eye, *borders)
    if right_eye is not None:
        right_eye = _subtract_borders(right_eye, *borders)
    
    detection.left_eye = left_eye
    detection.right_eye = right_eye

    # Reframe nose:
    nose = detection.nose
    if nose is not None:
        nose = _subtract_borders(nose, *borders)
    
    detection.nose = nose
    
    # Reframe mouth:
    mouth_left, mouth_right = detection.mouth_left, detection.mouth_right
    if mouth_left is not None:
        mouth_left = _subtract_borders(mouth_left, *borders)
    if mouth_right is not None:
        mouth_right = _subtract_borders(mouth_right, *borders)
    
    detection.mouth_left = mouth_left
    detection.mouth_right = mouth_right

    return detection


def extract_sub_image(img: np.ndarray, detection: FacialAreaRegion):
    '''
    Get the sub image with given facial area while expanding the facial region
    to ensure alignment does not shift the face outside the image.

    This function doubles the height and width of the face region,
    and adds black pixels if necessary.
    '''
    x, y, w, h = detection.x, detection.y, detection.w, detection.h
    x1, y1, x2, y2 = utils.xywh_xyxy((x, y, w, h))

    relative_x = int(0.5 * w)
    relative_y = int(0.5 * h)

    # Calculate expanded coordinates:
    x1, y1 = (x1 - relative_x), (y1 - relative_y)
    x2, y2 = (x2 + relative_x), (y2 + relative_y)
    
    if x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
        extracted_face = img[y1:y2, x1:x2]

    else:   # Add black pixels where the expanded region exceeds the image boundary:
        expanded_dims = (
            (y2 - y1), (x2 - x1), img.shape[2]
        )
        extracted_face = np.zeros(expanded_dims, dtype=img.dtype) # black image

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        cropped_region = img[y1:y2, x1:x2]

        # Map the cropped region:
        start_x = max(0, relative_x - x)
        start_y = max(0, relative_y - y)

        extracted_face[
            start_y : start_y + cropped_region.shape[0],
            start_x : start_x + cropped_region.shape[1]
        ] = cropped_region

    return extracted_face, relative_x, relative_y


def align_img_wrt_eyes(img: np.ndarray, detection: FacialAreaRegion):
    '''
    Aligns the image horizontally with respect to the left and right eye
    locations of a face_detection
    '''
    
    left_eye = detection.left_eye
    right_eye = detection.right_eye

    if (
        ((left_eye is None) or (right_eye is None)) or
        ((img.shape[0] == 0) or (img.shape[1] == 0))
    ):
        return img, 0

    angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    return img, angle


def project_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    '''
    Update pre-calculated facial area coordinates after image itself
        rotated with respect to the eyes.
    Inspried from the work of @UmutDeniz26 - github.com/serengil/retinaface/pull/80

    Args:
        facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
            x2 is equal to x1 + w1, and y2 is equal to y1 + h1
        angle (float): Angle of rotation in degrees. Its sign determines the direction of rotation.
                    Note that angles > 360 degrees are normalized to the range [0, 360).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        rotated_coordinates (tuple of int): Representing the new coordinates
            (x1, y1, x2, y2) or (x1, y1, x1+w1, y1+h1) of the rotated facial area.
    '''

    # Normalize the witdh of the angle so we don't have to
    # worry about rotations greater than 360 degrees.
    # We workaround the quirky behavior of the modulo operator
    # for negative angle values.
    direction = 1 if angle >= 0 else -1
    angle = abs(angle) % 360
    if angle == 0:
        return facial_area

    # Angle in radians
    angle = angle * np.pi / 180

    height, width = size

    # Translate the facial area to the center of the image
    x = ((facial_area[0] + facial_area[2]) / 2) - (width / 2)
    y = ((facial_area[1] + facial_area[3]) / 2) - (height / 2)

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + width / 2
    y_new = y_new + height / 2

    # Calculate projected coordinates after alignment
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    # validate projected coordinates are in image's boundaries
    x1 = int(max(x1, 0))
    y1 = int(max(y1, 0))
    x2 = int(min(x2, width))
    y2 = int(min(y2, height))

    return (x1, y1, x2, y2)


def convert_color(face_img, color_face):
    if color_face == 'rgb':
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    elif color_face == 'bgr':
        pass  # image is in BGR
    elif color_face == 'gray':
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f'The color_face can be rgb, bgr or gray, but it is {color_face}.')
    
    return face_img
