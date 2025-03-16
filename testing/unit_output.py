# standard dependencies
import os
import sys
from typing import Union
import uuid
import time
import math
from collections import deque

# 3rd-party dependencies
import numpy as np
import cv2
import torch

# internal dependencies
from models.face_iq import CenterFace
from models.yolov4 import YOLOv4
from utilities import utilities as utils
from utilities import io_utils


def detect_people_in_image():
    pass


def detect_people_in_video():
    pass


def detect_faces_in_image(image: Union[str, np.ndarray], image_name: str = None):
    detector = CenterFace()

    if isinstance(image, str):
        image_name = image.split('/')[-1]
        image = cv2.imread(image)

    if not image_name:
        image_name = str(uuid.uuid4())
    output_path = os.path.join('../files/output', image_name)
    
    face_detections = detector.detect_faces(image)
    detector.visualize_detections(image, face_detections, output_path=output_path)


def detect_faces_in_video(
        video: str, focus: str = 'global',
        output_dir: str = '../files/output'
    ):

    detector = CenterFace()

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)

    cap = cv2.VideoCapture(video)
    resolution, fps, total_frames = utils.get_video_info(cap, release=False)
    
    print(f'Resolution: {resolution}')
    print(f'FPS: {fps}')
    print(f'Total Frames: {total_frames}')
    
    video_file = video.split('/')[-1]
    prefix = video_file.split('.')[0]

    filename = io_utils.get_unique_filename(
        output_dir, f'{prefix}_face_detections.mp4'
    )
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, filename),
                          fourcc, fps, (1920, 1080))
    
    f_num = 0
    while f_num < total_frames:
        ret, frame = cap.read()
        if not ret:
            print('Nothing returned')
            break

        if (f_num % fps) == 0:
            if focus == 'global':
                face_detections = detector.detect_faces(frame)
            
            elif focus == 'local':
                face_detections = []
                bboxes = yolov4.detect(frame, 0)
                if not bboxes:
                    continue
                regions = utils.cluster_bboxes_into_regions(
                    bboxes, *resolution
                )

                for region in regions:
                    x1, y1 = region[0], region[1]
                    x2, y2 = region[0] + region[2], region[1] + region[3]

                    region_crop = frame[y1:y2, x1:x2].copy()
                    region_face_detections = detector.detect_faces(
                        region_crop, offset=region
                    )

                    face_detections.append(region_face_detections)
    
            frame = detector.visualize_detections(frame, face_detections)

        frame = cv2.resize(frame, (1920, 1080))
        out.write(frame)
        f_num += 1

    out.release()
    cap.release()


def recognize_faces_in_image():
    pass


def recognize_faces_in_video(video: str, focus: str = 'global'):
    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)

    cap = cv2.VideoCapture(video)


if __name__ == '__main__':
    all_args = sys.argv

    unit_categorization = sys.argv[1].split('=')

    category, subcategory = (
        unit_categorization if (len(unit_categorization) == 2) else 
        (unit_categorization[0], None)
    ) 
    
    input_path = sys.argv[2]
    file_extension = input_path.split('.')[-1]

    if category == 'detect':
        if subcategory == 'people':
            if file_extension in ['.png', '.jpg', '.jpeg']:
                detect_people_in_image()
            elif file_extension == '.mp4':
                detect_people_in_video()

        elif subcategory == 'faces':
            if file_extension in ['.png', '.jpg', '.jpeg']:
                detect_faces_in_image(input_path)
            elif file_extension == '.mp4':
                focus = sys.argv[3]
                detect_faces_in_video(input_path, focus=focus)

    elif category == 'recognize':
        if file_extension in ['.png', '.jpg', '.jpeg']:
            recognize_faces_in_image()
        elif file_extension == '.mp4':
            recognize_faces_in_video()
