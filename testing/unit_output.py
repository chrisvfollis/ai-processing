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


def detect_people_in_image():
    pass


def detect_people_in_video():
    pass


def detect_faces_in_image(image: Union[str, np.ndarray], image_name: str = None):
    detector = CenterFace()

    if isinstance(image, str):
        image = cv2.imread(image)
        image_name = image.split('/')[-1]

    if not image_name:
        image_name = str(uuid.uuid4())
    output_path = os.path.join('../files/output', image_name)
    
    face_detections = detector.detect_faces(image)
    detector.visualize_detections(image, face_detections, output_path=output_path)


def detect_faces_in_video():
    pass


def recognize_faces_in_image():
    pass


def recognize_faces_in_video(video: str, method: str = 'global'):
    if method == 'local':
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
    

    if category == 'detect':
        input_type, input_path = sys.argv[2].split('=')

        if subcategory == 'people':
            if input_type == 'img':
                print('Detecting people in image...')
                detect_people_in_image()
            elif input_type == 'vid':
                print('Detecting people in video...')
                detect_people_in_video()

        elif subcategory == 'faces':
            if input_type == 'img':
                print('Detecting faces in image...')
                detect_faces_in_image(input_path)
            elif input_type == 'vid':
                detect_faces_in_video()
                print('Detecting faces in image...')

    elif category == 'recognize':
        input_type, input_path = sys.argv[2].split('=')

        if input_type == 'img':
            print('Recognizing faces in image...')
            recognize_faces_in_image()
        elif input_type == 'vid':
            print('Recognizing faces in video...')
            recognize_faces_in_video()
