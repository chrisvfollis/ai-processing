# standard dependencies
import os
import sys
from typing import Union
import uuid
import time
import math

# 3rd-party dependencies
import numpy as np
import cv2
import torch

# internal dependencies
from models.face_iq import FaceIq
from models.centerface import CenterFace
from models.clearface import ClearFace
from models.yolov4 import YOLOv4
from utilities import utilities as utils
from utilities import io_utils


# ----------------------------------------------------------------------------
# Image Processing:


def detect_people_in_image():
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
    print(f'{len(face_detections)} faces detected')
    detector.visualize_detections(image, face_detections, output_path=output_path)


def recognize_faces_in_image(image: Union[str, np.ndarray],
                             image_name: str = None, focus='global'):
    
    face_iq = FaceIq('Facenet512', 'centerface_gpu', save_data=True)

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)
    
    if isinstance(image, str):
        image_name = image.split('/')[-1]
        image = cv2.imread(image)
    
    if not image_name:
        image_name = str(uuid.uuid4())
    output_path = os.path.join('../files/output', image_name)

    resolution = image.shape[:2][::-1]

    if focus == 'face':
        face_df = face_iq.recognize(image, id_cutoff=0.999)
        if not face_df.empty:
            for name, distance in (
                face_df[['name', 'distance']].itertuples(
                    index=False, name=None
                )
            ):
                print(f'Name: {name} | Distance: {distance}')

            face_iq.visualize_identifications(
                image, [face_df], output_path=output_path
            )
        return

    elif focus == 'global':
        regions = None

    elif focus == 'local':
        bboxes = yolov4.detect(image, 0)

        if not bboxes:
            return

        regions = utils.cluster_bboxes_into_regions(
            bboxes, *resolution
        )

    face_dfs = face_iq.identify_faces(image, id_cutoff=0.999, regions=regions)

    print(f'{len(face_dfs)} faces found...')
    for i, face_df in enumerate(face_dfs):
        if face_df.empty:
            continue

        for identity, distance in (
            face_df[['identity', 'distance']].itertuples(index=False, name=None)
        ):

            first_name, _ = io_utils.lookup_name(identity)

            print(f'Detection: {i} | Name: {first_name} | Distance: {distance}')

    face_iq.visualize_identifications(image, face_dfs, output_path=output_path)


def enhance_face(image: Union[str, np.ndarray], image_name: str = None):
    clearface = ClearFace(weights_path='../models/weights/clearface/90000_G.pth')

    if isinstance(image, str):
        image_name = image.split('/')[-1]
        image = cv2.imread(image)

    if not image_name:
        image_name = str(uuid.uuid4())
    output_path = os.path.join('../files/output', image_name)

    enhanced_face = clearface.forward(image, is_rgb=True)

    cv2.imwrite(output_path, enhanced_face)


# ----------------------------------------------------------------------------
# Video Processing:


def detect_people_in_video():
    pass


def detect_faces_in_video(
        video_file: str,
        focus: str = 'global',
        input_dir: str = '../files/input',
        output_dir: str = '../files/output'
    ):

    detector = CenterFace(save_data=True)

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)

        detector.regions = {}

    video_path = os.path.join(input_dir, video_file)

    output_filename = io_utils.get_unique_filename(
        output_dir, f'{video_file.split(".")[0]}_face_detections.mp4'
    )
    output_path = os.path.join(output_dir, output_filename)

    cap = cv2.VideoCapture(video_path)
    resolution, fps, total_frames = utils.get_video_info(cap, release=False)

    print(f'Resolution: {resolution}')
    print(f'FPS: {fps}')
    print(f'Total Frames: {total_frames}')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
    
    f_num = -1
    detector.i = f_num

    while f_num < total_frames:
        f_num += 1
        detector.i = f_num
        ret, frame = cap.read()
        if not ret:
            break

        if f_num % 500 == 0:
            print(f_num)

        if (f_num % fps) == 0:
            face_detections = []
    
            if focus == 'global':
                face_detections = detector.detect_faces(frame)
            
            elif focus == 'local':
                bboxes = yolov4.detect(frame, 0)

                if not bboxes:
                    continue
        
                regions = utils.cluster_bboxes_into_regions(
                    bboxes, *resolution
                )

                for region in regions:
                    x1, y1, x2, y2 = utils.xywh_xyxy(region)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    if detector.save_data:
                        detector.regions.setdefault(f_num, []).append(region)

                    frame_crop = utils.crop_region(frame, region)

                    region_face_detections = detector.detect_faces(
                        frame_crop, offset=region
                    )

                    face_detections += region_face_detections

            frame = detector.visualize_detections(frame, face_detections)

        cv2.putText(
            frame, f'frame {f_num}',
            (int(resolution[0]/2), int(resolution[1]/2)),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2
        )

        frame = cv2.resize(frame, (1920, 1080))
        out.write(frame)

        f_num += 1
        detector.i = f_num

    out.release()
    cap.release()

    detector.save_runtime_data()


def recognize_faces_in_video(
        video_file: str,
        focus: str = 'global',
        input_dir: str = '../files/input',
        output_dir: str = '../files/output'
    ):

    face_iq = FaceIq('Facenet512', 'centerface_gpu', save_data=True)

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)

    video_path = os.path.join(input_dir, video_file)

    output_filename = io_utils.get_unique_filename(
        output_dir, f'{video_file.split(".")[0]}_face_identifications.mp4'
    )
    output_path = os.path.join(output_dir, output_filename)

    cap = cv2.VideoCapture(video_path)
    resolution, fps, total_frames = utils.get_video_info(cap, release=False)

    print(f'Resolution: {resolution}')
    print(f'FPS: {fps}')
    print(f'Total Frames: {total_frames}')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

    f_num = -1
    face_iq.i = f_num

    while f_num < total_frames:
        f_num += 1
        face_iq.i = f_num
        ret, frame = cap.read()
        if not ret:
            break

        if f_num % 500 == 0:
            print(f_num)

        if (f_num % fps) == 0:
            if focus == 'global':
                regions = []
    
            elif focus == 'local':
                bboxes = yolov4.detect(frame, 0)

                if not bboxes:
                    continue
    
                regions = utils.cluster_bboxes_into_regions(
                    bboxes, *resolution
                )
                for region in regions:
                    x1, y1, x2, y2 = utils.xywh_xyxy(region)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
            all_face_dfs = face_iq.identify_faces(frame, id_cutoff=0.999, regions=regions)
            heatmaps = face_iq.face_detector.heatmaps

            frame = face_iq.visualize_identifications(frame, all_face_dfs)
            frame = face_iq.face_detector.visualize_heatmaps(frame, heatmaps, regions)

            face_iq.face_detector.heatmaps = []

        cv2.putText(
            frame, f'frame {f_num}',
            (int(resolution[0]/2), int(resolution[1]/2)),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2
        )

        frame = cv2.resize(frame, (1920, 1080))
        out.write(frame)
    
    cap.release()
    out.release()

    face_iq.save_runtime_data()


def test_enhanced_face_detections(
        video: str, focus: str = 'global',
        input_dir: str = '../files/input',
        output_dir: str = '../files/output'
    ):

    face_iq = FaceIq('Facenet512', 'centerface_gpu', save_data=True)

    if focus == 'local':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4('../models/weights/YOLOv4.pth', device)

    cap = cv2.VideoCapture(video)
    resolution, fps, total_frames = utils.get_video_info(cap, release=False)

    f_num = -1
    face_iq.i = f_num

    while f_num < total_frames:
        f_num += 1
        face_iq.i = f_num
        ret, frame = cap.read()
        if not ret:
            break

        if f_num % 500 == 0:
            print(f_num)

        if (f_num % fps) == 0:
            if focus == 'global':
                face_objects = face_iq.detection_pipeline(
                    frame, align=False, enhance=True, normalize_face=False
                )
    
            elif focus == 'local':
                bboxes = yolov4.detect(frame, 0)

                if not bboxes:
                    continue
    
                regions = utils.cluster_bboxes_into_regions(
                    bboxes, *resolution
                )
                for region in regions:
                    img_crop = utils.crop_region(frame, region)
                    local_face_objects = face_iq.detection_pipeline(img_crop, normalize_face=False)
    
    cap.release()

    face_iq.save_runtime_data()


if __name__ == '__main__':
    all_args = sys.argv

    unit_categorization = sys.argv[1].split('=')

    category, subcategory = (
        unit_categorization if (len(unit_categorization) == 2) else 
        (unit_categorization[0], None)
    ) 
    
    input_path = sys.argv[2]
    file_extension = input_path.split('.')[-1]
    print(f'File extension: {file_extension}')

    if category == 'detect':
        if subcategory == 'people':
            print('Detecting: people')
            if file_extension in ['png', 'jpg', 'jpeg']:
                print('Input: image')
                detect_people_in_image()
            elif file_extension == 'mp4':
                print('Input: video')
                detect_people_in_video()

        elif subcategory == 'faces':
            print('Detecting: faces')
            if file_extension in ['png', 'jpg', 'jpeg']:
                print('Input: image')
                detect_faces_in_image(input_path)
            elif file_extension == 'mp4':
                print('Input: video')
                focus = sys.argv[3]
                detect_faces_in_video(input_path, focus=focus)

    elif category == 'recognize':
        focus = sys.argv[3]
        if file_extension in ['png', 'jpg', 'jpeg']:
            print('Input: image')
            recognize_faces_in_image(input_path, focus=focus)
        elif file_extension == 'mp4':
            print('Input: video')
            
            recognize_faces_in_video(input_path, focus=focus)
    
    elif category == 'enhance':
        if file_extension in ['png', 'jpg', 'jpeg']:
            enhance_face(input_path)
        elif file_extension == 'mp4':
            focus = sys.argv[3]
            test_enhanced_face_detections(input_path, focus=focus)
