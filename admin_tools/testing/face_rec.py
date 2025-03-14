import pandas as pd
import tensorflow
from deepface import DeepFace
import torch
import cv2
import os
import numpy as np
import time
from models.yolov4 import YOLOv4
from models.face_iq import FaceIq
from utilities import utilities as utils
import math


def run_function(fx, video):
    video_path = os.path.join('input', video)

    face_iq = FaceIq('Facenet512', 'centerface_gpu', face_dir='../../files/input/faces',
                     db_path='../../files/data.db',
                     weights_path='../../models/weights/centerface.pth')

    if fx == 'whole_image':
        whole_image(video_path, face_iq)
    elif fx == 'cropped_detections':
        weights_path = '../../models/weights/YOLOv4.pth'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolov4 = YOLOv4(weights_path, device)

        cropped_detections(video_path, yolov4, face_iq)


def whole_image(video, face_iq):
    id_total = 0

    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    f_num = -1

    image_num = 0
    face_data = []

    ex_start = time.perf_counter()
    while f_num < total_frames:
        f_num += 1
        ret, frame = cap.read()
        if not ret:
            break

        if f_num % 500 == 0:
            print(f_num)

        if f_num % fps == 0:

            start = time.perf_counter()
            df_results = face_iq.identify_faces(frame, cutoff=0.999)
            end = time.perf_counter()
            id_total += (end - start)

            df_results = [df for df in df_results if not df.empty]
            print(f'Found {len(df_results)} faces')

            for df in df_results:
                row = df.loc[df['distance'].idxmin()]
                
                try:
                    source_x = int(row['source_x'])
                    source_y = int(row['source_y'])
                    source_w = int(row['source_w'])
                    source_h = int(row['source_h'])
                except (ValueError, TypeError):
                    print('Invalid coordinates')
                    continue

                predicted_identity = row['identity']
                cosine_distance = round(row['distance'], 3)

                if None in [source_x, source_y, source_w, source_h]:
                    print('No coordinates')
                    continue

                x1, y1, x2, y2 = int(source_x), int(source_y), int(source_x + source_w), int(source_y + source_h)

                cropped_image = frame[y1:y2, x1:x2]

                output_path = os.path.join('output', 'faces', f'{image_num}.jpg')
                cv2.imwrite(output_path, cropped_image)

                face_data.append({
                    'img_num': image_num,
                    'identity': predicted_identity,
                    'distance': cosine_distance
                })

                image_num += 1
    
    ex_end = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()

    print(f'\nMethod: Feed whole image to DeepFace.find()')
    print(f'Main execution time: {ex_end - ex_start}\n')
    print(f'Total ID time: {id_total}\n')

    face_df = pd.DataFrame(face_data)
    face_df['correct_id'] = ''
    output_path = os.path.join('output', 'face_data.csv')
    face_df.to_csv(output_path, index=False)


def cropped_detections(video, yolov4, face_iq):
    detect_total = 0
    id_total = 0

    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    resolution = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    f_num = -1

    image_num = 0
    face_data = []

    ex_start = time.perf_counter()
    while f_num < total_frames:
        f_num += 1
        ret, frame = cap.read()
        if not ret:
            break

        if f_num % 500 == 0:
            print(f_num)

        if f_num % fps == 0:
            start = time.perf_counter()
            detections = yolov4.detect(frame, 0, conf_thresh=0.65,
                                       resize_dims=(416, 416))
            end = time.perf_counter()
            detect_total += (end - start)

            person_boxes = [box for box in detections if (math.prod(box[2:4]) > 6400)]
            regions = utils.cluster_bboxes_into_regions(person_boxes, *resolution)

            faces_detected = 0

            start = time.perf_counter()
            df_results = face_iq.identify_faces(frame, cutoff=0.999, regions=regions)
            end = time.perf_counter()
            id_total += (end - start)
                
            if df_results:
                for df in df_results:
                    best_row = df.loc[df['distance'].idxmin()]
                    try:
                        source_x = int(best_row['source_x'])
                        source_y = int(best_row['source_y'])
                        source_w = int(best_row['source_w'])
                        source_h = int(best_row['source_h'])
                    except (ValueError, TypeError):
                        print('Invalid coordinates')
                        continue

                    x1, y1, x2, y2 = int(source_x), int(source_y), int(source_x + source_w), int(source_y + source_h)
                    cropped_image = frame[y1:y2, x1:x2]

                    predicted_identity = best_row['identity']
                    cosine_distance = round(best_row['distance'], 3)
                    faces_detected += 1

                    output_path = os.path.join('output', 'faces', f'{image_num}.jpg')
                    cv2.imwrite(output_path, cropped_image)

                    face_data.append({
                        'img_num': image_num,
                        'identity': predicted_identity,
                        'distance': cosine_distance
                    })

                    image_num += 1
            else:
                continue

            print(f'Found {faces_detected} faces')

    ex_end = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()

    print(f'\nMethod: Feed cropped images to DeepFace.find()')
    print(f'Main execution time: {ex_end - ex_start}\n')
    print(f'Total detection time: {detect_total}')
    print(f'Total ID time: {id_total}\n')
    print(f'YOLOv4 input size: {yolov4.resize_dims}')

    face_df = pd.DataFrame(face_data)
    face_df['correct_id'] = ''
    output_path = os.path.join('output', 'face_data.csv')
    face_df.to_csv(output_path, index=False)
