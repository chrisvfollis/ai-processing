import pandas as pd
import tensorflow
from deepface import DeepFace
import torch
import cv2
import os
import numpy as np
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from processing.inference import YOLOv4


def run_function():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    video_path = 'input/2025-02-04_14-15-17_2.mp4'
    weights_path = '../../processing/models/YOLOv4.pth'

    yolov4 = YOLOv4(weights_path, device)

    whole_image(video_path, yolov4)


def whole_image(video, yolov4):
    detect_total = 0
    id_total = 0

    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    f_num = -1

    image_num = 0
    face_data = []

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

            person_boxes = [box[:4] for box in detections]

            try:
                start = time.perf_counter()
                df_results = DeepFace.find(
                    img_path=frame, db_path='../../input_files/faces', model_name='Facenet512',
                    detector_backend='retinaface', threshold=0.999,
                    enforce_detection=True, silent=True
                )
                end = time.perf_counter()
                id_total += (end - start)
            except ValueError:
                continue

            filtered_dfs = []
            for df in df_results:
                df = df.loc[df.groupby('identity')['distance'].idxmin()]
                filtered_dfs.append(df)

            print(f'{len(filtered_dfs)} faces found')

            for df in filtered_dfs:
                row = df.iloc[0]
                source_x = row['source_x']
                source_y = row['source_y']
                source_w = row['source_w']
                source_h = row['source_h']
                predicted_identity = row['identity']
                cosine_distance = row['distance']

                if None in [source_x, source_y, source_w, source_h]:
                    print('No coordinates')
                    continue

                face_center = ((source_x + source_x + source_w) / 2, (source_y + source_y + source_h) / 2)

                min_distance = float('inf')
                closest_person_box = None

                for person_box in person_boxes:
                    person_center = (person_box[0] + (person_box[2] / 2),
                                     person_box[1] + (person_box[3] / 2))
                    distance = np.linalg.norm(np.array(face_center) - np.array(person_center))

                    if distance < min_distance:
                        min_distance = distance
                        closest_person_box = person_box

                if closest_person_box is not None:
                    x1, y1, w, h = map(int, closest_person_box[:4])
                    x1, y1, x2, y2 = x1, y1, int(x1 + w), int(y1 + h)
                else:
                    x1, y1, x2, y2 = int(source_x), int(source_y), int(source_x + source_w), int(source_y + source_h)

                cropped_image = frame[y1:y2, x1:x2]

                output_path = os.path.join('output', 'faces', f'{image_num}.jpg')
                cv2.imwrite(output_path, cropped_image)

                face_data.append({
                    'image_num': image_num,
                    'predicted_identity': predicted_identity,
                    'cosine_distance': cosine_distance
                })

                image_num += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f'Method: Feed whole image to DeepFace.find()\n')
    print(f'Total detection time: {detect_total}')
    print(f'Total ID time: {id_total}\n')
    print(f'YOLOv4 input size: {yolov4.resize_dims}')

    face_df = pd.DataFrame(face_data)
    face_df['correct_id'] = ''
    output_path = os.path.join('output', 'face_data.csv')
    face_df.to_csv(output_path, index=False)


def cropped_detections(video, yolov4):
    detect_total = 0
    id_total = 0

    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    f_num = -1

    image_num = 0
    face_data = []

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

            person_boxes = [box[:4] for box in detections]

            for box in person_boxes:
                x1, y1, w, h = map(int, box[:4])
                x2, y2 = (x1 + w), (y1 + h)

                cropped_image = frame[y1:y2, x1:x2]

                try:
                    start = time.perf_counter()
                    df_results = DeepFace.find(
                        img_path=cropped_image, db_path='../../input_files/faces', model_name='Facenet512',
                        detector_backend='retinaface', threshold=0.999,
                        enforce_detection=True, silent=True
                    )
                    end = time.perf_counter()
                    id_total += (end - start)
                except ValueError:
                    continue

            filtered_dfs = []
            for df in df_results:
                df = df.loc[df.groupby('identity')['distance'].idxmin()]
                filtered_dfs.append(df)

            print(f'{len(filtered_dfs)} faces found')

            for df in filtered_dfs:
                row = df.iloc[0]
                source_x = row['source_x']
                source_y = row['source_y']
                source_w = row['source_w']
                source_h = row['source_h']
                predicted_identity = row['identity']
                cosine_distance = row['distance']

                if None in [source_x, source_y, source_w, source_h]:
                    print('No coordinates')
                    continue

                face_center = ((source_x + source_x + source_w) / 2, (source_y + source_y + source_h) / 2)

                min_distance = float('inf')
                closest_person_box = None

                for person_box in person_boxes:
                    person_center = (person_box[0] + (person_box[2] / 2),
                                     person_box[1] + (person_box[3] / 2))
                    distance = np.linalg.norm(np.array(face_center) - np.array(person_center))

                    if distance < min_distance:
                        min_distance = distance
                        closest_person_box = person_box

                if closest_person_box is not None:
                    x1, y1, w, h = map(int, closest_person_box[:4])
                    x1, y1, x2, y2 = x1, y1, int(x1 + w), int(y1 + h)
                else:
                    x1, y1, x2, y2 = int(source_x), int(source_y), int(source_x + source_w), int(source_y + source_h)

                cropped_image = frame[y1:y2, x1:x2]

                output_path = os.path.join('output', 'faces', f'{image_num}.jpg')
                cv2.imwrite(output_path, cropped_image)

                face_data.append({
                    'image_num': image_num,
                    'predicted_identity': predicted_identity,
                    'cosine_distance': cosine_distance
                })

                image_num += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f'Method: Feed whole image to DeepFace.find()\n')
    print(f'Total detection time: {detect_total}')
    print(f'Total ID time: {id_total}\n')
    print(f'YOLOv4 input size: {yolov4.resize_dims}')

    face_df = pd.DataFrame(face_data)
    face_df['correct_id'] = ''
    output_path = os.path.join('output', 'face_data.csv')
    face_df.to_csv(output_path, index=False)
