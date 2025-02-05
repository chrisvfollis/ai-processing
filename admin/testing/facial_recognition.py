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
from processing.models.yolov4_architecture import Yolov4Model


class YOLOv4:
    def __init__(self, weights_path, device, nms_thresh=0.5):
        self.device = device

        self.model = Yolov4Model(inference=True)
        weights = torch.load(weights_path, map_location=device)

        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

        self.nms_thresh = nms_thresh
        self.conf_thresh = 0.70
        self.resize_dims = (416, 416)
        self.detection_time = 0
        
        
    def detect(self, img, class_num, conf_thresh=0.70, resize_dims=(416, 416)):
        def _preprocess_img(img, resize_dims):
            '''
            resize_dims — the width and height to resize the image to. YOLOv4
            only accepts image dimensions that can be expressed using the
            formula (320 + (96 * n)), where n is a positive integer.
        
            Examples of valid dimensions include 320, 416, 512, 608, etc
            '''
            self.resize_dims = resize_dims
            original_dims = img.shape[:2][::-1]
            w, h = resize_dims

            img_resized = cv2.resize(img, (w, h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            img_tensor = (torch.from_numpy(img_rgb.transpose(2, 0, 1))
                          .float().div(255.0).unsqueeze(0))
    
            if self.device.type == 'cuda':
                img_tensor = img_tensor.cuda()

            return img_tensor, original_dims
        
        def _postprocess_output(output):
            def _nms_filter(boxes, confs):
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]

                areas = (x2 - x1) * (y2 - y1)
                order = confs.argsort()[::-1]

                keep = []
                while order.size > 0:
                    idx_self = order[0]
                    idx_other = order[1:]

                    keep.append(idx_self)

                    xx1 = np.maximum(x1[idx_self], x1[idx_other])
                    yy1 = np.maximum(y1[idx_self], y1[idx_other])
                    xx2 = np.minimum(x2[idx_self], x2[idx_other])
                    yy2 = np.minimum(y2[idx_self], y2[idx_other])

                    w = np.maximum(0.0, xx2 - xx1)
                    h = np.maximum(0.0, yy2 - yy1)

                    inter = w * h
                    over = inter / (areas[order[0]] + areas[order[1:]] - inter)

                    inds = np.where(over <= self.nms_thresh)[0]
                    order = order[inds + 1]
                
                return np.array(keep)
    
            # [num, 1, 4]
            box_array = output[0][0]  # Extract first (and only) image
            # [num, num_classes]
            confs = output[1][0]  # Extract first (and only) image

            if type(box_array).__name__ != 'ndarray':
                box_array = box_array.cpu().detach().numpy()
                confs = confs.cpu().detach().numpy()

            num_classes = confs.shape[1]

            # [num, 4]
            box_array = box_array[:, 0]

            # [num, num_classes] --> [num]
            max_conf = np.max(confs, axis=1)
            max_id = np.argmax(confs, axis=1)

            # Filter by confidence threshold
            argwhere = max_conf > self.conf_thresh
            l_box_array = box_array[argwhere, :]
            l_max_conf = max_conf[argwhere]
            l_max_id = max_id[argwhere]

            bboxes = []
            # Non-Maximum Suppression (NMS) for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]

                keep = _nms_filter(ll_box_array, ll_max_conf)

                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append([
                            ll_box_array[k, 0], ll_box_array[k, 1],
                            ll_box_array[k, 2], ll_box_array[k, 3],
                            ll_max_conf[k], ll_max_conf[k], j  # j is class ID
                        ])

            return bboxes

        def _translate_detection(box, original_dims):
            x1, y1, x2, y2 = box
            img_w, img_h = original_dims

            scale_x = img_w
            scale_y = img_h

            x1 = int(round(x1 * scale_x))
            y1 = int(round(y1 * scale_y))
            x2 = int(round(x2 * scale_x))
            y2 = int(round(y2 * scale_y))

            x1 = int(max(0, min(x1, img_w)))
            y1 = int(max(0, min(y1, img_h)))
            x2 = int(max(0, min(x2, img_w)))
            y2 = int(max(0, min(y2, img_h)))

            w = x2 - x1
            h = y2 - y1

            return [x1, y1, w, h]

        start_detect = time.perf_counter()

        self.conf_thresh = conf_thresh

        img, original_dims = _preprocess_img(img, resize_dims)
        with torch.no_grad():
            raw_output = self.model(img)
        detections = _postprocess_output(raw_output)

        filtered = []
        for detection in detections:
            x1, y1, x2, y2, confidence, _, class_id = detection
            bbox = [x1, y1, x2, y2]
            confidence = float(confidence)

            if int(class_id) == class_num:
                x, y, w, h = _translate_detection(
                    bbox, original_dims
                )
                filtered.append([x, y, w, h, confidence])

        end_detect = time.perf_counter()
        self.detection_time += (end_detect - start_detect)

        return filtered


def run_function(fx):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    video_path = 'input/2025-02-04_14-15-17_2.mp4'
    weights_path = '../../processing/models/YOLOv4.pth'

    yolov4 = YOLOv4(weights_path, device)

    if fx == 'whole_image':
        whole_image(video_path)
    elif fx == 'cropped_detections':
        cropped_detections(video_path, yolov4)


def whole_image(video):
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

    cap.release()
    cv2.destroyAllWindows()

    print(f'Method: Feed whole image to DeepFace.find()\n')
    print(f'Total ID time: {id_total}\n')

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
                
                if df_results:
                    merged_df = pd.concat(df_results, ignore_index=True)
                    best_row = merged_df.loc[merged_df['distance'].idxmin()]

                    predicted_identity = best_row['identity']
                    cosine_distance = best_row['distance']
                else:
                    continue

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

    print(f'Method: Feed cropped images to DeepFace.find()\n')
    print(f'Total detection time: {detect_total}')
    print(f'Total ID time: {id_total}\n')
    print(f'YOLOv4 input size: {yolov4.resize_dims}')

    face_df = pd.DataFrame(face_data)
    face_df['correct_id'] = ''
    output_path = os.path.join('output', 'face_data.csv')
    face_df.to_csv(output_path, index=False)
