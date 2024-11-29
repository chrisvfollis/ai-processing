import numpy as np
import os
import torch
import cv2
import io_utils
import sys
import datetime
import warnings
import torchreid
import h5py
from deepface import DeepFace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'YOLOv4'))
from models import Yolov4
from YOLOv4.tool.torch_utils import do_detect


def facial_recognition(frame, boxes, max_distance=0.75,
                       min_area=900):
    def _world_model_filters(output, box, fraction=1.5):
        '''
        This function incorporates innate human knowledge about the world
        to help filter out unlikely output.

        For instance, you would generally expect to find someone's face
        towards the top of their bounding box. A face detection located near
        the bottom of the box is probably from another person crossing
        between them and the camera.
        '''

        output = output.loc[(output["source_y"] + box[1] + (output["source_h"] / 2))
                            <= (box[1] + (box[3] / fraction))]

        return output
    
    id_data = []
    for box in boxes:
        if (box[2] * box[3]) < min_area:
            id_data.append({'identity': None, 'distance': None})
            continue
        x1, y1 = box[0], box[1]
        x2, y2 = box[0] + box[2], box[1] + box[3]
        cropped = frame[y1:y2, x1:x2]
        try:
            dfs = DeepFace.find(
                img_path = cropped, db_path = '../input_files/faces',
                model_name = 'Facenet512', threshold = max_distance,
                detector_backend = 'retinaface', enforce_detection = True,
                silent = True
            )
            # if len(dfs[0]['identity']) > 0:
            #     print('Face detected')
            # dfs[0] = _world_model_filters(dfs[0], box)
            if dfs[0].empty:
                id_data.append({'identity': None, 'distance': None})
                continue
            else:
                min_index = dfs[0]['distance'].idxmin()
                min_row = dfs[0].loc[min_index]

                img_match = min_row['identity'].split('/')[-1]
                print(img_match)
                identity = io_utils.get_employee(img_match)
                # identity = img_match
                distance = min_row['distance']

                id_data.append({'identity': identity, 'distance': distance})
        except (KeyError, ValueError):
            id_data.append({'identity': None, 'distance': None})

    return id_data


def load_extractor(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['state_dict']

    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]

    model = torchreid.models.osnet.osnet_x1_0(num_classes=751, pretrained=False, loss='triplet')
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def load_yolov4(weights_path, device):
    model = Yolov4(inference=True)
    weights = torch.load(weights_path, map_location=device)

    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    return model


def detect_yolov4(img, class_num, model, device, conf_thresh=0.65,
                  nms_thresh=0.5):

    orig_h, orig_w = img.shape[:2]

    img_resized = cv2.resize(img, (416, 416))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        detections = do_detect(model, img_rgb, conf_thresh=conf_thresh,
                               nms_thresh=nms_thresh,
                               use_cuda=(device.type == 'cuda'))[0]

    scale_x = orig_w / 416
    scale_y = orig_h / 416

    filtered_detections = []

    for det in detections:
        x1, y1, x2, y2, confidence, class_score, class_id = det

        if int(class_id) == class_num:
            x1 = int(x1 * 416)
            y1 = int(y1 * 416)
            x2 = int(x2 * 416)
            y2 = int(y2 * 416)

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            filtered_detections.append([x, y, w, h, float(confidence)])

    return filtered_detections


def inference_pipeline(video_file, detector, stride=1, start=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_path = '../input_files/'
    cap = cv2.VideoCapture(base_path + video_file)
    out = cv2.VideoWriter(f'../output_files/{video_file.split(".")[0]}'
                          + '_boxes.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                          1, (1920, 1080))

    frame_data = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    f_num = start
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failure to read from file')
            break

        if (f_num - start) % stride == 0:
            detections = detect_yolov4(frame, 0, detector, device)
            if len(detections) > 0:
                frame_data.setdefault(f_num, [])
                id_data = facial_recognition(frame, detections)

                for i, record in enumerate(id_data):
                    record['detection'] = detections[i]
                    frame_data[f_num].append(record)

                    box = detections[i][:2] + [sum(detections[i][0:3:2]),
                                                sum(detections[i][1:4:2])]
                    cv2.rectangle(frame, box[:2], box[2:], (245, 104, 17))
                    if record['identity']:
                        cv2.putText(frame, record['identity'],
                                    (box[0]-5, box[1]-5), cv2.FONT_HERSHEY_PLAIN,
                                    3, (245, 104, 17))

                out.write(frame)

        if stride <= 15:
            f_num += 1
        else:
            f_num += stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)

    cap.release()
    out.release()

    return frame_data
