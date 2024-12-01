import numpy as np
import os
import torch
import cv2
import io_utils
import sys
import datetime
import torchreid
import h5py
from deepface import DeepFace


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'YOLOv4'))
from models import Yolov4
from YOLOv4.tool.torch_utils import do_detect


def facial_recognition(frame):
    try:
        faces = DeepFace.find(
            img_path=frame, db_path='../input_files/faces',
            model_name='Facenet512', detector_backend='retinaface',
            enforce_detection=True, silent=True
        )        
    except ValueError:
        print('No faces detected')
        return None

    filtered_faces = []
    for df in faces:
        filtered = df.sort_values(by='distance')[:3]
        filtered['identity'] = (
            filtered['identity'].map(lambda x: io_utils.get_employee(x))
            )
        filtered = filtered.loc[filtered.groupby('identity')
                                ['distance'].idxmin()]
        filtered_faces.append(filtered)

    return filtered_faces


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


def inference_pipeline(video_file, detector, extractor, track_stride=1,
                       id_stride=30, start=0, batch_size=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_path = '../input_files/'
    cap = cv2.VideoCapture(base_path + video_file)

    if (id_stride % track_stride) != 0:
        id_stride = id_stride - (id_stride % track_stride)

    person_data = {}
    face_data = {}

    embeddings = []
    frames_batch = []
    box_indices_batch = []

    hdf5_file = h5py.File(f'../intermediate_output/{video_file.split(".")[0]}_embeddings.hdf5', 'a')
    hdf5_file.create_dataset('embeddings', (0, 512), maxshape=(None, 512))
    hdf5_file.create_dataset('frames', (0,), maxshape=(None,), dtype='i')
    hdf5_file.create_dataset('box_indices', (0,), maxshape=(None,), dtype='i')

    f_num = start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (f_num - start) % track_stride == 0:
            det_xywhc = detect_yolov4(frame, 0, detector, device)
            if len(det_xywhc) > 0:
                person_data[f_num] = det_xywhc

                for i, box in enumerate(det_xywhc):
                    x, y, w, h, _ = box
                    cropped = frame[y:y+h, x:x+w]

                    try:
                        image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (128, 256))
                        image = image.astype(np.float32)
                        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                        image = image.to(device)

                        embedding = extractor(image).cpu().detach().numpy().flatten()
                        embeddings.append(embedding)
                        frames_batch.append(f_num)
                        box_indices_batch.append(i)
                    except Exception:
                        embeddings.append([])
                        print(f"Error processing bounding box at {x},{y},{w},{h}")
        if len(embeddings) >= batch_size:
            io_utils.write_embeddings(hdf5_file, embeddings, frames_batch, box_indices_batch)
            embeddings.clear()
            frames_batch.clear()
            box_indices_batch.clear()
        
        if (f_num - start) % id_stride == 0:
            faces = facial_recognition(frame)
            if faces:
                face_data[f_num] = faces

        if track_stride <= 15:
            f_num += 1
        else:
            f_num += track_stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)

    if len(embeddings) > 0:
        io_utils.write_embeddings(hdf5_file, embeddings, frames_batch, box_indices_batch)

    cap.release()
    hdf5_file.close()

    return person_data, face_data