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
from models.yolov4_architecture import Yolov4Model
import tensorflow as tf


class OSNet:
    def __init__(self, weights_path, device, input_shape=(128, 256),
                 output_shape=(512,), num_classes=751, loss='triplet'):
        self.device = device

        self.model = torchreid.models.osnet.osnet_x1_0(
            num_classes=num_classes, pretrained=False, loss='triplet'
        )
        self.input_shape = input_shape
        self.output_shape = output_shape

        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = state_dict[key]
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, img):
        def _preprocess_img(img):
            w, h = self.input_shape

            img_resized = cv2.resize(img, (w, h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32)

            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            return img_tensor
        
        def _postprocess_output(output):
            return output.cpu().detach().numpy().flatten()
        
        img = _preprocess_img(img)
        with torch.no_grad():
            output = self.model(img)

        embedding = _postprocess_output(output)

        return embedding

    def enable_buffers(self, output_path, buffer_limit=100):
        '''
        Sets up buffers and an output file so the OSNet instance can use
        bulk processing features like extraction batches.
        '''

        self.buffer_limit = buffer_limit

        self.embedding_buffer = []
        self.frame_buffer = []
        self.box_index_buffer = []

        self.hdf5_file = h5py.File(output_path, 'a')

        n_features = self.output_shape[0]
        idx_dataset_kwargs = {'shape': (0,), 'dtype': 'i', 'maxshape': (None,)}

        self.hdf5_file.create_dataset(
            'embeddings', (0, n_features), maxshape=(None, n_features)
        )
        self.hdf5_file.create_dataset('frames', **idx_dataset_kwargs)
        self.hdf5_file.create_dataset('box_indices', **idx_dataset_kwargs)

    def flush_buffers(self, close_file=False):
        io_utils.write_embeddings(
            self.hdf5_file, self.embedding_buffer, self.frame_buffer,
            self.box_index_buffer
        )
        self.embedding_buffer.clear()
        self.frame_buffer.clear()
        self.box_index_buffer.clear()

        if close_file == True:
            self.hdf5_file.close()

    def extraction_batch(self, img, detections, f_num):
        def _update_buffers(embedding, f_num, box_idx):
            self.embedding_buffer.append(embedding)
            self.frame_buffer.append(f_num)
            self.box_index_buffer.append(box_idx)

        for i, box in enumerate(detections):
            x, y, w, h, = box[:4]
            img_cropped = img[y:y+h, x:x+w]
            try:
                embedding = self.extract_features(img_cropped)
            except Exception as e:
                print(f"Error processing bounding box: {e}")
                embedding = []

            _update_buffers(embedding, f_num, i)
        
        if self.buffer_limit <= len(self.embedding_buffer):
            self.flush_buffers()


class YOLOv4:
    def __init__(self, weights_path, device, nms_thresh=0.5):
        self.device = device

        self.model = Yolov4Model(inference=True)
        weights = torch.load(weights_path, map_location=device)

        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()

        self.nms_thresh = nms_thresh
        
    def detect(self, img, class_num, conf_thresh=0.65, resize_dims=(416, 416)):
        def _preprocess_img(img, resize_dims):
            '''
            resize_dims — the width and height to resize the image to. YOLOv4
            only accepts image dimensions that can be expressed using the
            formula (320 + (96 * n)), where n is a positive integer.
        
            Examples of valid dimensions include 320, 416, 512, 608, etc
            '''
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

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            x1 = int(max(0, min(x1, img_w)))
            y1 = int(max(0, min(y1, img_h)))
            x2 = int(max(0, min(x2, img_w)))
            y2 = int(max(0, min(y2, img_h)))

            w = x2 - x1
            h = y2 - y1

            return [x1, y1, w, h]

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

        return filtered


class InferencePipeline:
    def __init__(self, video_file, weights_paths, hdf5_path, device,
                 track_stride=1, id_stride=30, buffer_limit=100):
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
                )
                print("TensorFlow GPU memory configured.")
            except RuntimeError as e:
                print(f"Error configuring TensorFlow GPU memory: {e}")

        self.yolov4 = YOLOv4(weights_paths[0], device, nms_thresh=0.5)
        self.osnet = OSNet(weights_paths[1], device)
        
        self.osnet.enable_buffers(hdf5_path, buffer_limit=buffer_limit)

        self.person_data = {}
        self.face_data = {}

        self.video_file = video_file
        self.cap = cv2.VideoCapture('../input_files/' + video_file)
        self.f_num = 0

        self.track_stride = track_stride
        self.id_stride = id_stride
        
        if (self.id_stride % self.track_stride) != 0:
            self.id_stride = id_stride - (id_stride % track_stride)

    def detection_skim(self, stride=120):
        print(f'skimming {self.video_file}...')

        prev_frame = -1
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.f_num < total_frames:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if (not ret) or (current_frame == prev_frame):
                return None
            prev_frame = current_frame

            if self.f_num % stride == 0:
                detections = self.yolov4.detect(frame, 0, conf_thresh=0.78)
                if detections:
                    self.f_num = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)
                    return True

            self.f_num += stride
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)
    
    def identify_faces(self, frame):
        try:
            face_dfs = DeepFace.find(
                img_path=frame, db_path='../input_files/faces',
                model_name='Facenet512', detector_backend='retinaface',
                threshold = 0.8, enforce_detection=True, silent=True
            )        
        except ValueError:
            return None

        filtered_dfs= []
        for df in face_dfs:
            if not df.empty:
                print('Possible identity match(es) found')
            df['identity'] = (
                df['identity'].map(lambda x: io_utils.get_employee(x))
            )
            df = df.loc[df.groupby('identity')['distance'].idxmin()]
            filtered_dfs.append(df)
        
        if filtered_dfs:
            self.face_data[self.f_num] = filtered_dfs

    def run(self):
        def _process_frame(frame):
            detections = None
            if self.f_num % self.track_stride == 0:
                detections = self.yolov4.detect(frame, 0, conf_thresh=0.65,
                                                resize_dims=(416, 416))
                self.osnet.extraction_batch(frame, detections, self.f_num)

                if detections:
                    self.person_data[self.f_num] = detections

            if self.f_num % self.id_stride == 0:
                self.identify_faces(frame)
        
        def _continue():
            if self.f_num % 600 == 0:
                print(self.f_num)
    
            if self.track_stride <= 15:
                self.f_num += 1
            else:
                self.f_num += self.track_stride
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)
        
        def _save_and_cleanup():
            io_utils.write_detection_csv(self.person_data, self.video_file)
            io_utils.write_face_csv(self.face_data, self.video_file)

            if len(self.osnet.embedding_buffer) > 0:
                self.osnet.flush_buffers(close_file=True)

            self.cap.release()

        preliminary_detections = self.detection_skim()
        if not preliminary_detections:
            return False

        self.f_num = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)

        prev_frame = -1
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.f_num < total_frames:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if (not ret) or (current_frame == prev_frame):
                break
            prev_frame = current_frame

            _process_frame(frame)
            _continue()

        _save_and_cleanup()

        return True
