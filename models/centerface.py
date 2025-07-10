# standard dependencies
from typing import Sequence
from collections.abc import Iterable
import math
import os

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch

# internal dependencies
from modules.identification.data_structures import FacialAreaRegion
from modules.spatial import bboxes
from utilities import utils, io_utils, log_utils


logger = log_utils.get_logger(__name__)


class CenterFace:
    def __init__(
            self,
            device: torch.device = None,
            checkpoint: str = 'centerface.pth',
            conf_thresh: float = 0.50,
            min_area: tuple[int] | int = (32, 32),
            ignore_landmarks: bool = False,
            expand_margin: float = 0.25,
            save_data: bool = False,
        ):
        
        self.device = device or utils.get_default_device()

        self.project_root = io_utils.get_project_root()
        self.output_dir = os.path.join(self.project_root, 'files/output/')

        weights_path = os.path.join(
            self.project_root, 'models/weights/', checkpoint
        )
        self.model = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )

        self.model.eval()
        self.model.to(self.device)
        
        self.conf_thresh = conf_thresh
        self.min_area = min_area

        self.ignore_landmarks = ignore_landmarks

        self.expand_margin = expand_margin

        self.save_data = save_data
        if self.save_data:
            self.i = 0
            self.face_detections = {}

    def detect_faces(
            self,
            imgs: np.ndarray | list[np.ndarray],
            regions: list[Sequence] = None,
            conf_thresh: float = None,
            min_area: tuple[int] | int = None,
        ) -> list[list[FacialAreaRegion]]:
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        conf_thresh = conf_thresh or self.conf_thresh
        min_area = min_area or self.min_area
        if isinstance(min_area, Iterable):
            min_area = math.prod(min_area)

        all_results = [[] for _ in range(len(imgs))]

        for original_idx, img in enumerate(imgs):
            h, w = img.shape[:2]
            if h * w < min_area:
                continue

            region = regions[original_idx] if regions else None
            heatmaps, scales_out, offsets, landmarks_out, scales_hw = self.inference(img)

            scale_h, scale_w = scales_hw[0]
            heatmap = heatmaps[0:1]
            scale_out = scales_out[0:1]
            offset = offsets[0:1]
            lms_out = landmarks_out[0:1]

            target_size = max(h, w)
            model_input_shape = (
                int(np.ceil(target_size / 32) * 32),
                int(np.ceil(target_size / 32) * 32),
            )

            if not self.ignore_landmarks:
                dets, lms = self.postprocess(
                    heatmap, lms_out, offset, scale_out,
                    model_input_shape, self.ignore_landmarks,
                    conf_thresh, min_area, scale_h, scale_w
                )
            else:
                dets = self.postprocess(
                    heatmap, None, offset, scale_out,
                    model_input_shape, self.ignore_landmarks,
                    conf_thresh, min_area, scale_h, scale_w
                )
                lms = None

            detected_faces = []
            for i, box in enumerate(dets):
                x1, y1, x2, y2 = map(int, box[:4])
                if self.expand_margin:
                    x1, y1, x2, y2 = bboxes.expand_bbox(
                        x1, y1, x2, y2, w, h, margin=self.expand_margin
                    )
                score = float(box[4])
                face_w = x2 - x1
                face_h = y2 - y1

                if region:
                    x1, y1 = bboxes.apply_offset((x1, y1), region)
                    x2, y2 = bboxes.apply_offset((x2, y2), region)

                if lms is not None:
                    lms_points = [
                        tuple(map(int, lms[i][j:j+2]))
                        for j in range(0, 9, 2)
                    ]
                    if region:
                        lms_points = bboxes.apply_offset(lms_points, region)
                    left_eye, right_eye, nose, mouth_right, mouth_left = lms_points
                else:
                    left_eye = right_eye = nose = mouth_right = mouth_left = None

                face_region = FacialAreaRegion(
                    x=x1,
                    y=y1,
                    w=face_w,
                    h=face_h,
                    left_eye=left_eye,
                    right_eye=right_eye,
                    nose=nose,
                    mouth_right=mouth_right,
                    mouth_left=mouth_left,
                    confidence=score,
                )
                detected_faces.append(face_region)

            if self.save_data:
                self.face_detections.setdefault(self.i + original_idx, []).extend(detected_faces)

            all_results[original_idx] = detected_faces

        if self.save_data:
            self.i += len(imgs)

        return all_results

    def inference(self, img_data):
        if isinstance(img_data, np.ndarray):
            img_data = [img_data]

        blobs = []
        scales = []

        for img in img_data:
            h, w = img.shape[:2]
            target_size = max(h, w)
            new_h = new_w = int(np.ceil(target_size / 32) * 32)

            scale_h = new_h / h
            scale_w = new_w / w

            resized = cv2.resize(img, dsize=(new_w, new_h))
            blob = (
                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                .transpose(2, 0, 1)
                .astype('float32')
            )
            blobs.append(torch.from_numpy(blob))
            scales.append((scale_h, scale_w))

        tensor = torch.stack(blobs).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)

        heatmaps, scales_out, offsets, landmarks = [
            output.cpu().numpy() for output in outputs
        ]

        return heatmaps, scales_out, offsets, landmarks, scales
    
    def postprocess(self, heatmap, lms, offset, scale, img_shape, ignore_landmarks, conf_thresh, min_area, scale_h, scale_w):
        if not ignore_landmarks:
            dets, lms_out = self._decode(heatmap, scale, offset, lms, img_shape, ignore_landmarks, conf_thresh, min_area)
        else:
            dets = self._decode(heatmap, scale, offset, None, img_shape, ignore_landmarks, conf_thresh, min_area)

        if len(dets) > 0:
            dets[:, 0:4:2] /= scale_w
            dets[:, 1:4:2] /= scale_h
            if not ignore_landmarks:
                lms_out[:, 0:10:2] /= scale_w
                lms_out[:, 1:10:2] /= scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if not ignore_landmarks:
                lms_out = np.empty(shape=[0, 10], dtype=np.float32)

        return dets if ignore_landmarks else (dets, lms_out)

    def _decode(self, heatmap, scale, offset, landmark, size, ignore_landmarks, conf_thresh, min_area):
        def _translate_dims(scale0, scale1, y_idx, x_idx):
            log_h = scale0[y_idx, x_idx]
            log_w = scale1[y_idx, x_idx]

            h = np.exp(log_h) * 4
            w = np.exp(log_w) * 4

            return h, w

        def _get_xyxy(offset0, offset1, y_idx, x_idx, h, w):
            o0 = offset0[y_idx, x_idx]
            o1 = offset1[y_idx, x_idx]

            x_cntr = (x_idx + 0.5 + o1) * 4
            y_cntr = (y_idx + 0.5 + o0) * 4

            x1 = max(0, x_cntr - w / 2)
            y1 = max(0, y_cntr - h / 2)
            x2 = min(size[1], x1 + w)
            y2 = min(size[0], y1 + h)

            return x1, y1, x2, y2

        heatmap = np.squeeze(heatmap)
        c0, c1 = np.where(heatmap > conf_thresh)

        offset0 = offset[0, 0, :, :]
        offset1 = offset[0, 1, :, :]

        scale0 = scale[0, 0, :, :]
        scale1 = scale[0, 1, :, :]

        boxes = []
        lms_out = [] if not ignore_landmarks else None

        for y_idx, x_idx in zip(c0, c1):
            h, w = _translate_dims(scale0, scale1, y_idx, x_idx)
            if h * w < min_area:
                continue

            x1, y1, x2, y2 = _get_xyxy(offset0, offset1, y_idx, x_idx, h, w)
            s = heatmap[y_idx, x_idx]

            if not ignore_landmarks:
                lm = []
                for j in range(5):
                    lm_x = landmark[0, (j * 2), y_idx, x_idx] * w + x1
                    lm_y = landmark[0, (j * 2 + 1), y_idx, x_idx] * h + y1
                    lm.append(lm_x)
                    lm.append(lm_y)
                lms_out.append(lm)

            boxes.append([x1, y1, x2, y2, s])

        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.size != 0:
            keep = self._nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]

            if not ignore_landmarks:
                lms_out = np.asarray(lms_out, dtype=np.float32)
                lms_out = lms_out[keep, :]
        else:
            boxes = boxes.reshape((0, 5))
            if not ignore_landmarks:
                lms_out = np.empty(shape=[0, 10], dtype=np.float32)

        return (boxes, lms_out) if not ignore_landmarks else boxes
    
    def _nms(self, boxes, scores, nms_thresh):
        keep = []
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=bool)

        order = np.argsort(scores)[::-1]
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1, iy1, ix2, iy2 = x1[i], y1[i], x2[i], y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1, yy1 = max(ix1, x1[j]), max(iy1, y1[j])
                xx2, yy2 = min(ix2, x2[j]), min(iy2, y2[j])
                w, h = max(0, xx2 - xx1), max(0, yy2 - yy1)
                inter = w * h
                iou = inter / (iarea + areas[j] - inter)

                if iou >= nms_thresh:
                    suppressed[j] = True
    
        return keep

    def forward(self, x):
        heatmaps, scales_out, offsets, landmarks = self.model(x)
        return heatmaps

    def visualize_detections(
            self, image: np.ndarray, face_detections: list[FacialAreaRegion],
            output_path: str = None
        ):
        '''
        Visualizes detected faces and their landmarks on the input image.

        Args:
            img (np.ndarray): The original input image.
            detected_faces (list[FacialAreaRegion]): The detected face regions with landmarks.
        '''
        def _bgr_color_tuples():
            blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)
            yellow, white = (0, 255, 255), (255, 255, 255)
            return blue, green, red, yellow, white

        image = image.copy()
        blue, green, red, yellow, white = _bgr_color_tuples()

        for face in face_detections:
            x1, y1, x2, y2 = bboxes.xywh_xyxy([face.x, face.y, face.w, face.h])
            cv2.rectangle(image, (x1, y1), (x2, y2), green, 2)

            cv2.putText(
                image, f'confidence: {face.confidence:.2f}', (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2
            )

            eyes = [face.left_eye, face.right_eye]
            mouth = [face.mouth_left, face.mouth_right]
            nose = face.nose
            
            for point in eyes:
                if point:
                    cv2.circle(image, point, 3, blue, -1)
            for point in mouth:
                if point:
                    cv2.circle(image, point, 3, yellow, -1)
            if nose:
                cv2.circle(image, nose, 3, red, -1)
    
        if output_path:
            cv2.imwrite(output_path, image)

        return image

    def visualize_heatmaps(
            self, image: np.ndarray, heatmaps: list[np.ndarray],
            regions: Sequence = None, alpha: float = 0.35
        ) -> np.ndarray:
        '''
        Overlay the CenterFace heatmap on the original image.

        Args:
            image (np.ndarray): Original BGR image.
            heatmaps (np.ndarray, optional): Heatmap from the last forward pass.
            regions (tuple or list, optional): (x, y, w, h) region in original
                image where the heatmap was generated.
            alpha (float): Blending factor.

        Returns:
            np.ndarray: Image with heatmap overlays.
        '''
        for i, heatmap in enumerate(heatmaps):
            heatmap = np.squeeze(heatmap[0, 0])  # shape: (H, W)

            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_uint8 = heatmap_norm.astype(np.uint8)

            if not regions:
                x = 0
                y = 0
                w = image.shape[1]
                h = image.shape[0]
            else:
                x, y, w, h = regions[i]

            heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

            roi = image[y:y + h, x:x + w]
            blended = cv2.addWeighted(roi, 1 - alpha, heatmap_colored, alpha, 0)

            image[y:y+h, x:x+w] = blended

        return image

    def save_runtime_data(self):
        if not self.save_data:
            return

        filename = os.path.join(self.output_dir, 'centerface_data.xlsx')

        detection_data = []
        for i, detections in self.face_detections.items():
            for det in detections:
                area = math.prod((det.w, det.h))
                detection_data.append({
                    'idx': i,
                    'a': area,
                    'c': det.confidence,
                    'x': det.x,
                    'y': det.y,
                    'w': det.w,
                    'h': det.h,
                })

        detections_df = pd.DataFrame(detection_data)
        
        data = {'idx': [], 'face_detections': []}
        if hasattr(self, 'regions'):
            data['regions'] = []

        for i, detections in self.face_detections.items():
            data['idx'].append(i)
            data['face_detections'].append(len(detections))

            if hasattr(self, 'regions'):
                data['regions'].append(len(self.regions.get(i, [])))
        
        artifact_df = pd.DataFrame(data)


        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            detections_df.to_excel(writer, sheet_name='Detections', index=False)
            artifact_df.to_excel(writer, sheet_name='Pipeline Artifacts', index=False)   
