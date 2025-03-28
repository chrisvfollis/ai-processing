# standard dependencies
from typing import List, Union, Sequence
from collections.abc import Iterable
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch
from deepface.models.Detector import FacialAreaRegion

# internal dependencies
from utilities import utilities as utils


class CenterFace:
    def __init__(
            self,
            device: torch.device = None,
            weights_path: str ='../models/weights/centerface.pth',
            conf_thresh: float = 0.65,
            min_area: Union[Iterable[int], int] = (40, 40),
            ignore_landmarks: bool = False,
            save_data: bool = False
        ):

        '''Inspired by https://github.com/Star-Clouds/CenterFace/'''

        self.device = device or torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        
        self.model = torch.load(weights_path, map_location=self.device)

        self.model.eval()
        self.model.to(self.device)
        
        self.conf_thresh = conf_thresh
        self.min_area = min_area

        self.ignore_landmarks = ignore_landmarks

        self.img_h_new, self.img_w_new = 0, 0
        self.scale_h, self.scale_w = 0, 0

        self.heatmaps = []

        self.save_data = save_data
        if self.save_data:
            self.i = 0
            self.face_detections = {}

    def detect_faces(
            self,
            img: np.ndarray,
            region: Sequence = None,
            conf_thresh: float = None,
            min_area: Union[Iterable[int], int] = None
        ) -> List[FacialAreaRegion]:

        def _inference_pytorch(img, conf_thresh, min_area):
            image_cv = cv2.resize(img, dsize=(self.img_w_new, self.img_h_new))
            blob = (
                cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                .transpose(2, 0, 1)
                .astype('float32')
            )
            tensor = torch.from_numpy(blob).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)

            heatmap, scale, offset, lms = [
                output.cpu().numpy() for output in outputs
            ]
            self.heatmaps.append(heatmap)

            return _postprocess(heatmap, lms, offset, scale, conf_thresh, min_area)

        def _postprocess(heatmap, lms, offset, scale, conf_thresh, min_area):
            if not self.ignore_landmarks:
                dets, lms = _decode(heatmap, scale, offset, lms,
                                    (self.img_h_new, self.img_w_new),
                                    conf_thresh, min_area)
            else:
                dets = _decode(heatmap, scale, offset, None,
                               (self.img_h_new, self.img_w_new),
                               conf_thresh, min_area)

            if len(dets) > 0:
                dets[:, 0:4:2] /= self.scale_w
                dets[:, 1:4:2] /= self.scale_h
                if not self.ignore_landmarks:
                    lms[:, 0:10:2] /= self.scale_w
                    lms[:, 1:10:2] /= self.scale_h
            else:
                dets = np.empty(shape=[0, 5], dtype=np.float32)
                if not self.ignore_landmarks:
                    lms = np.empty(shape=[0, 10], dtype=np.float32)

            return dets if self.ignore_landmarks else (dets, lms)

        def _decode(heatmap, scale, offset, landmark, size, conf_thresh, min_area):
            def _translate_dims(i, scale0, scale1, y_idx, x_idx):
                '''
                Converts downsampled log-space model output to normal pixel
                dimensions.
                '''
                log_h = scale0[y_idx, x_idx]    # predicted face height
                log_w = scale1[y_idx, x_idx]    # predicted face width

                h = np.exp(log_h)   # exponentiate to reverse the logarithm
                w = np.exp(log_w)

                h *= 4  # multiply by 4 to account for downsampling (stride)
                w *= 4 

                return h, w
            
            def _get_xyxy(i, offset0, offset1, y_idx, x_idx, h, w):   
                o0 = offset0[y_idx, x_idx]  # predicted sub-cell offsets
                o1 = offset1[y_idx, x_idx]

                x_cntr = x_idx + 0.5    # center position in cell
                y_cntr = y_idx + 0.5

                x_cntr += o1    # apply predicted offsets
                y_cntr += o0

                x_cntr *= 4    # multiply by 4 to account for downsampling (stride)
                y_cntr *= 4

                x1 = max(0, (x_cntr - w / 2))
                y1 = max(0, (y_cntr - h / 2))
                x2 = min(size[1], (x1 + w))
                y2 = min(size[0], (y1 + h))

                return x1, y1, x2, y2

            heatmap = np.squeeze(heatmap)

            c0, c1 = np.where(heatmap > conf_thresh)

            offset0 = offset[0, 0, :, :]    # detection offset within 4x4 grid cell
            offset1 = offset[0, 1, :, :]

            scale0 = scale[0, 0, :, :]  # log(height) predictions
            scale1 =  scale[0, 1, :, :] # log(width) predictions
            
            if not self.ignore_landmarks:
                boxes, lms = [], []
            else:
                boxes = []

            if len(c0) > 0:
                for i in range(len(c0)):
                    y_idx, x_idx = c0[i], c1[i]   # grid cell indices

                    h, w = _translate_dims(i, scale0, scale1, y_idx, x_idx)
                    if math.prod((h, w)) < min_area:
                        continue    # filter (skip) small detection

                    x1, y1, x2, y2 = _get_xyxy(
                        i, offset0, offset1, y_idx, x_idx, h, w
                    )
                    s = heatmap[y_idx, x_idx]

                    if not self.ignore_landmarks:
                        lm = []
                        for j in range(5):
                            lm_x = landmark[0, (j * 2), y_idx, x_idx]
                            lm_y = landmark[0, (j * 2 + 1), y_idx, x_idx]

                            lm_x = lm_x * w + x1
                            lm_y = lm_y * h + y1

                            lm.append(lm_x)
                            lm.append(lm_y)
                        lms.append(lm)
                    
                    boxes.append([x1, y1, x2, y2, s])

                boxes = np.asarray(boxes, dtype=np.float32)
                if boxes.size != 0:
                    keep = _nms(boxes[:, :4], boxes[:, 4], 0.3)
                    boxes = boxes[keep, :]

                    if not self.ignore_landmarks:
                        lms = np.asarray(lms, dtype=np.float32)
                        lms = lms[keep, :]
                else:
                    boxes = boxes.reshape((0, 5))
                    
            print(f'{len(c0) - len(boxes)} small face detections filtered')

            if not self.ignore_landmarks:
                return boxes, lms
            else:
                return boxes

        def _nms(boxes, scores, nms_thresh):
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
    
                ix1, iy1 = x1[i], y1[i]
                ix2, iy2 = x2[i], y2[i]

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

        conf_thresh = conf_thresh or self.conf_thresh
        min_area = min_area or self.min_area

        if isinstance(min_area, Iterable):
            min_area = math.prod(min_area)

        h, w = img.shape[:2]
        if math.prod((h, w)) < min_area:
            return []

        if (h >= 32) and (w >= 32):
            self.img_h_new = (h // 32) * 32
            self.img_w_new = (w // 32) * 32
        else:
            self.img_w_new = w + (w % 2)
            self.img_h_new = h + (h % 2)

        self.scale_h = h / self.img_h_new
        self.scale_w = w / self.img_w_new

        detections = _inference_pytorch(img, conf_thresh, min_area)

        if not self.ignore_landmarks:
            all_dets, all_lms = detections
        else:
            all_dets = detections
            lms = None

        detected_faces = []
        for i, box in enumerate(all_dets):

            x1, y1, x2, y2 = map(int, box[:4])

            w = x2 - x1
            h = y2 - y1

            score = float(box[4])

            if region:
                x1, y1 = utils.apply_offset((x1, y1), region)
                x2, y2 = utils.apply_offset((x2, y2), region)
            
            if all_lms is not None:
                lms = [
                    tuple(
                        map(int, all_lms[i][j:j+2])
                    )
                    for j in range(0, 9, 2)
                ]
                if region:
                    lms = utils.apply_offset(lms, region)

                left_eye, right_eye, nose, mouth_right, mouth_left = lms
            else:
                left_eye = right_eye = nose = mouth_right = mouth_left = None

            face_region = FacialAreaRegion(
                x=x1,
                y=y1,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                nose=nose,
                mouth_right=mouth_right,
                mouth_left=mouth_left,
                confidence=score
            )
            detected_faces.append(face_region)

        if self.save_data:
            self.face_detections.setdefault(self.i, []).extend(detected_faces)

        return detected_faces

    def visualize_detections(
            self, image: np.ndarray, face_detections: List[FacialAreaRegion],
            output_path: str = None
        ):
        '''
        Visualizes detected faces and their landmarks on the input image.

        Args:
            img (np.ndarray): The original input image.
            detected_faces (List[FacialAreaRegion]): The detected face regions with landmarks.
        '''
        def _bgr_color_tuples():
            blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)
            yellow, white = (0, 255, 255), (255, 255, 255)
            return blue, green, red, yellow, white

        image = image.copy()
        blue, green, red, yellow, white = _bgr_color_tuples()

        for face in face_detections:
            x1, y1, x2, y2 = utils.xywh_xyxy([face.x, face.y, face.w, face.h])
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

    def save_runtime_data(self, filename='../files/output/centerface_data.xlsx'):
        if not self.save_data:
            return
        
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
