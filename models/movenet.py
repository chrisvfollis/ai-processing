import numpy as np
import tensorflow as tf
import cv2
import time


class MoveNet:
    def __init__(self, model_dir):
        model = tf.saved_model.load(model_dir)
        self.model = model.signatures['serving_default']

        self.detection_time = 0
    
    def detect(self, img, conf_thresh=0.35, max_only=False):
        def _preprocess_img(img):
            self.conf_thresh = conf_thresh
            original_dims = img.shape[:2][::-1]
            w, h = original_dims
            
            scale = 1
            if min(original_dims) < 96:
                scale = 96 / min(original_dims)
                w, h = [int(round(x * scale)) for x in original_dims]
                img = cv2.resize(img, (w, h))
            
            target_w, target_h = [int((x // 32) * 32) for x in [w, h]]

            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0),
                                           target_h, target_w)
            img = tf.cast(img, dtype=tf.int32)

            pad_scale = min([target_w / w, target_h / h])
            pad_w = w - (w * pad_scale)
            pad_h = h - (h * pad_scale)

            mapping = {'scale': scale, 'pad_scale': pad_scale, 'pad_dims':
                       [pad_w, pad_h], 'target_dims': [target_w, target_h]}

            return img, mapping
        
        def _postprocess_output(output, mapping, max_only):
            def _map_keypoints(keypoints, mapping):
                scale = mapping['scale']
                pad_scale = mapping['pad_scale']
                pad_w, pad_h = mapping['pad_dims']
                target_w, target_h = mapping['target_dims']

                keypoints[:, 0] = (keypoints[:, 0] * target_w) - (pad_w / 2)
                keypoints[:, 1] = (keypoints[:, 1] * target_h) - (pad_h / 2)

                keypoints[:, :2] = (
                    np.rint(keypoints[:, :2] / (scale * pad_scale)).astype(int)
                )

                return keypoints

            detection_array = (
                output['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
            )
            detections = detection_array[~np.all(detection_array == 0,
                                                 axis=(1, 2))]
            filtered_detections = np.where(
                (detections[:, :, 2] > self.conf_thresh)[..., None], detections, 0
            )
            valid_detections = filtered_detections[
                ~np.all(filtered_detections[:, :, 2] == 0, axis=1)
            ]

            if max_only:
                if valid_detections.size > 0:
                    confidence_sums = valid_detections[:, :, 2].sum(axis=1)
                    max_index = np.argmax(confidence_sums)
                    return _map_keypoints(valid_detections[max_index], mapping)
                else:
                    return np.zeros((17, 3))
            else:
                if valid_detections.size > 0:
                    valid_detections = np.array([
                        _map_keypoints(x, mapping) for x in valid_detections
                    ])
                    return valid_detections
                else:
                    return np.zeros((6, 17, 3))

        start_detect = time.perf_counter()

        self.conf_thresh = conf_thresh

        img, mapping = _preprocess_img(img)
        raw_output = self.model(img)

        final_output = _postprocess_output(raw_output, mapping, max_only)

        end_detect = time.perf_counter()
        self.detection_time += (end_detect - start_detect)

        return final_output

    def detection_batch(self, img, bboxes):
        all_keypoints = []

        for box in bboxes:
            x, y, w, h, = box[:4]
            img_cropped = img[y:y+h, x:x+w]
            try:
                keypoints = self.detect(img_cropped, max_only=True)
                assert keypoints.shape == (17, 3)
                if not np.all(keypoints == 0):
                    keypoints[:, 0] += x
                    keypoints[:, 1] += y
            
            except AssertionError:
                print('Keypoint detection returned an invalid shape: ' +
                      f'{keypoints.shape}. Expected (17, 3) ')
    
            except Exception as e:
                print(f"Error processing bounding box: {e}")
                keypoints = np.zeros((17, 3))
            
            all_keypoints.append(keypoints)

        return all_keypoints
