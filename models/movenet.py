import numpy as np
import tensorflow as tf
import cv2
import time


class MoveNet:
    def __init__(self, model_dir, conf_thresh=0.35):
        model = tf.saved_model.load(model_dir)
        self.model = model.signatures['serving_default']

        self.detection_time = 0
        self.conf_thresh = conf_thresh
    
    def detect(self, img, conf_thresh=0.35, max_only=False):
        def _preprocess_img(img):
            original_dims = img.shape[:2][::-1]

            min_scale = max(1, 96 / min(original_dims))
            min_scale_dims = [round(d * min_scale) for d in original_dims]

            target_dims = [int((d // 32) * 32) for d in min_scale_dims]
            
            img = tf.image.resize_with_pad(
                tf.expand_dims(img, axis=0), *target_dims[::-1]
            )

            img = tf.cast(img, dtype=tf.int32)
            mapping = [original_dims, min_scale, min_scale_dims, target_dims]

            return img, mapping
        
        def _postprocess_output(output, mapping, max_only):
            def _map_keypoints(kpts, mapping):
                original_dims, min_scale, ms_dims, t_dims = mapping
                axis_boundaries = [[0, d - 1] for d in original_dims]

                rsz_scale = min([t_dims[i] / ms_dims[i] for i in range(2)])

                pad_vals = [
                    max(0, ((t_dims[i] - round(ms_dims[i] * rsz_scale)) / 2))
                    for i in range(2)
                ]

                kpts[:, 0] = (kpts[:, 0] * t_dims[0]) - pad_vals[0]
                kpts[:, 1] = (kpts[:, 1] * t_dims[1]) - pad_vals[1]

                kpts[:, 0] = np.clip(
                    np.rint(kpts[:, 0] / (rsz_scale * min_scale)).astype(int),
                    *axis_boundaries[0]
                )
                kpts[:, 1] = np.clip(
                    np.rint(kpts[:, 1] / (rsz_scale * min_scale)).astype(int),
                    *axis_boundaries[1]
                )

                return kpts

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
