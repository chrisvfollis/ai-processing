import numpy as np
import tensorflow as tf
import cv2
import time


class MoveNet:
    def __init__(self, model_dir, conf_thresh=0.35):
        model = tf.saved_model.load(model_dir)
        self.model = model.signatures['serving_default']

        self.conf_thresh = conf_thresh

        self.preprocess_time = 0
        self.detection_time = 0
        self.postprocess_time = 0
    
    def detect(self, img, conf_thresh=None, max_only=False):
        def _preprocess_img(img):
            start_preprocess = time.perf_counter()

            original_dims = img.shape[:2][::-1]

            min_scale = max(1, 96 / min(original_dims))
            min_scale_dims = [round(d * min_scale) for d in original_dims]

            target_dims = [int((d // 32) * 32) for d in min_scale_dims]
            
            img = tf.image.resize_with_pad(
                tf.expand_dims(img, axis=0), *target_dims[::-1]
            )

            img = tf.cast(img, dtype=tf.int32)
            mapping = [original_dims, min_scale, min_scale_dims, target_dims]

            end_preprocess = time.perf_counter()
            self.preprocess_time += (end_preprocess - start_preprocess)

            return img, mapping
        
        def _postprocess_output(output, mapping, conf_thresh, max_only):
            start_postprocess = time.perf_counter()

            detection_array = (output['output_0'].numpy()[:, :, :51]
                               .reshape((6, 17, 3)))
            filtered = detection_array[
                ~np.all(detection_array[:, :, 2] <= conf_thresh, axis=1)
            ]

            if max_only:
                if filtered.size > 0:
                    confidence_sums = filtered[:, :, 2].sum(axis=1)
                    max_index = np.argmax(confidence_sums)
                    final_output = self.map_keypoints(filtered[max_index], mapping)
                else:
                    final_output = np.zeros((17, 3))

            else:
                if filtered.size > 0:
                    final_output = np.array(
                        [self.map_keypoints(x, mapping) for x in filtered]
                    )
                else:
                    final_output = np.zeros((6, 17, 3))
            
            end_postprocess = time.perf_counter()
            self.postprocess_time += (end_postprocess - start_postprocess)

            return final_output
        
        conf_thresh = conf_thresh if conf_thresh else self.conf_thresh

        img, mapping = _preprocess_img(img)

        start_detect = time.perf_counter()
        raw_output = self.model(img)
    
        end_detect = time.perf_counter()
        self.detection_time += (end_detect - start_detect)

        results = _postprocess_output(raw_output, mapping, conf_thresh, max_only)

        return results

    def detection_batch(self, img, bboxes):
        def _preprocess(img, bboxes):
            start_preprocess = time.perf_counter()

            batch_images = []
            mappings = []

            original_dims = img.shape[:2][::-1]

            for box in bboxes:
                x, y, w, h = box[:4]
                img_cropped = img[y:y+h, x:x+w]

                if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
                    batch_images.append(None)
                    mappings.append(None)
                    continue

                min_scale = max(1, 96 / min(original_dims))
                min_scale_dims = [round(d * min_scale) for d in original_dims]
                target_dims = [int((d // 32) * 32) for d in min_scale_dims]

                img_resized = tf.image.resize_with_pad(
                    tf.expand_dims(img_cropped, axis=0), *target_dims[::-1]
                )

                batch_images.append(tf.cast(img_resized, dtype=tf.int32))
                mappings.append(
                    [original_dims, min_scale, min_scale_dims,target_dims]
                )
            
            if not any(batch_images):
                return [np.zeros((17, 3)) for _ in bboxes], None
            
            batch_tensor = tf.concat([img for img in batch_images if img is not None], axis=0)

            end_preprocess = time.perf_counter()
            self.preprocess_time += (end_preprocess - start_preprocess)

            return batch_tensor, mappings
        
        def _postprocess(output, mappings, bboxes):
            start_postprocess = time.perf_counter()

            detection_array = (output['output_0'].numpy()[:, :, 51]
                               .reshape((-1, 6, 17, 3)))
            
            all_keypoints = []
            for i, (bbox, mapping) in enumerate(zip(bboxes, mappings)):
                if mapping is None:
                    all_keypoints.append(np.zeros((17, 3)))
                    continue

                x, y, _, _ = bbox
                filtered = detection_array[i][~np.all(detection_array[i][:, :, 2] <= self.conf_thresh, axis=1)]

                if filtered.size > 0:
                    confidence_sums = filtered[:, :, 2].sum(axis=1)
                    max_index = np.argmax(confidence_sums)
                    keypoints = self.map_keypoints(filtered[max_index], mapping)

                    keypoints[:, 0] += x
                    keypoints[:, 1] += y
                else:
                    keypoints = np.zeros((17, 3))

                all_keypoints.append(keypoints)

            end_postprocess = time.perf_counter()
            self.postprocess_time += (end_postprocess - start_postprocess)

            return all_keypoints

        batch_tensor, mappings = _preprocess(img, bboxes)
        if mappings is None:
            return batch_tensor
        
        start_detect = time.perf_counter()
        raw_output = self.model(batch_tensor)
        end_detect = time.perf_counter()
        self.detection_time += (end_detect - start_detect)

        return _postprocess(raw_output, mappings, bboxes)

    def map_keypoints(self, keypoints, mapping):
        original_dims, min_scale, ms_dims, t_dims = mapping
        axis_boundaries = [[0, d - 1] for d in original_dims]

        rsz_scale = min([t_dims[i] / ms_dims[i] for i in range(2)])

        pad_vals = [
            max(0, ((t_dims[i] - round(ms_dims[i] * rsz_scale)) / 2))
            for i in range(2)
        ]

        keypoints[:, 0] = (keypoints[:, 0] * t_dims[0]) - pad_vals[0]
        keypoints[:, 1] = (keypoints[:, 1] * t_dims[1]) - pad_vals[1]

        keypoints[:, 0] = np.clip(
            np.rint(keypoints[:, 0] / (rsz_scale * min_scale)).astype(int),
            *axis_boundaries[0]
        )
        keypoints[:, 1] = np.clip(
            np.rint(keypoints[:, 1] / (rsz_scale * min_scale)).astype(int),
            *axis_boundaries[1]
        )

        return keypoints
