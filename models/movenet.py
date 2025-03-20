# standard dependencies
import time

# 3rd-party dependencies
import numpy as np
import tensorflow as tf

# internal dependencies
pass


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
        
        conf_thresh = conf_thresh or self.conf_thresh

        img, mapping = _preprocess_img(img)

        start_detect = time.perf_counter()
        raw_output = self.model(img)
    
        end_detect = time.perf_counter()
        self.detection_time += (end_detect - start_detect)

        results = _postprocess_output(raw_output, mapping, conf_thresh, max_only)

        return results

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
