import pandas as pd
import os
import cv2
from utilities import utilities as utils
from models.yolov4 import YOLOv4
from models.osnet import OSNet
from models.movenet import MoveNet
from models.face_iq import FaceIq


class InferencePipeline:
    def __init__(self, video_file, model_info, device, buffer_limit=100):

        self.yolov4 = YOLOv4(model_info[0], device, nms_thresh=0.5)
        self.osnet = OSNet(model_info[1], device)
        self.movenet = MoveNet(model_info[2])
        self.face_iq = FaceIq(*model_info[3])

        self.osnet.enable_buffers(video_file, buffer_limit=buffer_limit)

        self.person_data = {}
        self.face_data = {}
        self.keypoint_data = {}

        self.video_file = video_file
        self.cap = cv2.VideoCapture('../files/input/' + video_file)
        if not self.cap.isOpened():
            raise ValueError("Failed to open video file")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            raise ValueError("FPS returned as 0, check the video file")
        self.resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        self.f_num = 0

        self.track_stride = max(1, self.fps // 10)
        self.id_stride = (self.fps // self.track_stride) * self.track_stride
        self.kp_stride = self.id_stride * 3

        self.checkpoint_stride = (
            ((self.total_frames // 4) // self.track_stride) * self.track_stride
        )

    def skim(self):
        print(f'Skimming...')

        stride = self.fps * 2
        prev_frame = -1

        while self.f_num < self.total_frames:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if (
                (not ret) or
                (current_frame == prev_frame) or
                (utils.is_grayscale(frame, threshold=10))
            ):
                print(f"Nothing to process in {self.video_file}")
                return False

            prev_frame = current_frame

            if self.f_num % stride == 0:
                detections = self.yolov4.detect(frame, 0, conf_thresh=0.78)
                if detections:
                    self.f_num = 0
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)
                    return True

            self.f_num += stride
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)

    def run(self):
        def _process_frame(frame, focus='global'):
            if self.f_num % self.track_stride == 0:
                bboxes = self.yolov4.detect(frame, 0, conf_thresh=0.65,
                                            resize_dims=(416, 416))
                self.osnet.extraction_batch(frame, bboxes, self.f_num)
                if bboxes:
                    self.person_data[self.f_num] = bboxes

            if self.f_num % self.id_stride == 0:
                if focus == 'local':
                    regions = utils.cluster_bboxes_into_regions(
                        bboxes, *self.resolution
                    )
                    face_dfs = self.face_iq.identify_faces(
                        frame, cutoff=0.8, regions=regions
                    )
                elif focus == 'global':
                    face_dfs = self.face_iq.identify_faces(
                        frame, cutoff=0.8
                    )
                if face_dfs:
                    self.face_data[self.f_num] = face_dfs
        
            if self.f_num % self.kp_stride == 0:
                all_keypoints = self.movenet.detection_batch(frame, bboxes)
                if all_keypoints:
                    self.keypoint_data[self.f_num] = all_keypoints

            return True
    
        def _continue():    
            if self.track_stride <= 15:
                self.f_num += 1
            else:
                self.f_num += self.track_stride
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)
            
            if self.f_num % self.checkpoint_stride == 0:
                progress = int(round((self.f_num / self.total_frames) * 100, 0))
                print(f'{progress}%')
        
        def _wrap_up():
            def _format_face_data(face_data):
                merged_dfs = []
                for frame, dfs in face_data.items():
                    valid_dfs = [df for df in dfs if not df.empty]
                    if valid_dfs:
                        merged_df = pd.concat(valid_dfs, ignore_index=True)
                        merged_df['f'] = frame
                        merged_dfs.append(merged_df)

                if not merged_dfs:
                    return None

                full_df = pd.concat(merged_dfs, ignore_index=True)

                drop_columns = [
                    'target_x', 'target_y', 'target_w', 'target_h', 'threshold'
                ]
                full_df = full_df.drop([col for col in drop_columns if col in full_df.columns], axis=1)

                full_df = full_df.rename(columns={'source_x': 'x', 'source_y': 'y',
                                                'source_w': 'w', 'source_h': 'h'})

                return full_df

            if len(self.osnet.embedding_buffer) > 0:
                self.osnet.flush_buffers(close_file=True)
            self.cap.release()

            self.face_data = _format_face_data(self.face_data)
            
            self.collect_data()

        print(f"Running inference pipeline for {self.video_file}...")

        self.f_num = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)
        prev_frame = -1

        while self.f_num < self.total_frames:
            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()
            if (not ret) or (current_frame == prev_frame):
                break
            prev_frame = current_frame

            _process_frame(frame, focus='global')
            _continue()

        _wrap_up()
        return self.person_data, self.keypoint_data, self.face_data

    def collect_data(self, output_dir="../files/output/runtime_data"):
        print('Collecting inference data')
        git_commit_hash = utils.get_git_commit_hash()
        clip_identifier = self.video_file.split('.')[0] + '_' + git_commit_hash
        os.makedirs(output_dir, exist_ok=True)

        config_data = {
            "module": [
                "yolov4", "yolov4",
                "osnet", "osnet",
                "movenet",
                "faceiq", "faceiq",
                "video",
                "version"
            ],
            "parameter": [
                "nms_threshold", "confidence_threshold",
                "input_shape", "output_shape",
                "confidence_threshold",
                "id_model", "detect_model",
                "resolution",
                "git_commit_hash"
            ],
            "value": [
                self.yolov4.nms_thresh, self.yolov4.conf_thresh,
                self.osnet.input_shape, self.osnet.output_shape,
                self.movenet.conf_thresh,
                self.face_iq.id_model, self.face_iq.detect_model,
                f"{self.resolution[0]}x{self.resolution[1]} @ {self.fps} fps",
                git_commit_hash
            ]
        }
        config_df = pd.DataFrame(config_data)

        performance_data = {
            "metric": [
                "object_detection_time",
                "pose_estimation_time",
                "feature_extraction_time",
                "extraction_flush_time",
                "identification_time",
            ],
            "value": [
                self.yolov4.detection_time,
                self.movenet.detection_time,
                self.osnet.extraction_time,
                self.osnet.flush_time,
                self.face_iq.identification_time,
            ]
        }
        performance_df = pd.DataFrame(performance_data)

        excel_path = os.path.join(output_dir, f'inference_data_{clip_identifier}.xlsx')
        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                config_df.to_excel(writer, sheet_name='Inference Configuration', index=False)
                performance_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
        except Exception as e:
            print(f"Failed to save Excel file: {e}")

        print(f'Saved inference data to {excel_path}')
        return config_df, performance_df
