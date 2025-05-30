# standard dependencies
import os
import time
import pickle
import gc

# 3rd-party dependencies
import pandas as pd
import cv2
import torch

# internal dependencies
from models import YOLOv4, OSNet, FaceIq
from utilities import general_utils as utils
from utilities import io_utils
from utilities.log_utils import get_logger, press_stopwatch


logger = get_logger(__name__)


class InferencePipeline:
    def __init__(
            self,
            video_file: str,
            device: torch.device = None,
            yolo_cfg: dict = {},
            osnet_cfg: dict = {},
            faceiq_cfg: dict = {},
        ):
        press_stopwatch(self, 'init_time')

        # MODEL SETUP:
        self.device = device or utils.get_default_device()

        self.yolov4 = YOLOv4(device=self.device, **yolo_cfg)
        self.osnet = OSNet(device=self.device, **osnet_cfg)
        self.face_iq = FaceIq(device=self.device, **faceiq_cfg)

        self.osnet.activate_buffers(
            file_prefix=video_file.split('.')[0],
            structure='video_data'
        )

        # PATHS:
        self.project_root = io_utils.get_project_root()
        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')

        # VIDEO ATTRIBUTES:
        self.video_file = video_file
        self.video_path = os.path.join(self.input_dir, video_file)

        video_info = utils.get_video_info(self.video_path)

        self.resolution = video_info[0]
        self.frame_diag = video_info[1]
        self.fps = video_info[2]
        self.total_frames = video_info[3]

        self.f_num = 0

        # PARAMETERS:
        self.track_stride = max(1, self.fps // 10)
        self.id_stride = (self.fps // self.track_stride) * self.track_stride

        self.progress_interval = (
            ((self.total_frames // 4) // self.track_stride) * self.track_stride
        )

        # INFERENCE DATA STORAGE:
        self.object_detections = {}
        self.face_detections = {}

        # TIMING ATTRIBUTES:
        self.primary_run_time = 0
        self.read_time = 0
        self.garbage_collection_time = 0
        self.skim_time = 0

        press_stopwatch(self, 'init_time')

    def skim(self):
        logger.info(f'Skimming...')
        press_stopwatch(self, 'skim_time')

        cap = cv2.VideoCapture(self.video_path)
        stride = self.fps * 3

        prev_frame, f_num = (-1, 0)
        result = False
        while f_num < self.total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)

            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()

            if (not ret) or (current_frame == prev_frame):
                logger.info(f'Nothing to process in {self.video_file}')
                break
            elif utils.is_grayscale(frame, threshold=10):
                logger.info(f'Footage too dark in {self.video_file}')
                break

            if f_num % stride == 0:
                detections = self.yolov4.detect(frame, 0, conf_thresh=0.78)
                del frame
                if detections:
                    result = True
                    break
            
            if (f_num % 100) == 0:
                gc.collect()

            prev_frame = current_frame
            f_num += stride
            
        cap.release()

        press_stopwatch(self, 'skim_time')
        return result

    def collect_frames(self, batch_size: int = 32):
        batch_end = min(self.total_frames, (self.f_num + batch_size))

        frames = []
        while self.f_num < batch_end:
            
            ret, frame = self.cap.read()
            if not ret:
                break

            frames.append(frame)
            self.f_num += 1

            if self.f_num % self.progress_interval == 0:
                progress = int(round((self.f_num / self.total_frames) * 100, 0))
                logger.info(f'{progress}%')

        return frames

    def process_batch(self, frames):
        pass

    def run(self, batch_size: int = 32):
        logger.info(f'Running inference pipeline for {self.video_file}...')
        press_stopwatch(self, 'primary_run_time')

        self.cap = cv2.VideoCapture(self.video_path)

        while self.f_num < self.total_frames:
            press_stopwatch(self, 'read_time')
            frames = self.collect_frames(batch_size=batch_size)
            press_stopwatch(self, 'read_time')

            self.process_batch(frames)

            del frames

        logger.info(f'Exiting inference run on frame {self.f_num}')
        self.cap.release()

        return self.finalize_run()

    def finalize_run(self):
        if len(self.osnet.embedding_buffer) > 0:
            self.osnet.flush_buffers(structure='video_data', release=True)
        else:
            self.osnet.release_buffers()

        io_utils.clear_memory()
        self.save_runtime_data()

        press_stopwatch(self, 'primary_run_time')

        return self.object_detections, self.face_detections

    def save_runtime_data(self):
        runtime_data_dir = os.path.join(self.output_dir, 'runtime_data/')
        commit_hash, commit_datetime = utils.get_git_commit_info()

        clip_identifier = self.video_file.split('.')[0] + '_' + commit_hash
        os.makedirs(runtime_data_dir, exist_ok=True)

        prior_runtime_data = os.listdir(runtime_data_dir)
        if len(prior_runtime_data) > 200:
            for filename in prior_runtime_data:
                file_path = os.path.join(runtime_data_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        config_data = {
            'module': [
                *['software'] * 2,
                *['video'] * 2,
                *['yolov4'] * 3,
                *['osnet'] * 2,
                *['faceiq'] * 2
            ],
            'parameter': [
                'git_commit_hash',          # Software
                'git_commit_datetime',

                'resolution',               # Video
                'fps',
                
                'input_dims',               # YOLOv4
                'nms_threshold',
                'confidence_threshold',

                'input_dims',               # OSNet
                'output_shape',

                'detection_model',          # Faceiq
                'recognition_model'
            ],
            'value': [
                commit_hash,                                    
                commit_datetime,

                f'{self.resolution[0]}x{self.resolution[1]}',   
                f'{self.fps} fps',

                self.yolov4.input_dims,                         
                self.yolov4.nms_thresh,
                self.yolov4.conf_thresh,

                self.osnet.input_dims,                          
                self.osnet.output_shape,
    
                self.face_iq.det_model_name,
                self.face_iq.rec_model_name
            ]
        }
        config_df = pd.DataFrame(config_data)

        performance_data = {
            'module': [
                *['pipeline'] * 5,
                *['yolov4'] * 3,
                *['osnet'] * 3,
                *['faceiq'] * 4
            ],
            'metric': [             
                'primary_run_time',                 # Pipeline            
                'frame_read_time',
                'garbage_collection_time',
                'initialize_time',
                'video_skim_time',

                'preprocess_time',                  # YOLOv4
                'inference_time',
                'postprocess_time',
                
                'preprocess_time',                  # OSNet
                'inference_time',
                'flush_time',

                'identification_pipeline_time',     # Faceiq
                'detection_inference_time',
                'recognition_inference_time',
                'postprocess_time'
            ],
            'value': [
                self.primary_run_time,
                self.read_time,
                self.garbage_collection_time,
                self.init_time,
                self.skim_time,
                
                self.yolov4.preprocess_time,
                self.yolov4.detection_time,
                self.yolov4.postprocess_time,

                self.osnet.preprocess_time,
                self.osnet.embedding_time,
                self.osnet.flush_time,

                self.face_iq.identification_pipeline_time,
                self.face_iq.face_detection_time,
                self.face_iq.face_recognition_time,
                self.face_iq.other_processing_time
            ]
        }
        performance_df = pd.DataFrame(performance_data)
        
        filename = io_utils.get_unique_filename(runtime_data_dir, f'inference_data_{clip_identifier}.xlsx')
        excel_path = os.path.join(runtime_data_dir, filename)

        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                config_df.to_excel(writer, sheet_name='Inference Configuration', index=False)
                performance_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                logger.info(f'Saved inference runtime data to {excel_path}')
        except Exception as e:
            logger.info(f'Failed to save Excel file: {e}')

    def save_pipeline_state(self):
        logger.info('Saving inference pipeline state...')

        file_prefix = self.video_file.split('.')[0]
        filename = io_utils.get_unique_filename(
            self.output_dir, f'{file_prefix}_inference_pipeline.pkl'
        )
        save_path = os.path.join(self.output_dir, filename)

        # make shallow copy and remove unpickleable objects
        state = self.__dict__.copy()
        state['yolov4'] = None
        state['osnet'] = None
        state['face_iq'] = None

        for f, detections in state['object_detections'].items():
            for i, det in enumerate(detections):
                if isinstance(det, torch.Tensor):
                    detections[i] = det.cpu().numpy().tolist()

        press_stopwatch(self, 'pkl_io')
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        press_stopwatch(self, 'pkl_io')

        logger.info(f'Inference pipeline state saved to {save_path}')
