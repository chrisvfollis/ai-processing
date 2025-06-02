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
from models import YoloX, OSNet
from modules import FaceAnalysis
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
            faces_cfg: dict = {},
            track_stride: int = 1,
            id_freq: str = '1/s',
        ):
        press_stopwatch(self, 'init_time')

        # MODEL SETUP:
        self.device = device or utils.get_default_device()

        self.yolox = YoloX(device=self.device, **yolo_cfg)
        self.osnet = OSNet(device=self.device, **osnet_cfg)
        self.face_analysis = FaceAnalysis(device=self.device, **faces_cfg)

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
        self.f_total = video_info[3]

        self.f_num = 0

        # PARAMETERS:
        self.track_stride = track_stride

        self.effective_fps = self.fps // self.track_stride
        self.aligned_1s_interval = self.effective_fps * self.track_stride

        if id_freq == '1/s':
            self.id_stride = self.aligned_1s_interval
        else:
            id_freq = int(id_freq.split('/')[0])
            self.id_stride = self.aligned_1s_interval // id_freq

        self.progress_interval = (
            ((self.f_total // 4) // self.track_stride) * self.track_stride
        )

        # INFERENCE DATA STORAGE:
        self.person_detections = {}
        self.face_data = {}

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
        while f_num < self.f_total:
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
                detections = self.yolox.inference(frame, conf_thresh=0.78)
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

    def run(self, batch_size: int = 16):
        self.progress = 0
        logger.info(f'Running inference pipeline for {self.video_file}...')
        press_stopwatch(self, 'primary_run_time')

        self.cap = cv2.VideoCapture(self.video_path)

        while self.f_num < self.f_total:
            frame_batch, log_progress = self.collect_frames(batch_size)
            results = self.process_batch(frame_batch)

            self.person_detections = self.person_detections | results[0]
            self.face_data = self.face_data | results[1]

            del frame_batch

            if log_progress == True:
                logger.info(f'{self.progress}%')

        logger.info(f'Exiting inference run on frame {self.f_num}')
        self.cap.release()

        return self.finalize_run()

    def collect_frames(self, batch_size: int = 16):
        log_progress_update = False

        batch_start = self.f_num
        batch_end = min(self.f_total, (
            batch_start + (batch_size * self.track_stride)
        ))
        frames = []
        id_frames = {}

        press_stopwatch(self, 'read_time')
        while self.f_num < batch_end:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if self.f_num % self.track_stride == 0:
                frames.append(frame)
                idx = len(frames) - 1
                if self.f_num % self.id_stride == 0:
                    id_frames[idx] = {
                        'frame': frame,
                        'f_num': self.f_num,
                    }

            self.f_num += 1
            
            if self.f_num % self.progress_interval == 0:
                log_progress_update = True

                percent_complete = self.f_num / self.f_total * 100
                self.progress = int(round(percent_complete, 0))

        press_stopwatch(self, 'read_time')

        frame_batch = {
            'start': batch_start,    # included in the batch
            'end': batch_end,   # not included: only marks the end
            'frames': frames,
            'id_frames': id_frames,
        }

        return frame_batch, log_progress_update

    def process_batch(self, frame_data):
        frames = frame_data['frames']
        id_frames = frame_data['id_frames']

        yolo_output = self.yolox.inference(frames)

        face_data = {}
        for idx, id_frame in id_frames.items():
            if idx >= len(yolo_output):
                continue
            img = id_frame['frame']
            f_num = id_frame['f_num']
            
            detections = yolo_output[idx]
            if detections is None or len(detections) == 0:
                continue
            else:
                detections = [
                    utils.xywh_xyxy(d, out='xywh') for d in detections
                ]
            self.osnet.extraction_batch(img, detections, f_num)

            img_h, img_w = img.shape[:2]
            regions = utils.cluster_bboxes_into_regions(
                detections, img_h, img_w, margin=15
            )

            facial_areas = self.face_analysis.identify_faces(img, regions)
            face_data[f_num] = facial_areas
        
        person_detections = {}
        for idx, detections in enumerate(yolo_output):
            f_num = frame_data['start'] + (idx * self.track_stride)

            person_detections[f_num] = detections

        return person_detections, face_data

    def finalize_run(self):
        if len(self.osnet.embedding_buffer) > 0:
            self.osnet.flush_buffers(structure='video_data', release=True)
        else:
            self.osnet.release_buffers()

        io_utils.clear_memory()
        self.save_runtime_data()

        self.face_data = self.face_analysis.consolidate_face_data(self.face_data)

        press_stopwatch(self, 'primary_run_time')

        return self.person_detections, self.face_data

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
                *['pipeline'] * 3,
                *['yolox'] * 3,
                *['osnet'] * 2,
                *['face_analysis'] * 2,
            ],
            'parameter': [
                'git_commit_hash',          # Software
                'git_commit_datetime',

                'resolution',               # Video processing
                'nominal_fps',
                'effective_fps',
                
                'input_dims',               # YOLOX
                'nms_thresh',
                'conf_thresh',

                'input_dims',               # OSNet
                'output_shape',

                'id_cutoff',                # Face analysis
                'enhance',
            ],
            'value': [
                commit_hash,                                    
                commit_datetime,

                f'{self.resolution[0]}x{self.resolution[1]}',   
                f'{self.fps} fps',
                f'{self.effective_fps} fps',

                self.yolox.input_size,                         
                self.yolox.nms_thresh,
                self.yolox.conf_thresh,

                self.osnet.input_dims,                          
                self.osnet.output_shape,
    
                self.face_analysis.id_cutoff,
                self.face_analysis.enhance_faces,
            ],
        }
        config_df = pd.DataFrame(config_data)

        performance_data = {
            'module': [
                *['pipeline'] * 5,
                *['yolox'] * 3,
                *['osnet'] * 3,
                *['face_analysis'] * 4
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
                
                self.yolox.preprocess_time,
                self.yolox.inference_time,
                self.yolox.postprocess_time,

                self.osnet.preprocess_time,
                self.osnet.embedding_time,
                self.osnet.flush_time,

                self.face_analysis.identification_pipeline_time,
                self.face_analysis.face_detection_time,
                self.face_analysis.face_recognition_time,
                self.face_analysis.other_processing_time,
            ],
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
        state['yolox'] = None
        state['osnet'] = None
        state['face_analysis'] = None

        state['cap'] = None

        for f, dets in state['person_detections'].items():
            if isinstance(dets, torch.Tensor):
                state['person_detections'][f] = dets.detach().cpu().numpy().tolist()
            else:
                converted = []
                for d in dets:
                    if isinstance(d, torch.Tensor):
                        converted.append(d.detach().cpu().numpy().tolist())
                    else:
                        converted.append(d)
                state['person_detections'][f] = converted

        press_stopwatch(self, 'pkl_io')
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        press_stopwatch(self, 'pkl_io')

        logger.info(f'Inference pipeline state saved to {save_path}')
