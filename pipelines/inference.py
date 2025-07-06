# standard dependencies
import os
import pickle
import gc
import warnings

warnings.filterwarnings(
    'ignore', message='You are using `torch.load` with `weights_only=False`',
    category=FutureWarning
)

# 3rd-party dependencies
import pandas as pd
import cv2
import av
import torch
import numpy as np

# internal dependencies
from models import YoloX, OSNet
from modules.face_analysis import FaceAnalysis
from utilities import general_utils as utils
from utilities import io_utils, log_utils


logger = log_utils.get_logger(__name__)


class InferencePipeline:
    def __init__(
            self,
            video_file: str,
            model_cfg: dict = {},
            device: torch.device = None,
            track_stride: int = 1,
            id_freq: str = '2 Hz',
            use_features: bool = True,
    ):
        log_utils.press_stopwatch(self, 'init_time')

        # MODEL SETUP:
        self.device = device or utils.get_default_device()

        yolox_cfg = model_cfg['yolox'] | {'device': self.device}
        faces_cfg = model_cfg['faces'] | {'device': self.device}
            
        self.yolox = YoloX(**yolox_cfg)
        self.face_analysis = FaceAnalysis(**faces_cfg)

        if use_features:
            osnet_cfg = model_cfg['osnet'] | {'device': self.device}
            self.osnet = OSNet(**osnet_cfg)

            self.osnet.activate_buffers(
                file_prefix=video_file.split('.')[0],
                structure='video_data'
            )

        # PATHS/FILENAMES/ETC:
        self.project_root = io_utils.get_project_root()
        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')

        self.video_file = video_file
        self.video_path = os.path.join(self.input_dir, video_file)

        # VIDEO ATTRIBUTES:
        video_info = utils.get_video_info(self.video_path, release=True)

        self.resolution = video_info[0]
        self.frame_diag = video_info[1]
        self.fps = video_info[2]
        self.f_total = video_info[3]
        # self.f_total = 4800

        self.f_num = 0

        self.time_prefix, self.cam_id = utils.decode_vid_filename(self.video_file)

        # PARAMETERS:
        self.track_stride = track_stride

        self.prog_interval = (
            ((self.f_total // 4) // self.track_stride) * self.track_stride
        )
        self.effective_fps = self.fps // self.track_stride
        self.aligned_1s_interval = self.effective_fps * self.track_stride

        if id_freq == 'fps':
            self.id_stride = 1
        else:
            id_Hz_val = int(
                str(id_freq).split()[0]
            )
            self.id_stride = self.aligned_1s_interval // id_Hz_val

        self.use_features = use_features

        # INFERENCE DATA STORAGE:
        self.person_detections = {}
        self.face_data = {}

        self.region_log = []

        # TIMING ATTRIBUTES:
        self.primary_run_time = 0
        self.read_time = 0
        self.garbage_collection_time = 0
        self.skim_time = 0

        log_utils.press_stopwatch(self, 'init_time')

    def skim(self):
        logger.info(f'Skimming...')
        log_utils.press_stopwatch(self, 'skim_time')

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

        log_utils.press_stopwatch(self, 'skim_time')
        return result

    def run(self, batch_size: int = 20):
        self.progress = 0
        logger.info(f'Running inference pipeline for {self.video_file}...')
        log_utils.press_stopwatch(self, 'primary_run_time')

        while self.f_num < self.f_total:
            frame_batch, log_progress = self._collect_frames(batch_size)
            results = self.process_batch(frame_batch)

            self.person_detections = self.person_detections | results[0]
            self.face_data = self.face_data | results[1]

            del frame_batch

            if log_progress == True:
                logger.progress(f'inference —> {self.progress}%')
        
        log_utils.press_stopwatch(self, 'primary_run_time')

        self._cleanup()

        self.face_data = self.face_analysis.consolidate_face_data(
            self.face_data, self.fps, self.cam_id
        )
        if (self.face_data is None) or (self.face_data.empty):
            logger.info(f'No face data from inference run')

        return self.person_detections, self.face_data

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

            raw_detections = detections
            detections = [
                utils.xywh_xyxy(d, out='xywh') for d in detections
            ]
            high_conf_dets = [
                converted for converted, raw in zip(detections, raw_detections)
                if raw[4] >= 0.10   # threshold for detections to qualify in regions
            ]
            if self.use_features:
                try:
                    self.osnet.extraction_batch(img, detections, f_num)
                except ValueError as e:
                    print(e)
                    continue

            if high_conf_dets:
                img_h, img_w = img.shape[:2]
                regions = utils.cluster_bboxes_into_regions(
                    high_conf_dets, img_h, img_w
                )
                for x, y, w, h in regions:
                    self.region_log.append({
                        'f': f_num,
                        'x': int(x),
                        'y': int(y),
                        'w': int(w),
                        'h': int(h),
                        'cam_id': int(self.cam_id),
                    })
                facial_areas = self.face_analysis.identify_faces(
                    img, regions, id_cutoff=0.90
                )
                face_data[f_num] = facial_areas
        
        person_detections = {}
        for idx, detections in enumerate(yolo_output):
            f_num = frame_data['start'] + (idx * self.track_stride)

            if (
                isinstance(detections, torch.Tensor) or
                isinstance(detections, np.ndarray)
            ):
                person_detections[f_num] = detections

        return person_detections, face_data

    def _collect_frames(self, batch_size: int = 20):
        if not hasattr(self, 'av_container'):
            self.av_container = av.open(self.video_path)
            self.av_stream = self.av_container.streams.video[0]
            self.av_stream.thread_type = 'FRAME'
            self.av_stream.thread_count = 5
            self._av_frame_iter = self.av_container.decode(self.av_stream)
            self._av_next_pts = 0

        log_progress_update = False
        batch_start = self.f_num
        frames = []
        id_frames = {}

        log_utils.press_stopwatch(self, 'read_time')

        while len(frames) < batch_size and self.f_num < self.f_total:
            try:
                frame = next(self._av_frame_iter)
            except StopIteration:
                break

            if frame.pts is None or frame.pts < self._av_next_pts:
                continue

            img = frame.to_ndarray(format='bgr24')  # Converts to BGR numpy array

            if self.f_num % self.track_stride == 0:
                frames.append(img)
                idx = len(frames) - 1
                if self.f_num % self.id_stride == 0:
                    id_frames[idx] = {
                        'frame': img,
                        'f_num': self.f_num,
                    }

            if self.f_num % self.prog_interval == 0:
                log_progress_update = True
                self.progress = utils.calculate_progress(self.f_num, self.f_total)

            self.f_num += 1
            self._av_next_pts += 1  # advance expected pts

        log_utils.press_stopwatch(self, 'read_time')

        return {
            'start': batch_start,
            'end': self.f_num,
            'frames': frames,
            'id_frames': id_frames,
        }, log_progress_update

    def _cleanup(self):
        if self.use_features:
            self.osnet.flush_buffers(structure='video_data', release=True)

        io_utils.clear_memory()

    def save_run_info(self):
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

                self.osnet.input_dims if self.use_features else 'N/A',                          
                self.osnet.output_shape if self.use_features else 'N/A',
    
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

                self.osnet.preprocess_time if self.use_features else 'N/A',
                self.osnet.embedding_time if self.use_features else 'N/A',
                self.osnet.flush_time if self.use_features else 'N/A',

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
        except Exception as e:
            logger.info(f'Failed to save Excel file: {e}')

    def save_state(self):
        file_prefix = self.video_file.split('.')[0]
        filename = io_utils.get_unique_filename(
            self.output_dir, f'{file_prefix}_inference_pipeline.pkl'
        )
        save_path = os.path.join(self.output_dir, filename)

        # make shallow copy and remove unpickleable objects:
        state = self.__dict__.copy()
        state['yolox'] = None
        state['osnet'] = None
        state['face_analysis'] = None

        state['cap'] = None

        state['av_container'] = None
        state['av_stream'] = None
        state['_av_frame_iter'] = None
        state['_av_next_pts'] = None

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

        log_utils.press_stopwatch(self, 'pkl_io')
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        log_utils.press_stopwatch(self, 'pkl_io')

        logger.info('Inference pipeline state saved')
