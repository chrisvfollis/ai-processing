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
from models.yolov4 import YOLOv4
from models.osnet import OSNet
from models.face_iq import FaceIq
from utilities import utilities as utils
from utilities import io_utils
from utilities.utilities import press_stopwatch
from utilities.logger import get_logger


logger = get_logger(__name__)


class InferencePipeline:
    def __init__(self, video_file, model_info, device, yolo_params=None,
                 osnet_params=None, faceiq_params=None,
                 footage_dir='../files/input/'):
        def _instantiate_models(model_info, device, yolo_params, osnet_params,
                                faceiq_params):
            if not yolo_params:
                yolov4 = YOLOv4(model_info[0], device)
            else:
                yolov4 = YOLOv4(model_info[0], device, **yolo_params)
            
            if not osnet_params:
                osnet = OSNet(model_info[1], device)
            else:
                osnet = OSNet(model_info[1], device, **osnet_params)
            
            osnet.activate_buffers(video_file)
            
            if not faceiq_params:
                face_iq = FaceIq(*model_info[2], device=device)
            else:
                face_iq = FaceIq(*model_info[2], device=device, **faceiq_params)
            
            return yolov4, osnet, face_iq
        
        press_stopwatch(self, 'init_time')

        self.video_file = video_file
        self.video_path = os.path.join(footage_dir, video_file)

        resolution, fps, total_frames = utils.get_video_info(self.video_path)
        self.total_frames = total_frames
        self.fps = fps
        self.resolution = resolution
        self.f_num = 0

        yolov4, osnet, face_iq = _instantiate_models(
            model_info, device, yolo_params, osnet_params, faceiq_params
        )

        self.yolov4 = yolov4
        self.osnet = osnet
        self.face_iq = face_iq

        self.track_stride = max(1, self.fps // 10)
        self.id_stride = (self.fps // self.track_stride) * self.track_stride

        self.progress_interval = (
            ((total_frames // 4) // self.track_stride) * self.track_stride
        )

        self.object_detections = {}
        self.face_detections = {}

        self.primary_run_time = 0
        self.read_time = 0
        self.garbage_collection_time = 0
        self.skim_time = 0

        press_stopwatch(self, 'init_time')

    def skim(self):
        logger.info(f'Skimming...')
        press_stopwatch(self, 'skim_time')

        cap = cv2.VideoCapture(self.video_path)
        stride = self.fps * 2

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

    def run(self):
        def _process_frame(frame, focus='global'):
            if self.f_num % self.track_stride == 0:
                bboxes = self.yolov4.detect(frame, 0)
                if bboxes:
                    self.osnet.extraction_batch(frame, bboxes, self.f_num)
                    self.object_detections[self.f_num] = bboxes
                    
            face_dfs = []
            if self.f_num % self.id_stride == 0:
                if focus == 'local':
                    if bboxes:
                        regions = utils.cluster_bboxes_into_regions(
                            bboxes, *self.resolution
                        )
                        face_dfs = self.face_iq.identify_faces(
                            frame, id_cutoff=0.8, regions=regions
                        )
                elif focus == 'global':
                    if bboxes:
                        face_dfs = self.face_iq.identify_faces(
                            frame, id_cutoff=0.8
                        )
                if face_dfs:
                    self.face_detections[self.f_num] = face_dfs
    
        def _continue_forward(cap, current_frame):
            if self.track_stride <= 15:
                self.f_num += 1
            else:
                self.f_num += self.track_stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)
            
            if self.f_num % self.progress_interval == 0:
                progress = int(round((self.f_num / self.total_frames) * 100, 0))
                logger.info(f'{progress}%')
            
            if (self.f_num % 100) == 0:
                press_stopwatch(self, 'garbage_collection_time')
                gc.collect()
                torch.cuda.empty_cache()
                press_stopwatch(self, 'garbage_collection_time')
            
            prev_frame = current_frame
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            return (prev_frame, current_frame)

        logger.info(f'Running inference pipeline for {self.video_file}...')
        press_stopwatch(self, 'primary_run_time')

        cap = cv2.VideoCapture(self.video_path)
        frame_position = (-1, 0)

        while self.f_num < self.total_frames:
            prev_frame, current_frame = frame_position

            press_stopwatch(self, 'read_time')
            ret, frame = cap.read()
            press_stopwatch(self, 'read_time')

            if (not ret) or (current_frame == prev_frame):
                break

            _process_frame(frame, focus='local')
    
            frame_position = _continue_forward(cap, current_frame)
            del frame

        logger.info(f'Exiting inference run on frame {self.f_num}')
        cap.release()

        if len(self.osnet.embedding_buffer) > 0:
            self.osnet.flush_buffers(release=True)
        else:
            self.osnet.release_buffers()

        io_utils.clear_memory()

        self.face_detections = self.consolidate_face_data(self.face_detections)

        self.save_runtime_data()

        press_stopwatch(self, 'primary_run_time')
        return (self.object_detections, self.face_detections)

    def consolidate_face_data(self, face_data):
        merged_dfs = []
        for frame, dfs in face_data.items():
            valid_dfs = [df for df in dfs if not df.empty]
            if valid_dfs:
                merged_df = pd.concat(valid_dfs, ignore_index=True)
                merged_df['f'] = frame
                merged_dfs.append(merged_df)

        if not merged_dfs:
            return None

        return pd.concat(merged_dfs, ignore_index=True)

    def save_runtime_data(self, output_dir='../files/output/runtime_data'):
        commit_hash, commit_datetime = utils.get_git_commit_info()

        clip_identifier = self.video_file.split('.')[0] + '_' + commit_hash
        os.makedirs(output_dir, exist_ok=True)

        prior_runtime_data = os.listdir(output_dir)
        if len(prior_runtime_data) > 200:
            for filename in prior_runtime_data:
                file_path = os.path.join(output_dir, filename)
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
        
        filename = io_utils.get_unique_filename(output_dir, f'inference_data_{clip_identifier}.xlsx')
        excel_path = os.path.join(output_dir, filename)

        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                config_df.to_excel(writer, sheet_name='Inference Configuration', index=False)
                performance_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                logger.info(f'Saved inference runtime data to {excel_path}')
        except Exception as e:
            logger.info(f'Failed to save Excel file: {e}')

    def save_inference_data(self, output_dir='../files/output'):
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = self.video_file.split('.')[0]

        data = [self.object_detections, self.face_detections]

        filename = io_utils.get_unique_filename(
            output_dir, f'{file_prefix}_inference_data.pkl'
        )

        save_path = os.path.join(output_dir, filename)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        logger.info('Inference data saved')
