import pandas as pd
import os
import cv2
from utilities import utilities as utils
from utilities import io_utils
from models.yolov4 import YOLOv4
from models.osnet import OSNet
from models.movenet import MoveNet
from models.face_iq import FaceIq
import gc
import time
import torch
import pickle


class InferencePipeline:
    def __init__(self, video_file, model_info, device, yolo_params=None,
                 osnet_params=None, movenet_params=None, faceiq_params=None,
                 footage_dir='../files/input/'):
        def _instantiate_models(model_info, device, yolo_params, osnet_params,
                                movenet_params, faceiq_params):
            if not yolo_params:
                yolov4 = YOLOv4(model_info[0], device)
            else:
                yolov4 = YOLOv4(model_info[0], device, **yolo_params)
            
            if not osnet_params:
                osnet = OSNet(model_info[1], device)
            else:
                osnet = OSNet(model_info[1], device, **osnet_params)
            
            osnet.activate_buffers(video_file)
            
            if not movenet_params:
                movenet = MoveNet(model_info[2])
            else:
                movenet = MoveNet(model_info[2], **movenet_params)
            
            if not faceiq_params:
                face_iq = FaceIq(*model_info[3])
            else:
                face_iq = FaceIq(*model_info[3], **faceiq_params)
            
            return yolov4, osnet, movenet, face_iq
        
        start_init = time.perf_counter()

        self.video_file = video_file
        self.video_path = os.path.join(footage_dir, video_file)

        resolution, fps, total_frames = utils.get_video_info(self.video_path)
        self.total_frames = total_frames
        self.fps = fps
        self.resolution = resolution
        self.f_num = 0

        yolov4, osnet, movenet, face_iq = _instantiate_models(
            model_info, device, yolo_params, osnet_params, movenet_params,
            faceiq_params
        )

        self.yolov4 = yolov4
        self.osnet = osnet
        self.movenet = movenet
        self.face_iq = face_iq

        self.track_stride = max(1, self.fps // 10)
        self.id_stride = (self.fps // self.track_stride) * self.track_stride
        self.kp_stride = self.id_stride * 3

        self.progress_interval = (
            ((total_frames // 4) // self.track_stride) * self.track_stride
        )

        self.person_data = {}
        self.face_data = {}
        self.keypoint_data = {}

        self.primary_run_time = 0
        self.read_time = 0
        self.garbage_collection_time = 0
        self.skim_time = 0
        self.init_time = (time.perf_counter() - start_init)

    def skim(self):
        print(f'Skimming...')
        start_skim = time.perf_counter()

        cap = cv2.VideoCapture(self.video_path)
        stride = self.fps * 2

        prev_frame, f_num = (-1, 0)
        result = False
        while f_num < self.total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)

            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()

            if (not ret) or (current_frame == prev_frame):
                print(f'Nothing to process in {self.video_file}')
                break
            elif utils.is_grayscale(frame, threshold=10):
                print(f'Footage too dark in {self.video_file}')
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

        end_skim = time.perf_counter()
        self.skim_time = (end_skim - start_skim)

        return result

    def run(self):
        def _process_frame(frame, focus='global'):
            if self.f_num % self.track_stride == 0:
                bboxes = self.yolov4.detect(frame, 0)
                if bboxes:
                    self.osnet.extraction_batch(frame, bboxes, self.f_num)
                    self.person_data[self.f_num] = bboxes
                    
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
                    face_dfs = self.face_iq.identify_faces(
                        frame, id_cutoff=0.8
                    )
                if face_dfs:
                    self.face_data[self.f_num] = face_dfs
        
            if self.f_num % self.kp_stride == 0:
                all_keypoints = self.movenet.detection_batch(frame, bboxes)
                if all_keypoints:
                    self.keypoint_data[self.f_num] = all_keypoints
    
        def _continue_forward(cap, current_frame):
            if self.track_stride <= 15:
                self.f_num += 1
            else:
                self.f_num += self.track_stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.f_num)
            
            if self.f_num % self.progress_interval == 0:
                progress = int(round((self.f_num / self.total_frames) * 100, 0))
                print(f'{progress}%')
            
            if (self.f_num % 100) == 0:
                start_gc = time.perf_counter()
                gc.collect()
                torch.cuda.empty_cache()

                end_gc = time.perf_counter()
                self.garbage_collection_time += (end_gc - start_gc)
            
            prev_frame = current_frame
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            return (prev_frame, current_frame)

        print(f'Running inference pipeline for {self.video_file}...')
        start_run = time.perf_counter()
        
        cap = cv2.VideoCapture(self.video_path)
        frame_position = (-1, 0)

        while self.f_num < self.total_frames:
            prev_frame, current_frame = frame_position

            start_read = time.perf_counter()
            ret, frame = cap.read()

            end_read = time.perf_counter()
            self.read_time += (end_read - start_read)

            if (not ret) or (current_frame == prev_frame):
                break

            _process_frame(frame, focus='local')
    
            frame_position = _continue_forward(cap, current_frame)
            del frame

        print(f'Exiting inference run on frame {self.f_num}')
        cap.release()

        if len(self.osnet.embedding_buffer) > 0:
            self.osnet.flush_buffers(release=True)
        else:
            self.osnet.release_buffers()

        io_utils.clear_memory()

        self.face_data = self.format_face_data(self.face_data)

        self.save_runtime_data()

        end_run = time.perf_counter()
        self.primary_run_time += (end_run - start_run)

        return self.person_data, self.keypoint_data, self.face_data

    def format_face_data(self, face_data):
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

    def save_runtime_data(self, output_dir='../files/output/runtime_data'):
        commit_hash, commit_datetime = utils.get_git_commit_info()

        clip_identifier = self.video_file.split('.')[0] + '_' + commit_hash
        os.makedirs(output_dir, exist_ok=True)

        config_data = {
            'module': [
                *['software'] * 2,
                *['video'] * 2,
                *['yolov4'] * 3,
                *['osnet'] * 2,
                *['movenet'] * 1,
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

                'confidence_threshold',     # MoveNet

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
    
                self.movenet.conf_thresh,

                self.face_iq.detection_model,
                self.face_iq.recognition_model
            ]
        }
        config_df = pd.DataFrame(config_data)

        performance_data = {
            'module': [
                *['pipeline'] * 5,
                *['yolov4'] * 3,
                *['osnet'] * 3,
                *['movenet'] * 3,
                *['faceiq'] * 2
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

                'preprocess_time',                  # Movenet
                'inference_time',
                'postprocess_time',

                'inference_time',                   # Faceiq
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

                self.movenet.preprocess_time,
                self.movenet.detection_time,
                self.movenet.postprocess_time,

                self.face_iq.identification_time,
                self.face_iq.postprocess_time
            ]
        }
        performance_df = pd.DataFrame(performance_data)
        
        filename = io_utils.get_unique_filename(output_dir, f'inference_data_{clip_identifier}.xlsx')
        excel_path = os.path.join(output_dir, filename)

        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                config_df.to_excel(writer, sheet_name='Inference Configuration', index=False)
                performance_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                print(f'Saved inference runtime data to {excel_path}')
        except Exception as e:
            print(f'Failed to save Excel file: {e}')

    def save_inference_data(self, output_dir='../files/output'):
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = self.video_file.split('.')[0]

        data = [self.person_data, self.keypoint_data, self.face_data]

        filename = io_utils.get_unique_filename(
            output_dir, f'{file_prefix}_inference_data.pkl'
        )

        save_path = os.path.join(output_dir, filename)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        print('Inference data saved')
