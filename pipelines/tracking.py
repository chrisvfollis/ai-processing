# standard dependencies
import os
import pickle
from typing import Optional
import math

# 3rd-party dependencies
import pandas as pd
import cv2

# internal dependencies
from utilities import io_utils, log_utils
from utilities import general_utils as utils
from modules import OCSort, KalmanBoxTracker


logger = log_utils.get_logger(__name__)


class TrackingPipeline:
    def __init__(
            self,
            video_file: str,
            detections: dict,
            input_dims: tuple[int] = (800, 1440),
            aspect_ratio_thresh: float = 1.6,
            min_box_area: int = 100,
            det_thresh: float = 0.6,
            max_age=30,
            min_hits=3, 
            iou_threshold=0.3,
            delta_t=3,
            asso_func='iou',
            inertia=0.2,
            use_byte=False,
            f_start=0,
            f_end=None,
            min_lifespan=5,
            min_avg_size=0.004,
            prior_pkl=False,
    ):
        # INFERENCE DATA:
        self.detections = detections

        # PATHS/FILENAMES/ETC:
        self.project_root = io_utils.get_project_root()
        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')

        self.video_file = video_file
        self.video_path = os.path.join(self.input_dir, video_file)

        self.prior_pkl = prior_pkl or ''

        # VIDEO ATTRIBUTES:
        time_prefix, cam_id = utils.decode_vid_filename(video_file)
        res, _, fps, f_total = utils.get_video_info(self.video_path, release=True)

        self.resolution = res
        self.fps = fps

        self.f_start = f_start
        self.f_end = f_end or f_total

        self.start_time = utils.frame_timestamp(time_prefix, self.f_start, fps)
        self.end_time = utils.frame_timestamp(time_prefix, self.f_end, fps)
        
        self.progress_interval = self.f_total // 4

        self.time_prefix = time_prefix
        self.cam_id = cam_id

        # TRACKER:
        self.ocsort = OCSort(
            det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte,
            img_dims=self.resolution[::-1],
            input_dims=input_dims,
            aspect_ratio_thresh=aspect_ratio_thresh,
            min_box_area=min_box_area,
        )
        
        self.trk_obs_df = None
        self.trk_states_df = None
        self.trk_video_data = {}

        self.min_lifespan = min_lifespan
        self.min_avg_size = min_avg_size * math.prod(self.resolution)
        self.filtered_tracks = {}

        # SPEED/PERFORMANCE:
        self.primary_run_time = 0
        self.persist_time = 0

    @property
    def f_total(self):
        return self.f_end - self.f_start

    def continue_prior(self, prior_pkl_path=None):
        def _reset_trk_ids(prior_pipeline):
            reset = {}
            for new_id, (trk_id, trk) in enumerate(
                prior_pipeline.ocsort.active_trks.items()
            ):
                trk.id = new_id
                reset[new_id] = trk

            KalmanBoxTracker.next_id = len(reset)
            return reset

        log_utils.press_stopwatch(self, 'persist_time')

        if not prior_pkl_path:
            cam_id = self.video_file.split('.')[0].split('_')[-1]
            files = [
                f for f in os.listdir(self.output_dir)
                if f.endswith(cam_id + '.pkl')
            ]
            if not files:
                return
            self.prior_pkl = sorted(files)[-1]
            self.prior_pkl_path = os.path.join(self.output_dir, self.prior_pkl)

            prior_pkl_path = self.prior_pkl_path

        log_utils.press_stopwatch(self, 'pkl_io')
        with open(prior_pkl_path, 'rb') as f:
            prior_pipeline = pickle.load(f)
        log_utils.press_stopwatch(self, 'pkl_io')

        frame_gap = int(round(
            (self.start_time - prior_pipeline.end_time)
            .total_seconds() * self.fps
        ))
        prior_pipeline.f_end += frame_gap

        prior_pipeline.ocsort.active_trks = _reset_trk_ids(prior_pipeline)
        prior_pipeline.ocsort.inactive_trks = {}

        self.num_persisted = len(prior_pipeline.ocsort.active_trks)
        logger.info(f'Continuing {self.num_persisted} prior tracks...')

        prior_pipeline.run()

        for trk_id, trk in prior_pipeline.ocsort.active_trks.items():
            trk.start -= prior_pipeline.f_end
            self.ocsort.active_trks[trk_id] = trk

        for trk_id, trk in prior_pipeline.ocsort.inactive_trks.items():
            trk.start -= prior_pipeline.f_end
            self.ocsort.inactive_trks[trk_id] = trk

        log_utils.press_stopwatch(self, 'persist_time')

    def run(self) -> tuple[dict, ...]:
        log_utils.press_stopwatch(self, 'primary_run_time')
        self.f_num = self.f_start
        
        while self.f_num < self.f_end:
            detections = self.detections.get(self.f_num, None)

            online_targets = self.ocsort.update(detections, self.f_num)

            online_boxes = []
            online_ids = []
            for t in online_targets:
                trk_id, box = t[4], utils.xywh_xyxy(t[:4], out='xywh')

                valid_ratio = (box[2] / box[3]) <= self.ocsort.aspect_ratio_thresh
                valid_area = math.prod(box[2:4]) > self.ocsort.min_box_area

                if not (valid_ratio and valid_area):
                    continue
    
                online_boxes.append(box)
                online_ids.append(trk_id)
                self.trk_video_data[self.f_num] = {
                    'online_boxes': online_boxes,
                    'online_ids': online_ids,
                }

            if self.f_num % self.progress_interval == 0:
                progress = utils.calculate_progress(self.f_num, self.f_total)
                logger.progress(f'tracking —> {progress}%')
        
            self.f_num += 1

        log_utils.press_stopwatch(self, 'primary_run_time')
        return self.ocsort.active_trks, self.ocsort.inactive_trks

    def filter_tracks(self):
        low_lifespan = []
        small_avg_area = []

        for trk_id, trk in self.ocsort.inactive_trks.items():
            if io_utils.identity_is_known(trk.identity):
                continue

            lifespan = trk.age / self.fps
            if lifespan < self.min_lifespan:
                low_lifespan.append(trk_id)
            
            elif trk.average_area() < self.min_avg_size:
                small_avg_area.append(trk_id)
        
        for trk_id in set(low_lifespan + small_avg_area):
            try:
                del self.ocsort.inactive_trks[trk_id]
            except KeyError:
                continue

        self.filtered_tracks = {
            'low_lifespan': low_lifespan,
            'small_avg_area': small_avg_area,
        }

    def _calculate_run_stats(self):
        all_trks = self.ocsort.active_trks | self.ocsort.inactive_trks
        
        avg_lifespan = 0
        num_identified = 0
        
        lifespans = []
        for trk in all_trks.values():
            lifespans.append(trk.age / self.fps)

            if io_utils.identity_is_known(trk.identity):
                num_identified += 1
        
        if lifespans:
            avg_lifespan = sum(lifespans) / len(lifespans)

        return avg_lifespan, num_identified

    def save_run_info(self):
        logger.info('Saving tracking run info...')

        runtime_data_dir = os.path.join(self.output_dir, 'runtime_data/')
        os.makedirs(runtime_data_dir, exist_ok=True)

        commit_hash, commit_datetime = utils.get_git_commit_info()
        clip_identifier = f"{self.video_file.split('.')[0]}_{commit_hash}"

        if len(os.listdir(runtime_data_dir)) > 200:
            for f in os.listdir(runtime_data_dir):
                try:
                    os.remove(os.path.join(runtime_data_dir, f))
                except Exception:
                    continue

        config_data = {
            'module': [
                *['software'] * 2,
                *['video'] * 2,
                *['tracker'] * 5
            ],
            'parameter': [
                'git_commit_hash',          # Software
                'git_commit_datetime',

                'resolution',               # Video
                'fps',

                'input_dims',               # Tracker
                'iou_threshold',
                'min_box_area',
                'aspect_ratio_thresh',
                'min_lifespan_filter'
            ],
            'value': [
                commit_hash,
                commit_datetime,

                f'{self.resolution[0]}x{self.resolution[1]}',
                f'{self.fps}',

                self.ocsort.scale,
                self.ocsort.iou_threshold,
                self.ocsort.min_box_area,
                self.ocsort.aspect_ratio_thresh,
                self.min_lifespan,
            ]
        }
        config_df = pd.DataFrame(config_data)

        performance_data = {
            'module': ['pipeline'] * 2,
            'metric': ['primary_run_time', 'persist_time'],
            'value': [self.primary_run_time, self.persist_time],
        }
        performance_df = pd.DataFrame(performance_data)

        avg_lifespan, num_identified = self._calculate_run_stats()

        stats_data = {
            'module': ['tracks'] * 5,
            'metric': [
                'num_inactive',
                'num_lifespan_filtered',
                'num_avg_area_filtered',
                'avg_lifespan',
                'num_identified',
            ],
            'value': [
                len(self.ocsort.inactive_trks),
                len(self.filtered_tracks.get('low_lifespan', [])),
                len(self.filtered_tracks.get('small_avg_area', [])),
                avg_lifespan,
                num_identified,
            ],
        }
        stats_df = pd.DataFrame(stats_data)

        track_rows = []
        for trk_id, trk in {**self.ocsort.active_trks, **self.ocsort.inactive_trks}.items():
            for age, bbox in trk.observations.items():
                f_num = trk.map_offset(offset=age)
                area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                track_rows.append({
                    'track_id': trk_id,
                    'frame': f_num,
                    'area': area,
                    'is_valid': int(age in trk.valid_observations),
                    'identity': trk.identity,
                })
        track_df = pd.DataFrame(track_rows)

        filename = io_utils.get_unique_filename(
            runtime_data_dir, f'tracking_data_{clip_identifier}.xlsx'
        )
        excel_path = os.path.join(runtime_data_dir, filename)

        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                config_df.to_excel(writer, sheet_name='Configuration', index=False)
                performance_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                stats_df.to_excel(writer, sheet_name='Stats', index=False)
                track_df.to_excel(writer, sheet_name='Tracking Data', index=False)
            logger.info(f'Saved tracking runtime data to {excel_path}')
        except Exception as e:
            logger.error(f'Failed to save Excel file: {e}')

    def save_state(self):
        logger.info('Saving pipeline state...')

        file_prefix = self.video_file.split('.')[0]
        save_path = os.path.join(
            self.output_dir, f'{file_prefix}_tracking_pipeline.pkl'
        )

        log_utils.press_stopwatch(self, 'pkl_io')
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        
        if self.prior_pkl:
            if (
                os.path.exists(self.prior_pkl_path) and
                os.path.isfile(self.prior_pkl_path)
            ):
                os.remove(self.prior_pkl_path)
        log_utils.press_stopwatch(self, 'pkl_io')

        logger.info(f'{len(self.ocsort.active_trks.keys())} tracks saved to be continued')

    def generate_output_vid(
            self,
            trk_video_data: Optional[dict] = None,
            face_data: Optional[pd.DataFrame] = None,
            all_detections: bool = False,
    ):
        trk_video_data = trk_video_data or self.trk_video_data

        logger.info('Generating output video...')

        file_prefix = self.video_file.split('.')[0]
        output_filename = f'{file_prefix}_tracker_output.mp4'

        output_vid_dir = os.path.join(self.output_dir, 'videos/')
        output_vid_path = io_utils.get_unique_path(output_vid_dir, output_filename)

        cap = cv2.VideoCapture(self.video_path)

        font = cv2.FONT_HERSHEY_PLAIN
        text_scale = 3
        text_thickness = 3
        line_thickness = 3

        def _get_color(idx):
            idx *= 3
            return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

        output_dims = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_vid_path, fourcc, self.fps, output_dims)

        name_cache = {}

        f_num = 0
        while f_num < self.f_total:
            ret, frame = cap.read()
            if not ret:
                break

            global_f_num = self.f_start + f_num

            if all_detections == True:
                for det in self.detections.get(global_f_num, []):
                    scaled_det = det[:4] * (1.0 / self.ocsort.scale)
                    x1, y1, x2, y2 = map(int, scaled_det)
                    cv2.rectangle(
                        frame,
                        (x1 - 2, y1 -2),
                        (x2 + 2, y2 + 2),
                        (255, 255, 255),
                        text_thickness
                    )
            
            if face_data is not None:
                frame_face_data = face_data.loc[face_data['f'] == global_f_num]
                if not frame_face_data.empty:
                    cv2.putText(
                        frame,
                        f'FACE(S) PRESENT',
                        (1920, 20),
                        font,
                        text_scale,
                        (0, 0, 255),
                        text_thickness,
                    )
                    frame_face_data = (
                        frame_face_data
                        .sort_values('distance')
                        .groupby(['f', 'x', 'y', 'w', 'h'], as_index=False)
                        .first()
                    )
                for _, row in frame_face_data.iterrows():
                    x1, y1, w, h = map(int, row[['x', 'y', 'w', 'h']])
                    x2, y2 = x1 + w, y1 + h
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 90, 5), line_thickness)
                    cv2.putText(
                        frame,
                        row['name'],
                        (x1, y1 - 5),
                        font,
                        text_scale,
                        (200, 90, 5),
                        text_thickness,
                    )

            trk_data = trk_video_data.get(global_f_num, {})
            for box, track_id in zip(
                trk_data.get('online_boxes', []),
                trk_data.get('online_ids', [])
            ):
                x1, y1, w, h = map(int, box)
                xyxy = (x1, y1, x1 + w, y1 + h)
                color = _get_color(track_id)
                cv2.rectangle(frame, xyxy[:2], xyxy[2:], color, line_thickness)
                cv2.putText(
                    frame,
                    str(track_id),
                    (x1, y1 - 5),
                    font,
                    text_scale,
                    color,
                    text_thickness,
                )

                if track_id in name_cache:
                    full_name = name_cache[track_id]
                else:
                    full_name = 'Unidentified'
                    trk_obj = (
                        self.ocsort.active_trks.get(track_id) or
                        self.ocsort.inactive_trks.get(track_id)
                    )
                    if trk_obj and hasattr(trk_obj, 'identity'):
                        first, last = io_utils.lookup_name(trk_obj.identity)
                        if first or last:
                            full_name = f'{first} {last}'.strip()
                    name_cache[track_id] = full_name
    
                cv2.putText(
                    frame,
                    full_name,
                    (x1, y1 + h + 20),
                    font,
                    text_scale,
                    color,
                    text_thickness,
                )

            cv2.putText(
                frame,
                f'Frame: {global_f_num}',
                (5, 20),
                font,
                text_scale,
                (0, 0, 255),
                text_thickness,
            )

            resized_frame = cv2.resize(frame, output_dims)
            out.write(resized_frame)

            f_num += 1

        cap.release()
        out.release()
        logger.info('Output video saved')
