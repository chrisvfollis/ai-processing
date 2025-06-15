# standard dependencies
import os
import pickle
from typing import Optional

# 3rd-party dependencies
import pandas as pd

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
                logger.info(f'{progress}%')
        
            self.f_num += 1

        log_utils.press_stopwatch(self, 'primary_run_time')
        return self.ocsort.active_trks, self.ocsort.inactive_trks

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

    def generate_output_vid(self, trk_video_data):
        cap = cv2.VideoCapture(self.video_path)

        f_num = 0
