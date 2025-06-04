# standard dependencies
import os

# 3rd-party dependencies
pass

# internal dependencies
from utilities import io_utils, log_utils
from utilities import general_utils as utils
from modules import OCSort


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
            asso_func="iou",
            inertia=0.2,
            use_byte=False,
        ):
        # PATHS:
        self.project_root = io_utils.get_project_root()
        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')

        self.video_path = os.path.join(self.input_dir, video_file)

        # VIDEO ATTRIBUTES:
        res, _, fps, f_tot = utils.get_video_info(self.video_path)

        self.resolution = res
        self.fps = fps
        self.f_total = f_tot
        self.progress_interval = f_tot // 4

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
        
        # TIMING ATTRIBUTES:
        self.primary_run_time = 0
        self.persist_time = 0

        # INFERENCE DATA:
        self.detections = detections

    def run(self):
        log_utils.press_stopwatch(self, 'primary_run_time')

        self.f_num = 0
        
        while self.f_num < self.f_total:
            detections = self.detections.get(self.f_num, None)

            self.ocsort.update(detections, self.f_num)

            if self.f_num % self.progress_interval == 0:
                progress = utils.calculate_progress(self.f_num, self.f_total)
                logger.info(f'{progress}%')
        
            self.f_num += 1

        log_utils.press_stopwatch(self, 'primary_run_time')
