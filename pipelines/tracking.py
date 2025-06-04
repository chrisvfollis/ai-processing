# standard dependencies
import os
import tracemalloc
import pickle
import math
from itertools import permutations, islice
import sys

# 3rd-party dependencies
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import cv2
import torch
import torch.nn.functional as F

# internal dependencies
from utilities import io_utils, log_utils
from utilities import general_utils as utils
from utilities.log_utils import get_logger, press_stopwatch
from modules import OCSort


logger = get_logger(__name__)


class TrackingPipeline:
    def __init__(
            self,
            video_file: str,
            time_prefix: str,
            detections: dict,
            credentials,
            device: torch.device = None,
            yolox_input_size: tuple[int] = (800, 1440),
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

        # GENERAL ATTRIBUTES:
        self.device = device or utils.get_default_device()
        self.credentials = credentials

        # TRACKER:
        self.ocsort = OCSort(
            det_thresh,
            max_age,
            min_hits,
            iou_threshold,
            delta_t,
            asso_func,
            inertia,
            use_byte,
        )
        
        # VIDEO ATTRIBUTES:
        self.video_file = video_file
        self.video_path = os.path.join(self.input_dir, video_file)

        video_info = utils.get_video_info(self.video_path)

        self.resolution = video_info[0]
        self.frame_diag = video_info[1]
        self.fps = video_info[2]
        self.total_frames = video_info[3]

        self.img_hw = (self.resolution[1], self.resolution[0])
        self.progress_interval = self.total_frames // 4

        self.start_time = utils.frame_timestamp(time_prefix)
        self.end_time = utils.frame_timestamp(
            time_prefix, self.total_frames, self.fps
        )
        self.f_num = 0

        # TIMING ATTRIBUTES:
        self.primary_run_time = 0
        self.persist_time = 0
        
        self.pkl_io = 0
        self.read_embeddings = 0
        self.tensor_conversion = 0

        # PARAMETERS:
        self.yolox_input_size = yolox_input_size
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area

        # INFERENCE DATA:
        self.detections = detections

    def run(self):
        while self.f_num < self.total_frames:
            person_dets = self.detections.get(self.f_num, [])
            if not person_dets:
                self.f_num += 1
                continue
            
            self.ocsort.update(
                person_dets, self.img_hw, self.yolox_input_size, self.f_num
            )
