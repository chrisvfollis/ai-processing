# standard dependencies
import os
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


logger = log_utils.get_logger(__name__)


class IdentificationPipeline:
    def __init__(
            self,
            track_detections: pd.DataFrame,
            face_data: pd.DataFrame,
            video_file: str,
            embedding_file: str = None
        ):
        # DATA:
        self.trk_detections = track_detections
        self.face_data = face_data

        # PATHS/FILENAMES/ETC:
        self.project_root = io_utils.get_project_root()
        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')

        self.video_file = video_file
        self.video_path = os.path.join(self.input_dir, video_file)

        self.embedding_file = embedding_file or (
            f'{video_file.split(".")[0]}_embeddings.hdf5'
        )
        self.embedding_path = os.path.join(self.output_dir, self.embedding_file)

    def assign_faces(self):
        face_df = self.face_data.loc[self.face_data['f'] == self.f_num]
        face_boxes = (
            face_df[['x', 'y', 'w', 'h']].drop_duplicates()
            .values().tolist()
        )
        person_boxes = []

        for trk_id, trk in self.ocsort.active_trks.items():
            f_idx = trk.frame_mapping[self.f_num]
            bbox = trk.observations[f_idx]
            bbox = utils.xywh_xyxy(bbox, out='xywh')
            person_boxes.append(bbox)
        
        # construct cost matrix

    def reassociate(self):
        pass
