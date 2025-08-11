# standard dependencies
from typing import Optional
from dataclasses import dataclass, asdict

# 3rd-party dependencies
import numpy as np
import pandas as pd

# internal dependencies
pass

@dataclass
class FacialAreaRegion:
    """
    Initialize a Face object.

    Args:
        x (int): The x-coordinate of the top-left corner of the bounding box.
        y (int): The y-coordinate of the top-left corner of the bounding box.
        w (int): The width of the bounding box.
        h (int): The height of the bounding box.
        left_eye (tuple): The coordinates (x, y) of the left eye with respect to
            the person instead of observer. Default is None.
        right_eye (tuple): The coordinates (x, y) of the right eye with respect to
            the person instead of observer. Default is None.
        confidence (float, optional): Confidence score associated with the face detection.
            Default is None.
    """

    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[tuple[int, int]] = None
    right_eye: Optional[tuple[int, int]] = None
    confidence: Optional[float] = None
    nose: Optional[tuple[int, int]] = None
    mouth_right: Optional[tuple[int, int]] = None
    mouth_left: Optional[tuple[int, int]] = None


@dataclass
class DetectedFace:
    """
    Initialize detected face object.

    Args:
        img (np.ndarray): detected face image as numpy array
        facial_area (FacialAreaRegion): detected face's metadata (e.g. bounding box)
        confidence (float): confidence score for face detection
    """

    img: np.ndarray
    facial_area: FacialAreaRegion
    confidence: float


@dataclass
class AssessIdPresenceParams:
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    match_cutoff: float = 0.25
    mismatch_threshold: float = 0.90
    distance_score_weight: float = 0.55
    confidence_weight: float = 0.45
    n_matches: int = 1                 # max ID matches per face detection
    min_score: float = 0.45
    reliability_scale: float = 0.75    # α – scales `score` -> success probability
    fp_rate: float = 0.20              # β – per-detection false positive rate
    presence_prior: float = 0.05       # π – assumed prior for P(identity present)
    bias_score_boundary: float = 0.70
    penalty_biases: tuple[float, float] = (0.50, 1.25)
    decay_window: float = 0.9                      # seconds
    boost_range: tuple[float, float] = (3.0, 5.0)  # seconds (range)
    max_decay: float = 0.6
    max_boost: float = 0.8
    boost_per_neighbor: float = 0.075
    fallback_recall_est: float = 0.60
    presence_thresh: float = 0.55

    def as_dict(self):
        return asdict(self)
