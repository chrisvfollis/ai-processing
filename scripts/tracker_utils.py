import cv2
import cv2.legacy as cv2l
import numpy as np
from utilities import i_over_u
import math


def restrain_boxes(coordinates, image_size=[1920, 1080]):
    img_width, img_height = image_size

    # Restrain width and height to not exceed the dimensions of
    # the image:
    coordinates[2] = min(img_width, coordinates[2])
    coordinates[3] = min(img_height, coordinates[3])

    # Restrain centroids so that box can go no further than right
    # outside of the frame.
    half_width = coordinates[2] / 2
    half_height = coordinates[3] / 2

    coordinates[0] = min((img_width + half_width), coordinates[0])
    coordinates[0] = max((0 - half_width), coordinates[0])

    coordinates[1] = min((img_height + half_height), coordinates[1])
    coordinates[1] = max((0 - half_height), coordinates[1])

    return coordinates


def init_kf_args(cntr=[960, 540], wh=[200, 200], vel=[0, 0, 0, 0],
                 accvar=[10, 10, 20, 20], mvar=[5, 5, 5, 5],
                 evel=[25, 25, 25, 25], ewh=[30, 30], dt=1.0):
    
    '''
    -------------- MODEL PARAMETERS --------------
    accvar — higher magnitudes = greater process noise, so more weight on
             incoming measurements relative to the model predictions. This
             results in a greater impact on subsequent predictions.
    mvar — higher magnitudes = greater measurement noise (the R matrix), so more weight on
           the filter's predictions relative to the incoming measurements. mvar essentially
           represents what you expect the typical measurement error in pixels to be is, squared.
    
    --------------- INITIAL VALUES ---------------
    cntr — initial centroid
    wh — initial width and height
    vel — initial velocity
    evel — initial estimate velocity uncertainty
    ewh — initial estimate width and height uncertainty
    '''

    pos_var = (dt**4)/4
    vel_var = (dt**2)

    x_acc_var, y_acc_var, w_acc_var, h_acc_var = accvar

    # Q values:
    x_pvar = pos_var * x_acc_var
    y_pvar = pos_var * y_acc_var
    w_pvar = pos_var * w_acc_var
    h_pvar = pos_var * h_acc_var

    x_vvar = vel_var * x_acc_var
    y_vvar = vel_var * y_acc_var
    w_vvar = vel_var * w_acc_var
    h_vvar = vel_var * h_acc_var
    
    F = np.array([
        [1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, dt, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dt/8, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, dt/8],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])

    Q = np.array([
        [x_pvar, 0, 0, 0, 0, 0, 0, 0],
        [0, y_pvar, 0, 0, 0, 0, 0, 0],
        [0, 0, w_pvar, 0, 0, 0, 0, 0],
        [0, 0, 0, h_pvar, 0, 0, 0, 0],
        [0, 0, 0, 0, x_vvar, 0, 0, 0],
        [0, 0, 0, 0, 0, y_vvar, 0, 0],
        [0, 0, 0, 0, 0, 0, w_vvar, 0],
        [0, 0, 0, 0, 0, 0, 0, h_vvar]
        ])
    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
        ])
    R = np.diag(mvar)
    cntr_x, cntr_y = cntr
    vel_x, vel_y, vel_w, vel_h = vel
    w, h = wh
    x_init = np.array([cntr_x, cntr_y, w, h, vel_x, vel_y, vel_w, vel_h])
    P_init = np.diag([mvar[0], mvar[1], ewh[0], ewh[1], evel[0], evel[1],
                      evel[2], evel[3]])

    return F, Q, H, R, x_init, P_init


class KalmanFilter:
    def __init__(self, frame, F, Q, H, R, x_init, P_init, B=None, u=None):
        self.F = F  # State transition matrix
        self.Q = Q  # Process noise covariance
        self.H = H  # Measurement translation matrix
        self.R = R  # Measurement noise covariance
        self.x = x_init  # State vector
        self.P = P_init  # Estimate uncertainty
        self.I = np.eye(F.shape[0])  # Identity matrix

        self.states = {frame: x_init}

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x = restrain_boxes(self.x)

    def update(self, Z):
        Y = Z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ Y
        self.P = (self.I - K @ self.H) @ self.P
    
    def add_state(self, new_state, frame_number):
        self.states[frame_number] = new_state


class Track(KalmanFilter):
    def __init__(self, detection, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.detections = {args[0]: detection}
        self.embeddings = [embedding]
        self.first_detection_frame = args[0]
        self.last_detection_frame = args[0]

    def add_embedding(self, embedding):
        self.embeddings.append(embedding)
        self.embeddings = self.embeddings[-30:]

    def add_detection(self, new_detection, frame_number):
        self.detections[frame_number] = new_detection
        self.last_detection_frame = frame_number

