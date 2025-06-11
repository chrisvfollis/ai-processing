# standard dependencies
import math
from typing import Optional

# 3rd-party dependencies
import numpy as np

# internal dependencies
from modules.oc_sort import association
from .kalmanfilter import KalmanFilterNew as KalmanFilter
import utilities.general_utils as utils


# =============================================================================
#                            - GLOBAL TRACKER -
# -----------------------------------------------------------------------------


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {
    "iou": association.iou_batch,
    "giou": association.giou_batch,
    "ciou": association.ciou_batch,
    "diou": association.diou_batch,
}


class OCSort:
    def __init__(
            self,
            det_thresh,
            max_age=30,
            min_hits=3, 
            iou_threshold=0.3,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            use_byte=False,
            img_dims=(2160, 3840),
            input_dims=(800, 1440),
            aspect_ratio_thresh=1.6,
            min_box_area=100,
        ):
        # PIXEL SPACE TRANSLATION:
        img_h, img_w = img_dims
        input_h, input_w = input_dims
        self.scale = min(input_h / img_h, input_w / img_w)

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.active_trks = {}
        self.inactive_trks = {}
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte
        self.val_cfg = {
            'aspect_ratio_thresh': aspect_ratio_thresh,
            'min_box_area': min_box_area,
        }
        KalmanBoxTracker.next_id = 0

    def update(self, output_results, f_num=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1

        new_dets, trk_data = self._organize_data(output_results)

        dets, dets_indices, dets_second, dets_second_indices = new_dets
        trk_preds, velocities = trk_data[:2]
        last_boxes, k_observations = trk_data[2:]

        trk_id_list = list(self.active_trks.keys())
        """
            First round of association
        """
        assoc_results = association.associate(
            dets,
            trk_preds,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
        )
        matched, unmatched_dets, unmatched_trks = assoc_results

        for m in matched:
            trk_id = trk_id_list[m[1]]
            det_idx = dets_indices[m[0]]
            self.active_trks[trk_id].update(dets[m[0], :], det_idx)

        """
            Second round of associaton
        """
        if self.use_byte and (len(dets_second) > 0) and (unmatched_trks.shape[0] > 0):
            unmatched_trks = self._byte_association(
                dets_second, dets_second_indices, trk_preds, unmatched_trks
            )

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            unmatched_dets, unmatched_trks = self._second_association_rematch(
                dets, dets_indices, unmatched_dets, unmatched_trks, last_boxes
            )

        for m in unmatched_trks:
            trk_id = trk_id_list[m]
            self.active_trks[trk_id].update(None)

        # create and initialise new trackers for unmatched detections
        self._init_new_tracks(unmatched_dets, dets, f_num=f_num)

        bboxes = self._finalize_update(return_bboxes=True)
        if(len(bboxes) <= 0):
            return np.empty((0, 5))
        
        return bboxes

    def _organize_data(self, output_results):
        output_indices = np.arange(output_results.shape[0])

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        bboxes /= self.scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second_indices = output_indices[inds_second]
        dets_second = dets[inds_second]  # detections for second matching
        
        remain_inds = scores > self.det_thresh
        dets_indices = output_indices[remain_inds]
        dets = dets[remain_inds]
        
        new_dets = [dets, dets_indices, dets_second, dets_second_indices]

        # get predicted locations from existing trackers:
        trk_preds = []
        to_keep = {}

        for trk_id, trk in self.active_trks.items():
            pos = trk.predict()[0]
            if np.any(np.isnan(pos)):
                continue  # skip this tracker
            trk_preds.append([pos[0], pos[1], pos[2], pos[3], 0])
            to_keep[trk_id] = trk

        self.active_trks = to_keep
        trk_preds = np.array(trk_preds)

        trk_velocities = np.array([
            trk.velocity if (trk.velocity is not None) else np.array((0, 0))
            for trk in self.active_trks.values()
        ]) 
        last_boxes = np.array([trk.last_observation for trk in self.active_trks.values()])
        k_observations = np.array([
            self._k_previous_obs(trk.observations, trk.age, self.delta_t)
            for trk in self.active_trks.values()
        ])
        trk_data = [trk_preds, trk_velocities, last_boxes, k_observations]

        return new_dets, trk_data

    def _byte_association(self, dets_second, dets_second_indices, trk_preds, unmatched_trks):
        trk_id_list = list(self.active_trks.keys())

        u_trks = trk_preds[unmatched_trks]
        iou_left = self.asso_func(dets_second, u_trks)          # iou between low score detections and unmatched tracks
        iou_left = np.array(iou_left)
        if iou_left.max() > self.iou_threshold:
            """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
            """
            matched_indices = association.linear_assignment(-iou_left)
            to_remove_trk_indices = []
            for m in matched_indices:
                det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                if iou_left[m[0], m[1]] < self.iou_threshold:
                    continue
                trk_id = trk_id_list[trk_ind]
                det_global_idx = dets_second_indices[det_ind]
                self.active_trks[trk_id].update(dets_second[det_ind, :], det_global_idx)
                to_remove_trk_indices.append(trk_ind)
            unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        return unmatched_trks

    def _second_association_rematch(self, dets, dets_indices, unmatched_dets, unmatched_trks, last_boxes):
        trk_id_list = list(self.active_trks.keys())

        left_dets = dets[unmatched_dets]
        left_trks = last_boxes[unmatched_trks]
        iou_left = self.asso_func(left_dets, left_trks)
        iou_left = np.array(iou_left)
        if iou_left.max() > self.iou_threshold:
            """
                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                uniform here for simplicity
            """
            rematched_indices = association.linear_assignment(-iou_left)
            to_remove_det_indices = []
            to_remove_trk_indices = []
            for m in rematched_indices:
                det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                if iou_left[m[0], m[1]] < self.iou_threshold:
                    continue
                trk_id = trk_id_list[trk_ind]
                det_global_idx = dets_indices[det_ind]
                self.active_trks[trk_id].update(dets[det_ind, :], det_global_idx)
                to_remove_det_indices.append(det_ind)
                to_remove_trk_indices.append(trk_ind)
            unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
            unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        return unmatched_dets, unmatched_trks

    def _k_previous_obs(self, observations, cur_age, k):
        if len(observations) == 0:
            return [-1, -1, -1, -1, -1]
        for i in range(k):
            dt = k - i
            if cur_age - dt in observations:
                return observations[cur_age-dt]
        max_age = max(observations.keys())
        return observations[max_age]

    def _init_new_tracks(self, unmatched_dets, dets, f_num=None):
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :],
                delta_t=self.delta_t,
                start=f_num,
                **self.val_cfg
            )
            self.active_trks[trk.id] = trk
    
    def _finalize_update(self, return_bboxes=False):
        bboxes = []
        inactive = []
        for trk_id, trk in self.active_trks.items():
            # remove dead tracklet:
            if (trk.time_since_update > self.max_age):
                self.inactive_trks[trk_id] = trk
                inactive.append(trk_id)
            
            if return_bboxes:
                if trk.time_since_update < 1 and (
                    (trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)
                ):
                    if trk.last_observation is not None:
                        bbox = trk.last_observation[:4]
                    else:
                        bbox = trk.get_state()[0]
                    bboxes.append(np.concatenate((bbox, [trk.id])).reshape(1, -1))

        for trk_id in inactive:
            del self.active_trks[trk_id]

        if return_bboxes:
            return np.concatenate(bboxes)


# =============================================================================
#                           - INDIVIDUAL TRACKS -
# -----------------------------------------------------------------------------


class KalmanBoxTracker:
    next_id = 0
    def __init__(self, bbox, delta_t=3, start=0, aspect_ratio_thresh=1.6, min_box_area=100):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
    
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = utils.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.next_id
        KalmanBoxTracker.next_id += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.start = start
        self.age = 0
        
        self.last_observation = None    # np.ndarray
        self.observations = {}
        self.bbox_indices = {}
        self.valid_observations = []
        self.velocity = None
        self.delta_t = delta_t

        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
    
    def update(self, bbox, idx=None):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is None:
            self.kf.update(bbox)
            return

        if self.last_observation is not None:
            previous_box = None
            for i in range(self.delta_t):
                dt = self.delta_t - i
                if self.age - dt in self.observations:
                    previous_box = self.observations[self.age - dt]
                    break
            if previous_box is None:
                previous_box = self.last_observation

            self.velocity = self._speed_direction(previous_box, bbox)
        
        self.last_observation = bbox
        self.observations[self.age] = bbox
        if idx is not None:
            self.bbox_indices[self.age] = idx

        if self.validate(bbox) == True:
            self.valid_observations.append(self.age)

        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(utils.convert_bbox_to_z(bbox))            

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(utils.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def _speed_direction(self, bbox1, bbox2):
        cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
        cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
        speed = np.array([cy2-cy1, cx2-cx1])
        norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
        return speed / norm

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return utils.convert_x_to_bbox(self.kf.x)

    def validate(self, bbox) -> bool:
        _, _, w, h = utils.xywh_xyxy(bbox[:4], out='xywh')

        valid_ratio = (w / h) <= self.aspect_ratio_thresh
        valid_area = math.prod([w, h]) > self.min_box_area
        
        return valid_ratio and valid_area

    def map_offset(
            self,
            start: Optional[int] = None,
            offset: Optional[int] = None,
        ) -> int:
        start = start if (start is not None) else self.start
        offset = offset if (offset is not None) else self.age

        return start + offset
