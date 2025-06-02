# standard dependencies
import os
from typing import Union, Optional, Sequence, Tuple
import pickle
import gc
import math
from dataclasses import dataclass

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

# internal dependencies
from models import RetinaFace, CenterFace, ClearFace, FaceNet512
from utilities import io_utils, face_utils, image_utils
from utilities.log_utils import press_stopwatch
from utilities import general_utils as utils


class FaceAnalysis:
    def __init__(
            self,
            id_cutoff: float = 0.8,
            device: torch.device = None,
            centerface_cfg: dict = {},
            clearface_cfg: Optional[dict] = None,
            facenet_cfg: dict = {},
        ):
        self.device = device or utils.get_default_device()

        # PATHS:
        self.project_root = io_utils.get_project_root()
        self.input_dir = os.path.join(self.project_root, 'files/input/')
        self.output_dir = os.path.join(self.project_root, 'files/output/')

        self.face_dir = os.path.join(self.input_dir, 'faces/')
        self.db_path = os.path.join(self.project_root, 'files/', 'data.db')

        # MODELS:
        self.centerface = CenterFace(device=self.device, **centerface_cfg)
        self.facenet512 = FaceNet512(device=self.device, **facenet_cfg)

        if clearface_cfg is None:
            self.enhance_faces = False
        else:
            self.enhance_faces = True
            self.clearface = ClearFace(device=self.device, **clearface_cfg)

        self.id_cutoff = id_cutoff

        # PERFORMANCE DATA:
        self.identification_pipeline_time = 0
        self.face_detection_time = 0
        self.face_recognition_time = 0
        self.other_processing_time = 0

        # RESULTS:
        self.face_detections = {}
        self.results = {}
        self.id_matches = {}

    def _prepare_data(
            self,
            db_path,
            expand_percentage: int = 0,
            refresh_database: bool = True,
            enhance: bool = True,
            normalize_face: bool = True,
        ):
        def __find_bulk_embeddings(
                employees: set[str],
                expand_percentage: int = 0,
                enhance: bool = True,
            ) -> list[dict]:

            representations: list[dict] = []
            employee_list = sorted(employees)

            for employee_path in employee_list:
                img = cv2.imread(employee_path)
                if img is None:
                    continue

                img_obj_list = self.detect(
                    img,
                    detector="retinaface",
                    expand_percentage=expand_percentage,
                    enhance=enhance,
                    color_face="bgr",
                    normalize_face=normalize_face,
                )[0]

                if not img_obj_list:
                    representations.append({
                        "identity": employee_path,
                        "hash": image_utils.find_image_hash(employee_path),
                        "embedding": None,
                        "target_x": 0, "target_y": 0,
                        "target_w": 0, "target_h": 0,
                    })
                    continue

                face_imgs, facial_areas, confidences = [], [], []
                for obj in img_obj_list:
                    face_imgs.append(obj["face_img"])
                    facial_areas.append(obj["facial_area"])
                    confidences.append(obj.get("confidence", 0))

                embed_results = self.represent(
                    face_imgs,
                    facial_areas,
                    confidences,
                    postprocess=True,
                )

                for res in embed_results:
                    fa = res["facial_area"]
                    representations.append({
                        "identity": employee_path,
                        "hash": image_utils.find_image_hash(employee_path),
                        "embedding": res["embedding"],
                        "target_x": fa["x"], "target_y": fa["y"],
                        "target_w": fa["w"], "target_h": fa["h"],
                    })

            return representations

        if not os.path.isdir(db_path):
            raise ValueError(f'Passed path {db_path} does not exist!')

        file_parts = [
            'ds', 'model', 'facenet512',
            'detector', 'centerface',
            'expand', str(expand_percentage),
        ]
        file_name = '_'.join(file_parts) + '.pkl'
        file_name = file_name.replace('-', '').lower()

        datastore_path = os.path.join(db_path, file_name)
        representations = []

        # required cols for representations:
        df_cols = {
            'identity',
            'hash',
            'embedding',
            'target_x',
            'target_y',
            'target_w',
            'target_h',
        }

        if not os.path.exists(datastore_path):
            with open(datastore_path, 'wb') as f:
                pickle.dump([], f, pickle.HIGHEST_PROTOCOL)

        with open(datastore_path, 'rb') as f:
            representations = pickle.load(f)

        # check each item of representations list has required keys:
        for i, current_representation in enumerate(representations):
            missing_keys = df_cols - set(current_representation.keys())
            if len(missing_keys) > 0:
                raise ValueError(
                    f'{i}-th item does not have some required keys - {missing_keys}.'
                    f'Consider to delete {datastore_path}'
                )

        # Get the list of images on storage:
        storage_images = set(image_utils.yield_images(path=db_path))

        if len(storage_images) == 0 and refresh_database is True:
            raise ValueError(f'No item found in {db_path}')
        if len(representations) == 0 and refresh_database is False:
            raise ValueError(f'Nothing is found in {datastore_path}')

        must_save_pickle = False
        new_images, old_images, replaced_images = set(), set(), set()

        # enforce data consistency amongst on disk images and pickle file:
        if refresh_database:
            pickled_images = {
                representation['identity'] for representation in representations
            }

            new_images = storage_images - pickled_images  # images added to storage
            old_images = pickled_images - storage_images  # images removed from storage

            # determine any replaced images:
            for current_representation in representations:
                identity = current_representation['identity']
                if identity in old_images:
                    continue
                alpha_hash = current_representation['hash']
                beta_hash = image_utils.find_image_hash(identity)
                if alpha_hash != beta_hash:
                    replaced_images.add(identity)

        # Append replaced images into both old and new images. These will be dropped and re-added.
        new_images.update(replaced_images)
        old_images.update(replaced_images)

        # remove old images:
        if len(old_images) > 0:
            representations = [rep for rep in representations if rep['identity'] not in old_images]
            must_save_pickle = True

        # find representations for new images:
        if len(new_images) > 0:
            if not hasattr(self, 'retinaface'):
                self.retinaface = RetinaFace(device=self.device)
            representations += __find_bulk_embeddings(
                employees=new_images,
                expand_percentage=expand_percentage,
                enhance=enhance,
            )
            must_save_pickle = True

        if must_save_pickle:
            with open(datastore_path, 'wb') as f:
                pickle.dump(representations, f, pickle.HIGHEST_PROTOCOL)

        return representations

    def detect(
            self,
            imgs: Union[np.ndarray, list[np.ndarray]],
            detector: str = 'centerface',
            expand_percentage: int = 0,
            enhance: bool = True,
            color_face: str = 'rgb',
            normalize_face: bool = True,
        ) -> list[list[dict]]:
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        per_image_resp_objs = []
        args_template = {
            'expand_percentage': expand_percentage,
        }

        if detector == 'skip':
            for img in imgs:
                img_resp = []

                height, width = img.shape[:2]
                base_region = FacialAreaRegion(x=0, y=0, w=width, h=height, confidence=0)
                face_obj = DetectedFace(img, facial_area=base_region, confidence=0)

                args_ = {
                    'color_face': color_face,
                    'width': width,
                    'height': height,
                    'normalize_face': normalize_face,
                }
                resp_obj = face_utils.format_response(face_obj, **args_)
                img_resp.append(resp_obj)

                per_image_resp_objs.append(img_resp)
        else:
            press_stopwatch(self, 'face_detection_time')
            if detector == 'centerface':
                all_facial_areas = self.centerface.detect_faces(imgs)
            elif detector == 'retinaface':
                all_facial_areas = [self.retinaface.detect_faces(imgs[0])]
            press_stopwatch(self, 'face_detection_time')
        
            press_stopwatch(self, 'other_processing_time')
            for img_idx, (img, facial_areas) in enumerate(zip(imgs, all_facial_areas)):
                height, width = img.shape[:2]

                args_ = args_template.copy()
                args_['width_border'] = int(0.5 * width)
                args_['height_border'] = int(0.5 * height)

                format_args = {
                    'color_face': color_face,
                    'width': width,
                    'height': height,
                    'normalize_face': normalize_face,
                }

                img_resp = []
                for facial_area in facial_areas:
                    facial_area, face_img = face_utils.adjust_and_extract(
                        facial_area, img, **args_
                    )
                    if enhance:
                        face_img = self.enhance(face_img, is_rgb=True)

                    face_obj = DetectedFace(
                        img=face_img,
                        facial_area=facial_area,
                        confidence=facial_area.confidence
                    )
                    resp_obj = face_utils.format_response(face_obj, **format_args)
                    img_resp.append(resp_obj)

                per_image_resp_objs.append(img_resp)

            press_stopwatch(self, 'other_processing_time')

        return per_image_resp_objs

    def enhance(
        self,
        img: np.ndarray,
        is_rgb=True,
        output_path=None
    ):
        # Start timing
        enhanced_face = self.clearface.forward(img, is_rgb=is_rgb)
        # End timing
        
        if output_path:
            cv2.imwrite(output_path, enhanced_face)

        return enhanced_face

    def represent(
            self,
            face_imgs: list[np.ndarray],
            facial_areas: list[dict],
            confidences: list[float],
            postprocess: bool = True,
        ) -> list[dict]:
        """
        Args:
            face_imgs (List[np.ndarray]): List of cropped face images (from
                detection pipeline).
            facial_areas (List[dict]): List of {'x', 'y', 'w', 'h'} dicts.
            confidences (List[float]): List of face detection confidence scores.
            model (FaceNet512): Initialized FaceNet512 model.
            postprocess (bool): Whether to convert embeddings to NumPy.

        Returns:
            List[Dict[str, Any]]: List of dicts, each containing:
                - 'embedding': list[float]
                - 'facial_area': dict
                - 'face_confidence': float
        """
        if not (len(face_imgs) == len(facial_areas) == len(confidences)):
            raise ValueError('All input lists must be the same length')

        embeddings = self.facenet512.represent(face_imgs, postprocess=postprocess)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        results = []
        for i, embedding in enumerate(embeddings):
            results.append({
                'embedding': (
                    embedding.tolist() if isinstance(embedding, np.ndarray)
                    else embedding
                ),
                'facial_area': facial_areas[i],
                'face_confidence': confidences[i],
            })

        return results

    def find(
            self,
            imgs: list[np.ndarray],
            db_path: str,
            id_cutoff: Optional[float] = None,
            expand_percentage: int = 0,
            enhance: bool = True,
            refresh_database: bool = True,
        ) -> list[list[pd.DataFrame]]:
        per_image_resp_objs = []

        id_cutoff = id_cutoff or self.id_cutoff

        representations = self._prepare_data(
            db_path,
            expand_percentage=expand_percentage,
            refresh_database=refresh_database,
            enhance=enhance,
        )
        if len(representations) == 0:
            return []
        df = pd.DataFrame(representations)
        
        per_image_objs = self.detect(
            imgs,
            enhance=enhance,
            expand_percentage=expand_percentage
        )
        for source_objs in per_image_objs:
            resp_obj = []
            if not source_objs:
                resp_obj.append(pd.DataFrame)
                per_image_resp_objs.append(resp_obj)
                continue
            face_imgs = [obj['face_img'] for obj in source_objs]
            facial_areas = [obj['facial_area'] for obj in source_objs]
            confidences = [obj['confidence'] for obj in source_objs]

            press_stopwatch(self, 'face_recognition_time')
            target_embedding_objs = self.represent(
                face_imgs,
                facial_areas,
                confidences,
                postprocess=False,
            )
            press_stopwatch(self, 'face_recognition_time')

            press_stopwatch(self, 'other_processing_time')
            source_embeddings = np.stack(df['embedding'].tolist())
            target_embeddings = np.stack(
                [obj['embedding'] for obj in target_embedding_objs]
            )
            source_tensor = torch.tensor(source_embeddings).to(self.device)
            target_tensor = torch.tensor(target_embeddings).to(self.device)
            source_tensor = F.normalize(source_tensor, p=2, dim=1)
            target_tensor = F.normalize(target_tensor, p=2, dim=1)

            similarity = torch.mm(target_tensor, source_tensor.T)
            distance_matrix = 1 - similarity

            for i, embedding_obj in enumerate(target_embedding_objs):
                face_region = embedding_obj['facial_area']

                result_df = df.copy()
                result_df['x'] = face_region['x']
                result_df['y'] = face_region['y']
                result_df['w'] = face_region['w']
                result_df['h'] = face_region['h']

                distances = distance_matrix[i].detach().cpu().numpy().tolist()

                result_df['distance'] = distances

                result_df = result_df.drop(columns=['embedding'])
                result_df = result_df[result_df['distance'] <= id_cutoff]
                result_df = (
                    result_df.sort_values(by=['distance'], ascending=True)
                    .reset_index(drop=True)
                )
                resp_obj.append(result_df)
            per_image_resp_objs.append(resp_obj)

        press_stopwatch(self, 'other_processing_time')

        return per_image_resp_objs

    def identify_faces(
            self,
            img: np.ndarray,
            regions: Optional[Sequence] = None,
            id_cutoff: Optional[float] = None,
            align: bool = False,
            expand_percentage: int = 0,
            enhance: Optional[bool] = None,
            db_path: Optional[str] = None,
        ) -> list[pd.DataFrame]:
        def _postprocess_output(all_face_dfs):
            '''
            - Adds employee names, UUIDs, and designations to the dataframes.
            - Filters redundant results by retaining only the lowest distance
                rows in cases where the same employee is predicted multiple
                times for the same image.
            - Drops irrelevant columns.
            '''
            press_stopwatch(self, 'other_processing_time')
    
            drop_cols = ['target_x', 'target_y', 'target_w', 'target_h']

            filtered_face_dfs = []

            for df in all_face_dfs:
                validated_drop_cols = [c for c in drop_cols if c in df.columns]
                df = df.drop(validated_drop_cols, axis=1)
                
                if not df.empty:
                    results = io_utils.lookup_identities(df['identity'])

                    df[['identity', 'name', 'designation']] = pd.DataFrame(
                        [(result[1], f'{result[3]}_{result[4]}', result[5])
                        for result in results]
                    )
                    df = df.loc[df.groupby('identity')['distance'].idxmin()]
                    filtered_face_dfs.append(df)
    
                else:
                    filtered_face_dfs.append(df)
                    continue
            
            press_stopwatch(self, 'other_processing_time')

            return filtered_face_dfs

        if enhance is None:
            enhance = self.enhance_faces
        elif enhance == True and (not hasattr(self, 'clearface')):
            self.clearface = ClearFace(device=self.device)

        db_path = db_path or self.face_dir
        id_cutoff = id_cutoff or self.id_cutoff

        press_stopwatch(self, 'identification_pipeline_time')

        if not regions:
            batch_imgs = [img]
            kept_regions = []
        else:
            batch_imgs = []
            kept_regions = []
            for region in regions:
                crop = utils.crop_region(img, region)

                if crop is None or crop.size == 0:
                    continue
                batch_imgs.append(crop)
                kept_regions.append(region)
            if not batch_imgs:
                return []

        per_image_face_dfs = self.find(
            imgs=batch_imgs,
            id_cutoff=id_cutoff,
            expand_percentage=expand_percentage,
            enhance=enhance,
            db_path=db_path,
        )

        all_face_dfs = []
        for region_dfs, region in zip(per_image_face_dfs, kept_regions):
            for df in region_dfs:
                if not df.empty:
                    df[['x', 'y']] = df.apply(
                        lambda row: utils.apply_offset(
                            (row['x'], row['y']), region
                        ),
                        axis=1,
                        result_type='expand',
                    )
                all_face_dfs.append(df)

        all_face_dfs = _postprocess_output(all_face_dfs)

        press_stopwatch(self, 'identification_pipeline_time')
        
        return all_face_dfs

    def consolidate_face_data(
            self, face_data: dict[list[pd.DataFrame]]
        ) -> pd.DataFrame:
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

    def save_runtime_data(self):
        filename = os.path.join(self.output_dir, 'faceiq_data.xlsx')
    
        detection_data = []
        for i, detections in self.face_detections.items():
            identifications = self.id_matches.get(
                i, [{'name': '', 'distance': 1.0}] * len(detections)
            )
            for i_f, det in enumerate(detections):
                area = math.prod((det.w, det.h))

                name = identifications[i_f]['name']
                distance = identifications[i_f]['distance']

                detection_data.append({
                    'idx': i,
                    'name': name,
                    'distance': distance,
                    'a': area,
                    'c': det.confidence,
                    'x': det.x,
                    'y': det.y,
                    'w': det.w,
                    'h': det.h,
                })

        detections_df = pd.DataFrame(detection_data)

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            detections_df.to_excel(writer, sheet_name='Detections', index=False)

    def visualize_identifications(self, image, identifications, output_path: str = None):
        image = image.copy()
        color = (245, 104, 17)

        for face_df in identifications:
            if face_df.empty:
                continue
    
            best_match = face_df.loc[face_df['distance'].idxmin()]

            x, y, w, h = best_match[['x', 'y', 'w', 'h']]
            x1, y1, x2, y2 = utils.xywh_xyxy([x, y, w, h])

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if output_path:
            cv2.imwrite(output_path, image)

        return image


# =============================================================================
#                        - FACIAL DATA STRUCTURES -
# -----------------------------------------------------------------------------


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
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    nose: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None


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