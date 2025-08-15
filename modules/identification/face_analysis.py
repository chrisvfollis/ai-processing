# standard dependencies
import os
from typing import Optional, Sequence
import pickle
import math

# 3rd-party dependencies
import cv2
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

# internal dependencies
from utilities import utils, io_utils, image_utils, log_utils
from utilities.log_utils import press_stopwatch
from modules.identification.data_structures import DetectedFace, FacialAreaRegion
from modules.identification import face_utils
from modules.spatial import bboxes


logger = log_utils.get_logger(__name__)


class FaceAnalysis:
    def __init__(
            self,
            id_cutoff: float = 0.9,
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

        self.face_datastore = os.path.join(self.input_dir, 'faces/')

        # MODELS:
        from models import CenterFace, FaceNet
        self.centerface = CenterFace(device=self.device, **centerface_cfg)
        self.facenet = FaceNet(device=self.device, **facenet_cfg)

        if clearface_cfg is None:
            self.enhance_faces = False
        else:
            from models import ClearFace
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

        # FACE DATASTORE CACHING:
        self.datastore_path = None
        self.cache_key = None
        self.faces_df = None
        self.faces_tensor = None

        self.reconcile_cache(self.face_datastore, refresh=True)

    def prepare_datastore(
            self,
            face_datastore,
            refresh: bool = True,
            enhance: bool = True,
            normalize_face: bool = True,
        ):
        def _find_bulk_embeddings(
                employees: set[str],
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
                    detector='retinaface',
                    enhance=enhance,
                    color_face='bgr',
                    normalize_face=normalize_face,
                )[0]

                if not img_obj_list:
                    representations.append({
                        'identity': employee_path,
                        'hash': image_utils.find_image_hash(employee_path),
                        'embedding': None,
                        'target_x': 0,
                        'target_y': 0,
                        'target_w': 0,
                        'target_h': 0,
                    })
                    continue

                if len(img_obj_list) > 1:
                    img_obj_list.sort(
                        key=lambda obj: obj['confidence'],
                        reverse=True
                    )
                    img_obj_list = [img_obj_list[0]]

                face_imgs, facial_areas, confidences = [], [], []
                for obj in img_obj_list:
                    fa_dict = obj['facial_area']
                    detection = FacialAreaRegion(**fa_dict, confidence=obj['confidence'])
                    aligned_detection, aligned_img = face_utils.adjust_and_extract(
                        detection=detection,
                        source_img=img,
                        align=True,
                    )

                    face_imgs.append(aligned_img)
                    facial_areas.append({
                        'x': aligned_detection.x,
                        'y': aligned_detection.y,
                        'w': aligned_detection.w,
                        'h': aligned_detection.h,
                        'left_eye': aligned_detection.left_eye,
                        'right_eye': aligned_detection.right_eye,
                        'nose': aligned_detection.nose,
                        'mouth_left': aligned_detection.mouth_left,
                        'mouth_right': aligned_detection.mouth_right,
                    })
                    confidences.append(obj['confidence'])
                    # try:
                    #     ref_img = aligned_img
                    #     if ref_img.dtype == np.float32:
                    #         ref_img = (ref_img * 255).clip(0, 255).astype(np.uint8)

                    #     reference_img_name = io_utils.get_unique_path(
                    #         self.output_dir, 'reference_img.jpg'
                    #     )
                    #     cv2.imwrite(reference_img_name, ref_img)
                    # except Exception as e:
                    #     logger.info(e)

                embed_results = self.represent(
                    face_imgs,
                    facial_areas,
                    confidences,
                    postprocess=True,
                )

                for res in embed_results:
                    fa = res['facial_area']
                    representations.append({
                        'identity': employee_path,
                        'hash': image_utils.find_image_hash(employee_path),
                        'embedding': res['embedding'],
                        'target_x': fa['x'],
                        'target_y': fa['y'],
                        'target_w': fa['w'],
                        'target_h': fa['h'],
                    })

            return representations

        if not os.path.isdir(face_datastore):
            raise ValueError(f'Passed path {face_datastore} does not exist!')

        file_parts = [
            'ds', 'model', 'facenet512',
            'detector', 'centerface',
        ]
        file_name = '_'.join(file_parts) + '.pkl'
        file_name = file_name.replace('-', '').lower()

        datastore_path = os.path.join(face_datastore, file_name)
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
        storage_images = set(image_utils.yield_images(path=face_datastore))

        if len(storage_images) == 0 and refresh is True:
            raise ValueError(f'No item found in {face_datastore}')
        if len(representations) == 0 and refresh is False:
            raise ValueError(f'Nothing is found in {datastore_path}')

        must_save_pickle = False
        new_images, old_images, replaced_images = set(), set(), set()

        # enforce data consistency amongst on disk images and pickle file:
        if refresh:
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
                from models import RetinaFace
                self.retinaface = RetinaFace(device=self.device)
            representations += _find_bulk_embeddings(
                employees=new_images,
                enhance=enhance,
            )
            must_save_pickle = True

        if must_save_pickle:
            with open(datastore_path, 'wb') as f:
                pickle.dump(representations, f, pickle.HIGHEST_PROTOCOL)

        return representations

    def reset_cache(self):
        self.cache_key = None
        self.faces_df = None
        self.faces_tensor = None

    def reconcile_cache(self, face_datastore: str, refresh: bool) -> None:
        file_parts = ['ds', 'model', 'facenet512', 'detector', 'centerface']
        file_name = '_'.join(file_parts).replace('-', '').lower() + '.pkl'
        datastore_path = os.path.join(face_datastore, file_name)

        if refresh:
            _ = self.prepare_datastore(face_datastore, refresh=True)
        
        try:
            stat = os.stat(datastore_path)
        except FileNotFoundError:
            raise ValueError(f'Datastore not found {datastore_path}')
        
        cache_key = (stat.st_mtime_ns, stat.st_size)

        # early return if cache is still valid:
        if (
            (self.datastore_path == datastore_path) and
            (self.cache_key == cache_key) and
            (self.faces_df is not None) and (self.faces_tensor is not None)
        ):
            return
            
        with open(datastore_path, 'rb') as f:
            representations = pickle.load(f)

        if not representations:
            raise ValueError(f'No representations in {datastore_path}')

        df = pd.DataFrame(representations)

        mask = df['embedding'].notna()
        df = df[mask].reset_index(drop=True)
        
        embeddings = np.stack(df['embedding'].tolist()).astype(np.float32)
        embeddings = torch.from_numpy(embeddings).to(
            self.device, non_blocking=True
        )

        df = df.drop(columns=['embedding']).reset_index(drop=True)

        self.datastore_path = datastore_path
        self.cache_key      = cache_key
        self.faces_df       = df
        self.faces_tensor   = F.normalize(embeddings, p=2, dim=1)

    def detect(
            self,
            imgs: np.ndarray | list[np.ndarray],
            detector: str = 'centerface',
            enhance: bool = True,
            color_face: str = 'rgb',
            normalize_face: bool = True,
        ) -> list[list[dict]]:
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        per_image_resp_objs = []

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

                format_args = {
                    'color_face': color_face,
                    'width': width,
                    'height': height,
                    'normalize_face': normalize_face,
                }

                img_resp = []
                for facial_area in facial_areas:
                    facial_area, face_img = face_utils.adjust_and_extract(
                        facial_area, img, align=False
                    )
                    if (face_img is None) or (face_img.size == 0):
                        logger.info('Invalid facial area image')
                        continue
        
                    if enhance:
                        face_img = self.enhance(face_img, is_rgb=True)
                        if (face_img is None) or (face_img.size == 0):
                            continue

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

        embeddings = self.facenet.represent(face_imgs, postprocess=postprocess)

        results = []
        for i, embedding in enumerate(embeddings):
            if postprocess:
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().cpu().numpy()
                if isinstance(embedding, np.ndarray):
                    embedding_out = embedding.tolist()
                else:
                    embedding_out = embedding
            else:
                embedding_out = embedding

            results.append({
                'embedding'       : embedding_out,
                'facial_area'     : facial_areas[i],
                'face_confidence' : confidences[i],
            })

        return results

    def find(
            self,
            imgs: list[np.ndarray],
            face_datastore: str,
            id_cutoff: Optional[float] = None,
            enhance: bool = False,
        ) -> list[list[pd.DataFrame]]:
        id_cutoff = id_cutoff or self.id_cutoff

        self.reconcile_cache(face_datastore, refresh=False)

        per_image_resp_objs = []
        per_image_objs = self.detect(imgs, enhance=enhance)

        for source_objs in per_image_objs:
            resp_obj = []

            if not source_objs:
                resp_obj.append(pd.DataFrame())
                per_image_resp_objs.append(resp_obj)
                continue

            face_imgs    = [obj['face_img']    for obj in source_objs]
            facial_areas = [obj['facial_area'] for obj in source_objs]
            confidences  = [obj['confidence']  for obj in source_objs]

            press_stopwatch(self, 'face_recognition_time')
            target_embedding_objs = self.represent(
                face_imgs,
                facial_areas,
                confidences,
                postprocess=False,
            )
            press_stopwatch(self, 'face_recognition_time')

            press_stopwatch(self, 'other_processing_time')
            target_tensor = (
                torch.stack([obj['embedding'] for obj in target_embedding_objs])
                .to(self.device, non_blocking=True)
            )

            # compute cosine sims:
            target_tensor = F.normalize(target_tensor, p=2, dim=1)
            similarity_matrix = torch.mm(target_tensor, self.faces_tensor.T)
            
            similarity_matrix = (
                similarity_matrix.detach().cpu().float().numpy()
            )
            distance_matrix = 1 - similarity_matrix

            for i, embedding_obj in enumerate(target_embedding_objs):
                result_df = self.faces_df.copy(deep=False)
                
                result_df['x']          = embedding_obj['facial_area']['x']
                result_df['y']          = embedding_obj['facial_area']['y']
                result_df['w']          = embedding_obj['facial_area']['w']
                result_df['h']          = embedding_obj['facial_area']['h']

                result_df['confidence'] = embedding_obj['face_confidence']
                result_df['distance']   = distance_matrix[i]
                
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
            enhance: Optional[bool] = None,
            face_datastore: Optional[str] = None,
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
            from models import ClearFace
            self.clearface = ClearFace(device=self.device)

        face_datastore = face_datastore or self.face_datastore
        id_cutoff = id_cutoff or self.id_cutoff

        press_stopwatch(self, 'identification_pipeline_time')

        if not regions:
            batch_imgs = [img]
            kept_regions = []
        else:
            batch_imgs = []
            kept_regions = []
            for region in regions:
                crop = bboxes.crop_region(img, region)

                if crop is None or crop.size == 0:
                    continue
                batch_imgs.append(crop)
                kept_regions.append(region)
        if not batch_imgs:
            press_stopwatch(self, 'identification_pipeline_time')
            return []

        per_image_face_dfs = []
        batch_size = 2
        for i in range(0, len(batch_imgs), batch_size):
            batch_imgs_chunk = batch_imgs[i:i+batch_size]
            result = self.find(
                imgs=batch_imgs_chunk,
                id_cutoff=id_cutoff,
                enhance=enhance,
                face_datastore=face_datastore,
            )
            per_image_face_dfs.extend(result)

        all_face_dfs = []
        for region_dfs, region in zip(per_image_face_dfs, kept_regions):
            for df in region_dfs:
                if not df.empty:
                    df[['x', 'y']] = df.apply(
                        lambda row: bboxes.apply_offset(
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
            self,
            face_data: dict[list[pd.DataFrame]],
            fps: int | float,
            cam_id: int = None,
        ) -> pd.DataFrame:
        merged_dfs = []
        for f_num, dfs in face_data.items():
            for i, df in enumerate(dfs):
                if df.empty:
                    continue
                df = df.copy()
                df['cam_id'] = cam_id
                df['f'] = f_num
                df['s'] = f_num / fps
                df['face_idx'] = i
                merged_dfs.append(df)

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
            x1, y1, x2, y2 = bboxes.xywh_xyxy([x, y, w, h])

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if output_path:
            cv2.imwrite(output_path, image)

        return image
