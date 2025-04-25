# standard dependencies
import os
from typing import Any, Dict, Set, List, IO, Union, Optional
import pickle
import time
import gc
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch
from deepface.commons import image_utils
from deepface.modules import detection, representation, verification, recognition
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion

# internal dependencies
from models.centerface import CenterFace
from models.clearface import ClearFace
from utilities import face_utils
from utilities import io_utils
from utilities import general_utils as utils
from utilities.logging_utils import press_stopwatch


class FaceIq:
    def __init__(
            self,
            recognition_model,
            detection_model,
            id_cutoff=0.8,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            face_dir='../files/input/faces',
            db_path='../files/data.db',
            detector_weights='../models/weights/centerface.pth',
            enhancer_weights='../models/weights/clearface/90000_G.pth',
            save_data=False
        ):

        self.rec_model_name = recognition_model
        self.det_model_name = detection_model

        self.face_detector = CenterFace(
            device=device, weights_path=detector_weights
        )
        self.face_enhancer = ClearFace(
            device=device, weights_path=enhancer_weights
        )

        self.id_cutoff = id_cutoff

        self.face_dir = face_dir
        self.db_path = db_path

        self.identification_pipeline_time = 0
        self.face_detection_time = 0
        self.face_recognition_time = 0
        self.other_processing_time = 0

        self.save_data = save_data

        if self.save_data:
            self.i = 0
            self.i_f = 0

            self.regions = {}
            self.face_detections = {}
            self.face_objs = {}
            self.source_objs = {}
            self.det_recognition_dfs = {}

            self.results = {}
            self.id_matches = {}

    def prepare_data(
        self,
        db_path,
        model_name,
        detector_backend,
        align,
        expand_percentage,
        normalization,
        refresh_database
    ):
        def __find_bulk_embeddings(
            employees: Set[str],
            model_name: str = 'VGG-Face',
            detector_backend: str = 'opencv',
            align: bool = False,
            expand_percentage: int = 0,
            normalization: str = 'base',
        ) -> List[Dict['str', Any]]:
            
            representations = []
            for employee in employees:
                file_hash = image_utils.find_image_hash(employee)

                try:
                    img_objs = detection.extract_faces(
                        img_path=employee,
                        detector_backend='retinaface',
                        grayscale=False,
                        enforce_detection=False,
                        align=False,
                        expand_percentage=expand_percentage,
                        color_face='bgr'  # `represent` expects images in bgr format.
                    )
                except ValueError as err:
                    print(f'Exception while extracting faces from {employee}: {str(err)}')
                    img_objs = []

                if len(img_objs) == 0:
                    representations.append(
                        {
                            'identity': employee,
                            'hash': file_hash,
                            'embedding': None,
                            'target_x': 0,
                            'target_y': 0,
                            'target_w': 0,
                            'target_h': 0,
                        }
                    )
                else:
                    for i, img_obj in enumerate(img_objs):
                        img_content = img_obj['face']
                        img_region = img_obj['facial_area']

                        img_to_save = img_content
                        if img_to_save.dtype == np.float32 or img_to_save.max() <= 1.0:
                            img_to_save = (img_to_save * 255).astype(np.uint8)
                        cv2.imwrite(f'{employee.split("/")[-1].split(".")[0]}_{i}.jpg', img_to_save)

                        embedding_obj = representation.represent(
                            img_path=img_content,
                            model_name=model_name,
                            detector_backend='skip',
                            align=align,
                            normalization=normalization,
                        )
                        img_representation = embedding_obj[0]['embedding']
                        representations.append(
                            {
                                'identity': employee,
                                'hash': file_hash,
                                'embedding': img_representation,
                                'target_x': img_region['x'],
                                'target_y': img_region['y'],
                                'target_w': img_region['w'],
                                'target_h': img_region['h'],
                            }
                        )

            return representations

        if not os.path.isdir(db_path):
            raise ValueError(f'Passed path {db_path} does not exist!')

        file_parts = [
            'ds', 'model', model_name,
            'detector', detector_backend,
            'aligned' if align else 'unaligned',
            'normalization', normalization,
            'expand', str(expand_percentage),
        ]

        file_name = '_'.join(file_parts) + '.pkl'
        file_name = file_name.replace('-', '').lower()

        datastore_path = os.path.join(db_path, file_name)
        representations = []

        # Required columns for representations
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

        # check each item of representations list has required keys
        for i, current_representation in enumerate(representations):
            missing_keys = df_cols - set(current_representation.keys())
            if len(missing_keys) > 0:
                raise ValueError(
                    f'{i}-th item does not have some required keys - {missing_keys}.'
                    f'Consider to delete {datastore_path}'
                )

        # Get the list of images on storage
        storage_images = set(image_utils.yield_images(path=db_path))

        if len(storage_images) == 0 and refresh_database is True:
            raise ValueError(f'No item found in {db_path}')
        if len(representations) == 0 and refresh_database is False:
            raise ValueError(f'Nothing is found in {datastore_path}')

        must_save_pickle = False
        new_images, old_images, replaced_images = set(), set(), set()

        # Enforce data consistency amongst on disk images and pickle file
        if refresh_database:
            pickled_images = {
                representation['identity'] for representation in representations
            }

            new_images = storage_images - pickled_images  # images added to storage
            old_images = pickled_images - storage_images  # images removed from storage

            # Determine any replaced images
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

        # Remove old images
        if len(old_images) > 0:
            representations = [rep for rep in representations if rep['identity'] not in old_images]
            must_save_pickle = True

        # Find representations for new images
        if len(new_images) > 0:
            representations += __find_bulk_embeddings(
                employees=new_images,
                model_name=model_name,
                detector_backend=detector_backend,
                align=align,
                expand_percentage=expand_percentage,
                normalization=normalization,
            )
            must_save_pickle = True

        if must_save_pickle:
            with open(datastore_path, 'wb') as f:
                pickle.dump(representations, f, pickle.HIGHEST_PROTOCOL)

        return representations

    def identify_faces(self, img, id_cutoff=None, regions=None, enhance=False, config=None):
        def _postprocess_output(all_face_dfs):
            press_stopwatch(self, 'other_processing_time')
    
            filtered_face_dfs = []
    
            for df in all_face_dfs:
                if not df.empty:
                    results = io_utils.lookup_identities(
                        df['identity'], db_path=self.db_path
                    )

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

        id_cutoff = id_cutoff or self.id_cutoff
        config = config or {
            'db_path': self.face_dir,
            'model_name': self.rec_model_name,
            'detector_backend': self.det_model_name,
            'threshold': id_cutoff,
            'batched': False,
            'align': False
        }

        all_face_dfs = []
        
        if self.save_data:
            self.i_f = 0

        press_stopwatch(self, 'identification_pipeline_time')

        if not regions:
            face_dfs = self.find(img_path=img, enhance=enhance, **config)
            for df in face_dfs:
                df = utils.reformat_face_df(df)

                all_face_dfs.append(df)

        else:
            if self.save_data:
                self.regions.setdefault(self.i, []).extend(regions)

            for region in regions:
                img_crop = utils.crop_region(img, region)
                local_face_dfs = self.find(
                    img_path=img_crop,
                    enhance=enhance,
                    **config
                )

                del img_crop
                gc.collect()

                for df in local_face_dfs:
                    df = utils.reformat_face_df(df)
                    if df.empty:
                        all_face_dfs.append(df)
                        continue
    
                    df[['x', 'y']] = df.apply(
                        lambda row:
                        utils.apply_offset(
                            (row['x'], row['y']), region
                        ),
                        axis=1
                    ).apply(pd.Series)

                    all_face_dfs.append(df)

        press_stopwatch(self, 'identification_pipeline_time')
        
        results = _postprocess_output(all_face_dfs)

        if self.save_data:
            i_f = -1
            output_dir = '../files/output'
            for face_df in results:
                i_f += 1
                
                if face_df.empty:
                    continue

                best_match = face_df.loc[face_df['distance'].idxmin()]

                name = best_match['name']
                distance = best_match['distance']

                match_dict = {'name': name, 'distance': distance}

                self.id_matches.setdefault(self.i, []).append(match_dict)

                filename = f'{self.i}_{i_f}_{name}_detection.png'
                output_path = os.path.join(output_dir, filename)

                x, y, w, h = best_match[['x', 'y', 'w', 'h']]
                x1, y1, x2, y2 = utils.xywh_xyxy([x, y, w, h])

                face_img = img[y1:y2,x1:x2]
                face_img = cv2.resize(face_img, (w * 10, h * 10))

                cv2.imwrite(output_path, face_img)

        return results

    def recognize(
            self, img: np.ndarray, id_cutoff: Optional[float] = None
        ) -> pd.DataFrame:
        '''
        Only does recognition — should be used on cropped face detection images. 
        '''

        if not os.path.isdir(self.face_dir):
            raise ValueError(f'Face DB path {self.face_dir} does not exist')

        file_name = '_'.join([
            'ds', 'model', self.rec_model_name,
            'detector', self.det_model_name,
            'unaligned',
            'normalization', 'base',
            'expand', '0'
        ]).replace('-', '').lower() + '.pkl'

        datastore_path = os.path.join(self.face_dir, file_name)
        if not os.path.exists(datastore_path):
            raise FileNotFoundError(f'Embedding cache {datastore_path} not found')

        with open(datastore_path, 'rb') as f:
            representations = pickle.load(f)

        df = pd.DataFrame(representations)
        if df.empty:
            return pd.DataFrame()
        print('Representations dataframe:')
        print(df)
        print(f'Columns: {list(df.columns)}')

        press_stopwatch(self, 'face_recognition_time')
        embedding_obj = representation.represent(
            img_path=img,
            model_name=self.rec_model_name,
            detector_backend='skip',
            align=False,
            normalization='base',
        )
        press_stopwatch(self, 'face_recognition_time')

        target_embedding = embedding_obj[0]['embedding']

        distances = []
        for _, row in df.iterrows():
            src_embedding = row['embedding']
            if src_embedding is None:
                distances.append(float('inf'))
                continue

            distances.append(
                verification.find_distance(
                    src_embedding, target_embedding, 'cosine'
                )
            )
        
        print('Distances:')
        print(distances)
        
        df['x'] = [0] * len(distances)
        df['y'] = [0] * len(distances)
        df['w'] = [img.shape[1]] * len(distances)
        df['h'] = [img.shape[0]] * len(distances)

        df['distance'] = distances
        target_threshold = id_cutoff or verification.find_threshold(
            self.rec_model_name, 'cosine'
        )
        df['threshold'] = target_threshold

        print('Distances dataframe:')
        print(df)

        df = df[df['distance'] <= target_threshold]
        df = df.sort_values(by='distance', ascending=True).reset_index(drop=True)

        df = utils.reformat_face_df(df)

        print('Formatted dataframe:')
        print(df)

        results = io_utils.lookup_identities(df['identity'], db_path=self.db_path)
        print(df)
        df[['identity', 'name', 'designation']] = pd.DataFrame(
            [(result[1], f'{result[3]}_{result[4]}', result[5])
            for result in results]
        )
        return df

    def find(
        self, 
        img_path: Union[str, np.ndarray],
        db_path: str,
        model_name: str = 'Facenet512',
        distance_metric: str = 'cosine',
        detector_backend: str = 'centerface_gpu',
        align: bool = True,
        expand_percentage: int = 0,
        enhance: bool = True,
        threshold: Optional[float] = None,
        normalization: str = 'base',
        refresh_database: bool = True,
        batched: bool = False,
    ) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:

        representations = self.prepare_data(
            db_path,
            model_name,
            detector_backend,
            align,
            expand_percentage,
            normalization,
            refresh_database
        )

        if len(representations) == 0:
            return []
        
        source_objs = self.detection_pipeline(
            img_path=img_path,
            detector_backend=detector_backend,
            align=align,
            enhance=enhance,
            expand_percentage=expand_percentage
        )
        if self.save_data:
            self.source_objs.setdefault(self.i, []).extend(source_objs)

        if batched:
            press_stopwatch(self, 'face_recognition_time')

            batched_results = recognition.find_batched(
                representations,
                source_objs,
                model_name,
                distance_metric,
                align,
                threshold,
                normalization,
            )
            press_stopwatch(self, 'face_recognition_time')

            return batched_results
        
        df = pd.DataFrame(representations)

        resp_obj = []

        for source_obj in source_objs:
            face_img = source_obj['face_img']
            face_region = source_obj['facial_area']

            press_stopwatch(self, 'face_recognition_time')
            target_embedding_obj = representation.represent(
                img_path=face_img,
                model_name=model_name,
                detector_backend='skip',
                align=align,
                normalization=normalization,
            )
            press_stopwatch(self, 'face_recognition_time')

            press_stopwatch(self, 'other_processing_time')
            target_representation = target_embedding_obj[0]['embedding']

            result_df = df.copy()  # df will be filtered in each img
            result_df['source_x'] = face_region['x']
            result_df['source_y'] = face_region['y']
            result_df['source_w'] = face_region['w']
            result_df['source_h'] = face_region['h']

            distances = []
            for _, instance in df.iterrows():
                source_representation = instance['embedding']
                if source_representation is None:
                    distances.append(float('inf'))  # no representation for this image
                    continue

                target_dims = len(list(target_representation))
                source_dims = len(list(source_representation))
                if target_dims != source_dims:
                    raise ValueError(
                        'Source and target embeddings must have same dimensions but '
                        + f'{target_dims}:{source_dims}. Model structure may change'
                        + ' after pickle created. Delete the {file_name} and re-run.'
                    )

                distance = verification.find_distance(
                    source_representation, target_representation, distance_metric
                )

                distances.append(distance)

            target_threshold = threshold or verification.find_threshold(model_name, distance_metric)

            result_df['threshold'] = target_threshold
            result_df['distance'] = distances

            result_df = result_df.drop(columns=['embedding'])
            result_df = result_df[result_df['distance'] <= target_threshold]
            result_df = result_df.sort_values(by=['distance'], ascending=True).reset_index(drop=True)

            resp_obj.append(result_df)

            press_stopwatch(self, 'other_processing_time')

        if self.save_data:
            self.det_recognition_dfs.setdefault(self.i, []).extend(resp_obj)

        return resp_obj

    def detection_pipeline(
        self,
        img_path: Union[str, np.ndarray, IO[bytes]],
        detector_backend: str = 'centerface_gpu',
        align: bool = True,
        expand_percentage: int = 0,
        enhance: bool = True,
        color_face: str = 'rgb',
        normalize_face: bool = True,
        warn: bool = False
    ) -> List[Dict[str, Any]]:

        resp_objs = []
        img, img_name = image_utils.load_image(img_path)

        if img is None:
            raise ValueError(f'Exception while loading {img_name}')

        height, width, _ = img.shape

        args_ = {
            'color_face': color_face,
            'width': width,
            'height': height,
            'normalize_face': normalize_face
        }

        base_region = FacialAreaRegion(x=0, y=0, w=width, h=height, confidence=0)

        if detector_backend == 'skip':
            face_objs = [DetectedFace(img=img, facial_area=base_region, confidence=0)]
        else:
            face_objs = self.detect_faces(
                img=img,
                align=align,
                expand_percentage=expand_percentage,
                enhance=enhance,
                warn=warn
            )

        if self.save_data:
            self.face_objs.setdefault(self.i, []).extend(face_objs)

        for face_obj in face_objs:
            resp_obj = face_utils.format_response(face_obj, **args_)

            resp_objs.append(resp_obj)

        return resp_objs

    def detect_faces(
        self,
        img: np.ndarray,
        align: bool = True,
        expand_percentage: int = 0,
        enhance: bool = True,
        warn: bool = False
    ) -> List[DetectedFace]:

        height, width, _ = img.shape

        height_border, width_border = int(0.5 * height), int(0.5 * width)
        if align is True:
            img = cv2.copyMakeBorder(
                img,
                height_border,
                height_border,
                width_border,
                width_border,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],  # black border 
            )
        
        args_ = {
            'expand_percentage': expand_percentage,
            'align': align,
            'width_border': width_border,
            'height_border': height_border,
            'save_data': self.save_data
        } 
        
        press_stopwatch(self, 'face_detection_time')
        facial_areas = self.face_detector.detect_faces(img)
        if warn and (not facial_areas):
            print('No faces detected')
        press_stopwatch(self, 'face_detection_time')

        press_stopwatch(self, 'other_processing_time')
        
        if self.save_data:
            self.face_detections.setdefault(self.i, []).extend(facial_areas)
            args_['data_index'] = (self.i, self.i_f)

        results = []
        for facial_area in facial_areas:
            facial_area, face_img = face_utils.adjust_and_extract(
                facial_area, img, **args_
            )

            if enhance:
                face_img = self.enhance_face(face_img, is_rgb=True)

            face_obj = DetectedFace(
                img=face_img,
                facial_area=facial_area,
                confidence=facial_area.confidence
            )

            results.append(face_obj)

            if self.save_data:
                self.i_f += 1   # Increment secondary index
                args_['data_index'] = (self.i, self.i_f)

        press_stopwatch(self, 'other_processing_time')

        return results

    def enhance_face(
        self,
        img: np.ndarray,
        is_rgb=True,
        output_path=None
    ):
        # Start timing

        enhanced_face = self.face_enhancer.forward(img, is_rgb=is_rgb)

        # End timing
        
        if self.save_data or output_path:
            if not output_path:
                output_dir = '../files/output'
                filename = f'{self.i}_{self.i_f}_enhanced_detection.png'
                output_path = os.path.join(output_dir, filename)

            cv2.imwrite(output_path, enhanced_face)

        return enhanced_face

    def save_runtime_data(self, filename='../files/output/faceiq_data.xlsx'):
        if not self.save_data:
            return
        
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
        
        all_idxs = sorted(set([
            *self.regions.keys(),
            *self.face_detections.keys(),
            *self.face_objs.keys(),
            *self.source_objs.keys(),
            *self.det_recognition_dfs.keys()
        ]))

        artifact_data = {
            'idx': all_idxs,
            'regions': [len(self.regions.get(i, [])) for i in all_idxs],
            'face_detections': [len(self.face_detections.get(i, [])) for i in all_idxs],
            'face_objs': [len(self.face_objs.get(i, [])) for i in all_idxs],
            'source_objs': [len(self.source_objs.get(i, [])) for i in all_idxs],
            'det_recognition_dfs': [len(self.det_recognition_dfs.get(i, [])) for i in all_idxs]
        }
        
        artifact_df = pd.DataFrame(artifact_data)

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            detections_df.to_excel(writer, sheet_name='Detections', index=False)
            artifact_df.to_excel(writer, sheet_name='Pipeline Artifacts', index=False)

    def visualize_identifications(self, image, identifications, output_path: str = None):
        image = image.copy()
        color = (245, 104, 17)

        for face_df in identifications:
            if face_df.empty:
                continue
    
            best_match = face_df.loc[face_df['distance'].idxmin()]
            name = best_match['name']

            x, y, w, h = best_match[['x', 'y', 'w', 'h']]
            x1, y1, x2, y2 = utils.xywh_xyxy([x, y, w, h])

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                image, f'distance: {best_match["distance"]:.2f}', (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2
            )

            cv2.putText(
                image, f'name: {name}', (x2 - 5, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2
            )

        if output_path:
            cv2.imwrite(output_path, image)

        return image
