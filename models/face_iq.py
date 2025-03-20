# standard dependencies
import os
from typing import Any, Dict, Set, List, Tuple, IO, Union, Optional, Sequence
import pickle
from heapq import nlargest
import time
import gc
import csv
import math

# 3rd-party dependencies
import numpy as np
import pandas as pd
import cv2
import torch
from deepface.commons import image_utils
from deepface.modules import modeling, representation, verification, recognition
from deepface.commons.logger import Logger
from deepface.models.Detector import Detector, DetectedFace, FacialAreaRegion
from tqdm import tqdm

# internal dependencies
from utilities import io_utils
from utilities import utilities as utils


class FaceIq:
    def __init__(self, recognition_model, detection_model, id_cutoff=0.8,
                 face_dir='../files/input/faces', db_path='../files/data.db',
                 weights_path='../models/weights/centerface.pth',
                 save_data=False):
        self.recognition_model = recognition_model
        self.detection_model = detection_model

        if detection_model == 'centerface_gpu':
            self.face_detector = CenterFace(weights_path=weights_path)
        else:
            self.face_detector: Detector = modeling.build_model(
                task="face_detector", model_name=detection_model
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
            self.regions = {}
            self.face_detections = {}       # <-- self.face_detector.detect_faces() <-- self.detect_faces()
            self.face_objs = {}             # <-- self.detect_faces() <-- self.extract_faces()
            self.source_objs = {}           # <-- self.extract_faces() <-- self.find()
            self.det_recognition_dfs = {}   # <-- self.find()

    def identify_faces(self, img, id_cutoff=None, regions=None):
        def _postprocess_output(all_face_dfs):
            start_other_processing = time.perf_counter()
    
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
                    continue
            
            end_other_processing = time.perf_counter()
            self.other_processing_time += (end_other_processing - start_other_processing)

            return filtered_face_dfs

        id_cutoff = id_cutoff or self.id_cutoff

        config = {'db_path': self.face_dir, 'model_name': self.recognition_model,
                  'detector_backend': self.detection_model, 'threshold': id_cutoff,
                  'batched': False}

        all_face_dfs = []
        
        start_id = time.perf_counter()

        if not regions:
            face_dfs = self.find(img_path=img, **config)
            for df in face_dfs:
                df = utils.reformat_face_df(df)

                all_face_dfs.append(df)

        else:
            if self.save_data:
                self.regions.setdefault(self.i, []).extend(regions)

            for region in regions:
                img_crop = utils.crop_region(img, region)
                local_face_dfs = self.find(img_path=img_crop, **config)

                del img_crop
                gc.collect()

                for df in local_face_dfs:
                    df = utils.reformat_face_df(df)
                    if df.empty:
                        continue
    
                    df[['x', 'y']] = df.apply(
                        lambda row:

                        utils.apply_offset(
                            (row['x'], row['y']), region
                        ),

                        axis=1
                    ).apply(pd.Series)

                    all_face_dfs.append(df)

        end_id = time.perf_counter()
        self.identification_pipeline_time += (end_id - start_id)
        
        results = _postprocess_output(all_face_dfs)

        return results

    def visualize_identifications(self, image, identifications, output_path: str = None):
        image = image.copy()
        color = (245, 104, 17)

        for face_df in identifications:
            if face_df.empty:
                continue
    
            best_match = face_df.loc[face_df['distance'].idxmin()]

            x, y, w, h = best_match[['x', 'y', 'w', 'h']]
            x1, y1, x2, y2 = utils.xywh_to_xyxy([x, y, w, h])

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                image, f"distance: {best_match['distance']:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            first_name, _ = io_utils.lookup_name(best_match['identity'])
            cv2.putText(
                image, f"name: {first_name}", (x2 - 5, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

        if output_path:
            cv2.imwrite(output_path, image)

        return image

    def find(
        self, 
        img_path: Union[str, np.ndarray],
        db_path: str,
        model_name: str = "Facenet512",
        distance_metric: str = "cosine",
        detector_backend: str = "centerface_gpu",
        align: bool = True,
        expand_percentage: int = 0,
        threshold: Optional[float] = None,
        normalization: str = "base",
        refresh_database: bool = True,
        batched: bool = False,
    ) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:
        
        def __find_bulk_embeddings(
            employees: Set[str],
            model_name: str = "VGG-Face",
            detector_backend: str = "opencv",
            align: bool = True,
            expand_percentage: int = 0,
            normalization: str = "base",
        ) -> List[Dict["str", Any]]:
            
            representations = []
            for employee in employees:
                file_hash = image_utils.find_image_hash(employee)

                start_detection = time.perf_counter()
                try:
                    img_objs = self.extract_faces(
                        img_path=employee,
                        detector_backend=detector_backend,
                        align=align,
                        expand_percentage=expand_percentage,
                        color_face='bgr'  # `represent` expects images in bgr format.
                    )

                except ValueError as err:
                    self.logger.error(f"Exception while extracting faces from {employee}: {str(err)}")
                    img_objs = []

                end_detection = time.perf_counter()
                self.face_detection_time += (end_detection - start_detection)

                if len(img_objs) == 0:
                    representations.append(
                        {
                            "identity": employee,
                            "hash": file_hash,
                            "embedding": None,
                            "target_x": 0,
                            "target_y": 0,
                            "target_w": 0,
                            "target_h": 0,
                        }
                    )
                else:
                    for img_obj in img_objs:
                        img_content = img_obj["face"]
                        img_region = img_obj["facial_area"]

                        start_recognition = time.perf_counter()

                        embedding_obj = representation.represent(
                            img_path=img_content,
                            model_name=model_name,
                            detector_backend="skip",
                            align=align,
                            normalization=normalization,
                        )

                        end_recognition = time.perf_counter()
                        self.face_recognition_time += (end_recognition - start_recognition)

                        img_representation = embedding_obj[0]["embedding"]
                        representations.append(
                            {
                                "identity": employee,
                                "hash": file_hash,
                                "embedding": img_representation,
                                "target_x": img_region["x"],
                                "target_y": img_region["y"],
                                "target_w": img_region["w"],
                                "target_h": img_region["h"],
                            }
                        )

            return representations

        start_other_processing = time.perf_counter()
        self.logger = Logger()

        if not os.path.isdir(db_path):
            raise ValueError(f"Passed path {db_path} does not exist!")

        img, _ = image_utils.load_image(img_path)
        if img is None:
            raise ValueError(f"Passed image path {img_path} does not exist!")

        file_parts = [
            "ds", "model", model_name,
            "detector", detector_backend,
            "aligned" if align else "unaligned",
            "normalization", normalization,
            "expand", str(expand_percentage),
        ]

        file_name = "_".join(file_parts) + ".pkl"
        file_name = file_name.replace("-", "").lower()

        datastore_path = os.path.join(db_path, file_name)
        representations = []

        # required columns for representations
        df_cols = {
            "identity",
            "hash",
            "embedding",
            "target_x",
            "target_y",
            "target_w",
            "target_h",
        }

        if not os.path.exists(datastore_path):
            with open(datastore_path, "wb") as f:
                pickle.dump([], f, pickle.HIGHEST_PROTOCOL)

        with open(datastore_path, "rb") as f:
            representations = pickle.load(f)

        # check each item of representations list has required keys
        for i, current_representation in enumerate(representations):
            missing_keys = df_cols - set(current_representation.keys())
            if len(missing_keys) > 0:
                raise ValueError(
                    f"{i}-th item does not have some required keys - {missing_keys}."
                    f"Consider to delete {datastore_path}"
                )

        # Get the list of images on storage
        storage_images = set(image_utils.yield_images(path=db_path))

        if len(storage_images) == 0 and refresh_database is True:
            raise ValueError(f"No item found in {db_path}")
        if len(representations) == 0 and refresh_database is False:
            raise ValueError(f"Nothing is found in {datastore_path}")

        must_save_pickle = False
        new_images, old_images, replaced_images = set(), set(), set()

        # Enforce data consistency amongst on disk images and pickle file
        if refresh_database:
            # embedded images
            pickled_images = {
                representation["identity"] for representation in representations
            }

            new_images = storage_images - pickled_images  # images added to storage
            old_images = pickled_images - storage_images  # images removed from storage

            # detect replaced images
            for current_representation in representations:
                identity = current_representation["identity"]
                if identity in old_images:
                    continue
                alpha_hash = current_representation["hash"]
                beta_hash = image_utils.find_image_hash(identity)
                if alpha_hash != beta_hash:
                    replaced_images.add(identity)

        # append replaced images into both old and new images. these will be dropped and re-added.
        new_images.update(replaced_images)
        old_images.update(replaced_images)

        # remove old images first
        if len(old_images) > 0:
            representations = [rep for rep in representations if rep["identity"] not in old_images]
            must_save_pickle = True

        end_other_processing = time.perf_counter()
        self.other_processing_time += (start_other_processing - end_other_processing)

        # find representations for new images
        if len(new_images) > 0:
            representations += __find_bulk_embeddings(
                employees=new_images,
                model_name=model_name,
                detector_backend=detector_backend,
                align=align,
                expand_percentage=expand_percentage,
                normalization=normalization,
            )  # add new images
            must_save_pickle = True

        start_other_processing = time.perf_counter()
        if must_save_pickle:
            with open(datastore_path, "wb") as f:
                pickle.dump(representations, f, pickle.HIGHEST_PROTOCOL)

        end_other_processing = time.perf_counter()
        self.other_processing_time += (start_other_processing - end_other_processing)

        # Should we have no representations bailout
        if len(representations) == 0:
            return []
        
        # ----------------------------
        # now, we got representations for facial database

        # img path might have more than once face
        source_objs = self.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            align=align,
            expand_percentage=expand_percentage,
        )
        if self.save_data:
            self.source_objs.setdefault(self.i, []).extend(source_objs)

        if batched:
            start_recognition = time.perf_counter()

            batched_results = recognition.find_batched(
                representations,
                source_objs,
                model_name,
                distance_metric,
                align,
                threshold,
                normalization,
            )

            end_recognition = time.perf_counter()
            self.face_recognition_time += (end_recognition - start_recognition)

            return batched_results
        
        df = pd.DataFrame(representations)

        resp_obj = []

        for source_obj in source_objs:
            source_img = source_obj["face"]
            source_region = source_obj["facial_area"]

            start_recognition = time.perf_counter()

            target_embedding_obj = representation.represent(
                img_path=source_img,
                model_name=model_name,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            end_recognition = time.perf_counter()
            self.face_recognition_time += (end_recognition - start_recognition)

            start_other_processing = time.perf_counter()

            target_representation = target_embedding_obj[0]["embedding"]

            result_df = df.copy()  # df will be filtered in each img
            result_df["source_x"] = source_region["x"]
            result_df["source_y"] = source_region["y"]
            result_df["source_w"] = source_region["w"]
            result_df["source_h"] = source_region["h"]

            distances = []
            for _, instance in df.iterrows():
                source_representation = instance["embedding"]
                if source_representation is None:
                    distances.append(float("inf"))  # no representation for this image
                    continue

                target_dims = len(list(target_representation))
                source_dims = len(list(source_representation))
                if target_dims != source_dims:
                    raise ValueError(
                        "Source and target embeddings must have same dimensions but "
                        + f"{target_dims}:{source_dims}. Model structure may change"
                        + " after pickle created. Delete the {file_name} and re-run."
                    )

                distance = verification.find_distance(
                    source_representation, target_representation, distance_metric
                )

                distances.append(distance)

                # ---------------------------
            target_threshold = threshold or verification.find_threshold(model_name, distance_metric)

            result_df["threshold"] = target_threshold
            result_df["distance"] = distances

            result_df = result_df.drop(columns=["embedding"])
            result_df = result_df[result_df["distance"] <= target_threshold]
            result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

            resp_obj.append(result_df)

            end_other_processing = time.perf_counter()
            self.other_processing_time += (start_other_processing - end_other_processing)

        if self.save_data:
            self.det_recognition_dfs.setdefault(self.i, []).extend(resp_obj)

        return resp_obj

    def extract_faces(
        self,
        img_path: Union[str, np.ndarray, IO[bytes]],
        detector_backend: str = "centerface_gpu",
        align: bool = True,
        expand_percentage: int = 0,
        color_face: str = "rgb",
        normalize_face: bool = True,
        max_faces: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        resp_objs = []

        img, img_name = image_utils.load_image(img_path)

        if img is None:
            raise ValueError(f"Exception while loading {img_name}")

        height, width, _ = img.shape

        base_region = FacialAreaRegion(x=0, y=0, w=width, h=height, confidence=0)

        if detector_backend == "skip":
            face_objs = [DetectedFace(img=img, facial_area=base_region, confidence=0)]
        else:
            face_objs = self.detect_faces(
                img=img,
                align=align,
                expand_percentage=expand_percentage,
                max_faces=max_faces,
            )

        if self.save_data:
            self.face_objs.setdefault(self.i, []).extend(face_objs)

        for face_obj in face_objs:
            current_img = face_obj.img
            current_region = face_obj.facial_area

            if color_face == "rgb":
                cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
            elif color_face == "bgr":
                pass  # image is in BGR
            elif color_face == "gray":
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"The color_face can be rgb, bgr or gray, but it is {color_face}.")

            if normalize_face:
                current_img = current_img / 255  # normalize input in [0, 1]

            # cast to int for flask, and do final checks for borders
            x = max(0, int(current_region.x))
            y = max(0, int(current_region.y))
            w = min(width - x - 1, int(current_region.w))
            h = min(height - y - 1, int(current_region.h))

            facial_area = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "left_eye": current_region.left_eye,
                "right_eye": current_region.right_eye,
            }

            # optional nose, mouth_left and mouth_right fields are coming just for retinaface
            if current_region.nose is not None:
                facial_area["nose"] = current_region.nose
            if current_region.mouth_left is not None:
                facial_area["mouth_left"] = current_region.mouth_left
            if current_region.mouth_right is not None:
                facial_area["mouth_right"] = current_region.mouth_right

            resp_obj = {
                "face": current_img,
                "facial_area": facial_area,
                "confidence": round(float(current_region.confidence or 0), 2),
            }

            resp_objs.append(resp_obj)

        return resp_objs

    def detect_faces(
        self,
        img: np.ndarray,
        align: bool = True,
        expand_percentage: int = 0,
        max_faces: Optional[int] = None,
    ) -> List[DetectedFace]:

        height, width, _ = img.shape

        # validate expand percentage score
        if expand_percentage < 0:
            self.logger.warn(
                f"Expand percentage cannot be negative but you set it to {expand_percentage}."
                "Overwritten it to 0."
            )
            expand_percentage = 0

        # If faces are close to the upper boundary, alignment move them outside
        # Add a black border around an image to avoid this.
        height_border = int(0.5 * height)
        width_border = int(0.5 * width)
        if align is True:
            img = cv2.copyMakeBorder(
                img,
                height_border,
                height_border,
                width_border,
                width_border,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],  # Color of the border (black)
            )

        # find facial areas of given image
        start_detection = time.perf_counter()

        facial_areas = self.face_detector.detect_faces(img)

        end_detection = time.perf_counter()
        self.face_detection_time += (end_detection - start_detection)

        if self.save_data:
            self.face_detections.setdefault(self.i, []).extend(facial_areas)

        start_other_processing = time.perf_counter()
        if max_faces is not None and max_faces < len(facial_areas):
            facial_areas = nlargest(
                max_faces, facial_areas, key=lambda facial_area: facial_area.w * facial_area.h
            )

        results = [
            self.extract_face(
                facial_area=facial_area,
                img=img,
                align=align,
                expand_percentage=expand_percentage,
                width_border=width_border,
                height_border=height_border,
            )
            for facial_area in facial_areas
        ]

        end_other_processing = time.perf_counter()
        self.other_processing_time += (start_other_processing - end_other_processing)

        return results

    def extract_face(
        self, 
        facial_area: FacialAreaRegion,
        img: np.ndarray,
        align: bool,
        expand_percentage: int,
        width_border: int,
        height_border: int,
    ) -> DetectedFace:
        x = facial_area.x
        y = facial_area.y
        w = facial_area.w
        h = facial_area.h
        left_eye = facial_area.left_eye
        right_eye = facial_area.right_eye
        confidence = facial_area.confidence
        nose = facial_area.nose
        mouth_left = facial_area.mouth_left
        mouth_right = facial_area.mouth_right

        if expand_percentage > 0:
            # Expand the facial region height and width by the provided percentage
            # ensuring that the expanded region stays within img.shape limits
            expanded_w = w + int(w * expand_percentage / 100)
            expanded_h = h + int(h * expand_percentage / 100)

            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(img.shape[1] - x, expanded_w)
            h = min(img.shape[0] - y, expanded_h)

        # extract detected face unaligned
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
        # align original image, then find projection of detected face area after alignment
        if align is True:  # and left_eye is not None and right_eye is not None:
            # we were aligning the original image before, but this comes with an extra cost
            # instead we now focus on the facial area with a margin
            # and align it instead of original image to decrese the cost
            sub_img, relative_x, relative_y = self.extract_sub_image(img=img, facial_area=(x, y, w, h))

            aligned_sub_img, angle = self.align_img_wrt_eyes(
                img=sub_img, left_eye=left_eye, right_eye=right_eye
            )

            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = self.project_facial_area(
                facial_area=(
                    relative_x,
                    relative_y,
                    relative_x + w,
                    relative_y + h,
                ),
                angle=angle,
                size=(sub_img.shape[0], sub_img.shape[1]),
            )
            detected_face = aligned_sub_img[
                int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
            ]

            # do not spend memory for these temporary variables anymore
            del aligned_sub_img, sub_img

            # restore x, y, le and re before border added
            x = x - width_border
            y = y - height_border
            # w and h will not change
            if left_eye is not None:
                left_eye = (left_eye[0] - width_border, left_eye[1] - height_border)
            if right_eye is not None:
                right_eye = (right_eye[0] - width_border, right_eye[1] - height_border)
            if nose is not None:
                nose = (nose[0] - width_border, nose[1] - height_border)
            if mouth_left is not None:
                mouth_left = (mouth_left[0] - width_border, mouth_left[1] - height_border)
            if mouth_right is not None:
                mouth_right = (mouth_right[0] - width_border, mouth_right[1] - height_border)

        return DetectedFace(
            img=detected_face,
            facial_area=FacialAreaRegion(
                x=x,
                y=y,
                h=h,
                w=w,
                confidence=confidence,
                left_eye=left_eye,
                right_eye=right_eye,
                nose=nose,
                mouth_left=mouth_left,
                mouth_right=mouth_right,
            ),
            confidence=confidence or 0,
        )

    def extract_sub_image(
        self, img: np.ndarray, facial_area: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, int, int]:
        """
        Get the sub image with given facial area while expanding the facial region
            to ensure alignment does not shift the face outside the image.

        This function doubles the height and width of the face region,
        and adds black pixels if necessary.

        Args:
            - img (np.ndarray): pre-loaded image with detected face
            - facial_area (tuple of int): Representing the (x, y, w, h) of the facial area.

        Returns:
            - extracted_face (np.ndarray): expanded facial image
            - relative_x (int): adjusted x-coordinates relative to the expanded region
            - relative_y (int): adjusted y-coordinates relative to the expanded region
        """
        x, y, w, h = facial_area
        relative_x = int(0.5 * w)
        relative_y = int(0.5 * h)

        # calculate expanded coordinates
        x1, y1 = x - relative_x, y - relative_y
        x2, y2 = x + w + relative_x, y + h + relative_y

        # most of the time, the expanded region fits inside the image
        if x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
            return img[y1:y2, x1:x2], relative_x, relative_y

        # but sometimes, we need to add black pixels
        # ensure the coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        cropped_region = img[y1:y2, x1:x2]

        # create a black image
        extracted_face = np.zeros(
            (h + 2 * relative_y, w + 2 * relative_x, img.shape[2]), dtype=img.dtype
        )

        # map the cropped region
        start_x = max(0, relative_x - x)
        start_y = max(0, relative_y - y)
        extracted_face[
            start_y : start_y + cropped_region.shape[0], start_x : start_x + cropped_region.shape[1]
        ] = cropped_region

        return extracted_face, relative_x, relative_y

    def align_img_wrt_eyes(
        self,
        img: np.ndarray,
        left_eye: Optional[Union[list, tuple]],
        right_eye: Optional[Union[list, tuple]],
    ) -> Tuple[np.ndarray, float]:
        """
        Align a given image horizontally with respect to their left and right eye locations
        Args:
            img (np.ndarray): pre-loaded image with detected face
            left_eye (list or tuple): coordinates of left eye with respect to the person itself
            right_eye(list or tuple): coordinates of right eye with respect to the person itself
        Returns:
            img (np.ndarray): aligned facial image
        """
        # if eye could not be detected for the given image, return image itself
        if left_eye is None or right_eye is None:
            return img, 0

        # sometimes unexpectedly detected images come with nil dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            return img, 0

        angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        return img, angle

    def project_facial_area(
        self, facial_area: Tuple[int, int, int, int], angle: float, size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Update pre-calculated facial area coordinates after image itself
            rotated with respect to the eyes.
        Inspried from the work of @UmutDeniz26 - github.com/serengil/retinaface/pull/80

        Args:
            facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
                x2 is equal to x1 + w1, and y2 is equal to y1 + h1
            angle (float): Angle of rotation in degrees. Its sign determines the direction of rotation.
                        Note that angles > 360 degrees are normalized to the range [0, 360).
            size (tuple of int): Tuple representing the size of the image (width, height).

        Returns:
            rotated_coordinates (tuple of int): Representing the new coordinates
                (x1, y1, x2, y2) or (x1, y1, x1+w1, y1+h1) of the rotated facial area.
        """

        # Normalize the witdh of the angle so we don't have to
        # worry about rotations greater than 360 degrees.
        # We workaround the quirky behavior of the modulo operator
        # for negative angle values.
        direction = 1 if angle >= 0 else -1
        angle = abs(angle) % 360
        if angle == 0:
            return facial_area

        # Angle in radians
        angle = angle * np.pi / 180

        height, weight = size

        # Translate the facial area to the center of the image
        x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
        y = (facial_area[1] + facial_area[3]) / 2 - height / 2

        # Rotate the facial area
        x_new = x * np.cos(angle) + y * direction * np.sin(angle)
        y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

        # Translate the facial area back to the original position
        x_new = x_new + weight / 2
        y_new = y_new + height / 2

        # Calculate projected coordinates after alignment
        x1 = x_new - (facial_area[2] - facial_area[0]) / 2
        y1 = y_new - (facial_area[3] - facial_area[1]) / 2
        x2 = x_new + (facial_area[2] - facial_area[0]) / 2
        y2 = y_new + (facial_area[3] - facial_area[1]) / 2

        # validate projected coordinates are in image's boundaries
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), weight)
        y2 = min(int(y2), height)

        return (x1, y1, x2, y2)

    def save_runtime_data(self, filename='../files/output/faceiq_data.xlsx'):
        if not self.save_data:
            return
        
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

        detection_data = []
        for frame_idx, detections in self.face_detections.items():
            for det in detections:  # Assuming `det` has `.w` and `.h` attributes
                area = math.prod((det.w, det.h))
                detection_data.append({'idx': frame_idx, 'w': det.w, 'h': det.h, 'a': area})

        size_df = pd.DataFrame(detection_data)

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            artifact_df.to_excel(writer, sheet_name='Pipeline Artifacts', index=False)
            size_df.to_excel(writer, sheet_name='Detection Sizes', index=False)


class CenterFace:
    def __init__(self, weights_path: str ='../models/weights/centerface.pth',
                 landmarks: bool = True, save_data: bool = False,
                 min_dims: Sequence = None):
        '''
        Adapted from https://github.com/Star-Clouds/CenterFace/ and modified
        for compatibility with DeepFace
        '''
    
        self.landmarks = landmarks
        
        self.model = torch.load(weights_path, map_location='cuda' if
                                torch.cuda.is_available() else 'cpu')
        self.model.eval()

        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

        self.save_data = save_data
        if self.save_data:
            self.i = 0
            self.face_detections = {}
        
        self.min_dims = min_dims

    def detect_faces(
            self, img: np.ndarray, threshold=0.5,
            offset: Sequence = None, min_dims: Sequence = None
        ) -> List[FacialAreaRegion]:

        min_dims = min_dims or self.min_dims

        h, w = img.shape[:2]
        if (h == 0) or (w == 0):
            return []

        if (h >= 32) and (w >= 32):
            self.img_h_new = (h // 32) * 32
            self.img_w_new = (w // 32) * 32
        else:
            self.img_h_new = h + 1 if ((h % 2) != 0) else h
            self.img_w_new = w + 1 if ((w % 2) != 0) else w

        self.scale_h = h / self.img_h_new
        self.scale_w = w / self.img_w_new

        detections = self.inference_pytorch(img, threshold)

        if self.landmarks:
            all_dets, all_lms = detections
        else:
            all_dets = detections
            lms = None

        detected_faces = []
        for i, box in enumerate(all_dets):
            x1, y1, x2, y2 = map(int, box[:4])
            if offset:
                x1, y1 = utils.apply_offset((x1, y1), offset)
                x2, y2 = utils.apply_offset((x2, y2), offset)
            
            score = float(box[4])
            w, h = (x2 - x1), (y2 - y1)
            
            if all_lms is not None:
                lms = [
                    tuple(map(int, all_lms[i][j:j+2])) for j in range(0, 9, 2)
                ]
                if offset:
                    lms = utils.apply_offset(lms, offset)

                left_eye, right_eye, nose, mouth_right, mouth_left = lms
            else:
                left_eye = right_eye = nose = mouth_right = mouth_left = None

            face_region = FacialAreaRegion(
                x=x1,
                y=y1,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                nose=nose,
                mouth_right=mouth_right,
                mouth_left=mouth_left,
                confidence=score
            )
            detected_faces.append(face_region)

        if self.save_data:
            self.face_detections.setdefault(self.i, []).extend(detected_faces)

        return detected_faces

    def inference_pytorch(self, img, threshold):
        image_cv = cv2.resize(img, dsize=(self.img_w_new, self.img_h_new))
        blob = (
            cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            .transpose(2, 0, 1)
            .astype('float32')
        )
        tensor = torch.from_numpy(blob).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        heatmap, scale, offset, lms = outputs

        return self.postprocess(
            heatmap.cpu().numpy(), lms.cpu().numpy(), offset.cpu().numpy(),
            scale.cpu().numpy(), threshold
        )

    def postprocess(self, heatmap, lms, offset, scale, threshold):
        if self.landmarks:
            dets, lms = self.decode(heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold=threshold)
        else:
            dets = self.decode(heatmap, scale, offset, None, (self.img_h_new, self.img_w_new), threshold=threshold)

        if len(dets) > 0:
            dets[:, 0:4:2] /= self.scale_w
            dets[:, 1:4:2] /= self.scale_h
            if self.landmarks:
                lms[:, 0:10:2] /= self.scale_w
                lms[:, 1:10:2] /= self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            if self.landmarks:
                lms = np.empty(shape=[0, 10], dtype=np.float32)

        return (dets, lms) if self.landmarks else dets

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        if self.landmarks:
            boxes, lms = [], []
        else:
            boxes = []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)
                lms = lms[keep, :]
        if self.landmarks:
            return boxes, lms
        else:
            return boxes

    def nms(self, boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep

    def visualize_detections(
            self, image: np.ndarray, face_detections: List[FacialAreaRegion],
            output_path: str = None
        ):
        '''
        Visualizes detected faces and their landmarks on the input image.

        Args:
            img (np.ndarray): The original input image.
            detected_faces (List[FacialAreaRegion]): The detected face regions with landmarks.
        '''
        def _bgr_color_tuples():
            blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)
            yellow, white = (0, 255, 255), (255, 255, 255)
            return blue, green, red, yellow, white

        image = image.copy()
        blue, green, red, yellow, white = _bgr_color_tuples()

        for face in face_detections:
            x1, y1, x2, y2 = utils.xywh_to_xyxy([face.x, face.y, face.w, face.h])
            cv2.rectangle(image, (x1, y1), (x2, y2), green, 2)

            cv2.putText(
                image, f'confidence: {face.confidence:.2f}', (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2
            )

            eyes = [face.left_eye, face.right_eye]
            mouth = [face.mouth_left, face.mouth_right]
            nose = face.nose
            
            for point in eyes:
                if point:
                    cv2.circle(image, point, 3, blue, -1)
            for point in mouth:
                if point:
                    cv2.circle(image, point, 3, yellow, -1)
            if nose:
                cv2.circle(image, nose, 3, red, -1)
    
        if output_path:
            cv2.imwrite(output_path, image)

        return image

    def save_runtime_data(self, filename='../files/output/centerface_data.xlsx'):
        if not self.save_data:
            return
        
        data = {'idx': [], 'face_detections': []}
        if hasattr(self, 'regions'):
            data['regions'] = []

        for i, detections in self.face_detections.items():
            data['idx'].append(i)
            data['face_detections'].append(len(detections))

            if hasattr(self, 'regions'):
                data['regions'].append(len(self.regions.get(i, [])))
        
        artifact_df = pd.DataFrame(data)

        detection_data = []
        for frame_idx, detections in self.face_detections.items():
            for det in detections:
                area = math.prod((det.w, det.h))
                detection_data.append(
                    {'idx': frame_idx, 'w': det.w, 'h': det.h, 'a': area}
                )

        size_df = pd.DataFrame(detection_data)

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            artifact_df.to_excel(writer, sheet_name='Pipeline Artifacts', index=False)
            size_df.to_excel(writer, sheet_name='Detection Sizes', index=False)
