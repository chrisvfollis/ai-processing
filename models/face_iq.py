from deepface import DeepFace
import time
from utilities import io_utils
import pandas as pd


class FaceIq:
    def __init__(self, id_model, detect_model, face_dir='../files/input/faces',
                 db_path='../files/data.db'):
        self.id_model = id_model
        self.detect_model = detect_model
        self.face_dir = face_dir
        self.db_path = db_path

        self.identification_time = 0

    def identify_faces(self, img, cutoff=0.8, regions=None):
        def _package_args(cutoff):
            config = {
                'db_path': self.face_dir, 'model_name': self.id_model,
                'detector_backend': self.detect_model, 'threshold': cutoff,
                'enforce_detection': True, 'silent': True
            }
            return config

        start_identification = time.perf_counter()
        config = _package_args(cutoff)
        all_face_dfs = []

        if not regions:
            try:
                all_face_dfs = DeepFace.find(img_path=img, **config)
            except ValueError:
                return all_face_dfs
        else:
            for region in regions:
                x1, y1 = region[0], region[1]
                x2, y2 = region[0] + region[2], region[1] + region[3]
                crop = img[y1:y2, x1:x2]

                try:
                    local_face_dfs = DeepFace.find(img_path=crop, **config)
                    if local_face_dfs:
                        for df in local_face_dfs:
                            if not df.empty:
                                df['source_x'] += x1
                                df['source_y'] += y1

                                all_face_dfs.append(df)
                except ValueError as e:
                    print(f"DeepFace error: {e}")
        
        filtered_face_dfs = []
        for df in all_face_dfs:
            df['identity'] = (
                df['identity'].map(lambda x: io_utils.get_employee(
                    x, db_path=self.db_path
                ))
            )
            df = df.loc[df.groupby('identity')['distance'].idxmin()]
            filtered_face_dfs.append(df)
        
        end_identification = time.perf_counter()
        self.identification_time += (end_identification - start_identification)
        
        return filtered_face_dfs
