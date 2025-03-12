from deepface import DeepFace
import time
from utilities import io_utils
import time
import pandas as pd
import gc
import cv2


class FaceIq:
    def __init__(self, recognition_model, detection_model, id_cutoff=0.8,
                 face_dir='../files/input/faces', db_path='../files/data.db'):
        self.recognition_model = recognition_model
        self.detection_model = detection_model

        self.id_cutoff = id_cutoff

        self.face_dir = face_dir
        self.db_path = db_path

        self.identification_time = 0
        self.postprocess_time = 0

    def identify_faces(self, img, id_cutoff=None, regions=None):
        def _postprocess_output(all_face_dfs):
            start_postprocess = time.perf_counter()
    
            filtered_face_dfs = []
    
            for df in all_face_dfs:
                results = io_utils.lookup_identities(df['identity'])

                df[['identity', 'name', 'designation']] = pd.DataFrame(
                    [(result[1], f'{result[3]}_{result[4]}', result[5])
                     for result in results]
                )

                df = df.loc[df.groupby('identity')['distance'].idxmin()]
                filtered_face_dfs.append(df)
            
            end_postprocess = time.perf_counter()
            self.postprocess_time += (end_postprocess - start_postprocess)

            return filtered_face_dfs

        id_cutoff = id_cutoff if id_cutoff else self.id_cutoff

        config = {'db_path': self.face_dir, 'model_name': self.recognition_model,
                  'detector_backend': self.detection_model, 'threshold': id_cutoff,
                  'enforce_detection': False, 'silent': True}
        
        all_face_dfs = []
        
        start_id = time.perf_counter()

        if not regions:
            try:
                img_resized = cv2.resize(img, (1080, 720))
                all_face_dfs = DeepFace.find(img_path=img, **config)
                del img_resized
                gc.collect()
            except Exception as e:
                print(f"DeepFace error: {e}")
                
                end_id = time.perf_counter()
                self.identification_time += (end_id - start_id)
                
                return all_face_dfs
        else:
            for region in regions:
                x1, y1 = region[0], region[1]
                x2, y2 = region[0] + region[2], region[1] + region[3]
                crop = img[y1:y2, x1:x2].copy()

                try:
                    local_face_dfs = DeepFace.find(img_path=crop, **config)
                    del crop
                    gc.collect()
                    if local_face_dfs:
                        for df in local_face_dfs:
                            if not df.empty:
                                df['source_x'] += x1
                                df['source_y'] += y1

                                all_face_dfs.append(df)
                except Exception as e:
                    print(f"DeepFace error: {e}")

        end_id = time.perf_counter()
        self.identification_time += (end_id - start_id)
        
        results = _postprocess_output(all_face_dfs)

        return results
