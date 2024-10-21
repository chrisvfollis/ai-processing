from deepface import DeepFace
import tensorflow as tf
import cv2
import input_output as io_utils
import datetime
import sys


def facial_recognition(file_path, track_data, threshold=0.8):
    detection_data = track_data['detections']
    
    start = min(detection_data.keys())
    f_num = start

    base_path = '../input_files/'
    cap = cv2.VideoCapture(base_path + file_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    face_detected = False
    while face_detected == False:

        ret, frame = cap.read()
        if not ret:
            return None, None
        if (f_num - start) % 20 == 0:
            if f_num not in detection_data:
                f_num += 1
                start = f_num
                continue
        
            box = detection_data[f_num]
            if (box[2] * box[3]) >= 30000:
                x1, y1 = box[0], box[1]
                x2, y2 = box[0] + box[2], box[1] + box[3]
                cropped = frame[y1:y2, x1:x2]

                try:
                    dfs = DeepFace.find(
                        img_path = cropped, db_path = '../input_files/faces',
                        model_name = 'Facenet', threshold = threshold,
                        enforce_detection = True, silent = True
                    )
                    
                    if len(dfs[0]['identity']) > 0:
                        face_detected = True
                except ValueError:
                    face_detected = False
        f_num += 1
    cap.release()

    dfs[0] = dfs[0].loc[(dfs[0]["source_y"] + box[1] + (dfs[0]["source_h"]/2))
                        <= (box[1] + (box[3]/3))]
    if len(dfs[0]) == 0:
        return None, None

    min_index = dfs[0]['distance'].idxmin()
    min_row = dfs[0].loc[min_index]

    identity = min_row['identity'].split('/')[-1]
    distance = min_row['distance']

    return identity, distance



if __name__ == '__main__':
    location = sys.argv[1]
    timestamp = sys.argv[2]
    trks = sys.argv[3].split(',')

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
            print("Using GPU: ", physical_devices[0], flush=True)
        except RuntimeError as e:
            print(e, flush=True)
    else:
        print("No GPU available", flush=True)

    
    cams = [trk.split('_')[0] for trk in trks]

    trk_file = f'../intermediate_output/{location}_{timestamp}_trk_data.hdf5'

    _, trk_data = io_utils.get_trk_data(trk_file, cams, min_span=240)

    trks = [trk for trk in trks if trk_data.get(trk, None)]
    print(trks, flush=True)
    
    for trk in trks:
        cam = trk.split('_')[0].strip('c')
        vid_file = f'{location}_{timestamp}_{cam}.mp4'
        data = trk_data.get(trk, None)
        if data:
            identity, distance = facial_recognition(vid_file, trk_data[trk])
            print(f'{trk}: {identity}, {distance}', flush=True)