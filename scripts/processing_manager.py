import torch
import io_utils
from object_detection import (load_extractor, load_yolov4, detect_yolov4,
                              process_clip)
from object_tracking import track
from track_identification import identify_all
import cv2
import time
import utilities


def detection_skim(file, model, stride=60):
    f_num = 0
    footage_path = '../input_files/'
    cap = cv2.VideoCapture(footage_path + file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False

        if f_num % stride == 0:
            det_xywhc = detect_yolov4(frame, 0, model, device,
                                      conf_thresh=.75)
            if len(det_xywhc) > 0:
                cap.release()
                return True

        f_num += stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = load_extractor('model.pth.tar-250', device)
    yolov4 = load_yolov4('YOLOv4.pth', device)

    while True:
        primary = io_utils.get_queue_block('../appdata/data.db',
                                        designation='primary')
        
        if len(primary) > 0:
            detections = False
            for row in primary:
                file = row[1]
                print(f'skimming {file}...')
                if detection_skim(file, yolov4):
                    detections = True
                    print('detected')
                    break
            
            if detections == True:
                for row in primary:
                    file = row[1]
                    print('detecting and embedding...')
                    frame_data, embeddings = process_clip(file, yolov4,
                                                          extractor,
                                                          stride=3)
                    io_utils.write_detection_csv(frame_data, file)
                    print('tracking...')
                    trk_data, span = track(file, 3, previous_data=None)
                    
                    io_utils.write_trk_data(file, trk_data, span)

                trk_file = file.rsplit('_', 1)[0]
                identify_all(trk_file)



                def process_events(trk_file):
                    base_path = f'../intermediate_output/{trk_file}'
                    trk_path = base_path + '_trk_data.hdf5'

                    config = io_utils.get_config()
                    primary_cams = config['primary_cameras']
                    _, all_trks = io_utils.get_trk_data(trk_path, primary_cams, min_span=60)

                    shop_id = io_utils.get_shop('../appdata/data.db')[0]

                    event_data = {'shop_id': [], 'employee_id': [],
                                  'event': [], 'start_time': [],
                                  'duration': [], 'image': []}

                    entry_subset = [trk for trk in all_trks.keys() if all_trks[trk]['entry']]
                    for trk in entry_subset:
                        event_data['shop_id'].append(shop_id)
                        event_data['employee_id'].append()
                        event_data['event'].append('entry')
                        start_time = str(utilities.frame_timestamp(trk_file, all_trks[trk]['trk_span'][0]))
                        event_data['start_time'].append(start_time)
                        event_data['duration'].append(0)

                        event_data['image'].append()
                        


                    exit_subset = [trk for trk in all_trks.keys() if all_trks[trk]['exit']
                                and not all_trks[trk].get('identity', None)]

        else:
            time.sleep(300)
