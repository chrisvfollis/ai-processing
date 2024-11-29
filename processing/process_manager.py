import torch
import io_utils
from detect_embed import (load_extractor, load_yolov4, detect_yolov4,
                          detect_embed_pipeline)
from object_tracking import track
from identify import identification_pipeline
import cv2
import time
import utilities
import os


def detection_skim(video_file, model, device, stride=120):
    f_num = 0
    footage_path = '../input_files/'
    cap = cv2.VideoCapture(footage_path + video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False

        if f_num % stride == 0:
            det_xywhc = detect_yolov4(frame, 0, model, device,
                                      conf_thresh=.78)
            if len(det_xywhc) > 0:
                return True

        f_num += stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)


def delete_files(time_prefix, footage_path='../input_files/',
                 intermediate_output='../intermediate_output/'):

    for file in os.listdir(footage_path):
        if file.startswith(time_prefix):
            os.remove(footage_path + file)

    for file in os.listdir(intermediate_output):
        if file.startswith(time_prefix):
            os.remove(intermediate_output + file)


def main(stride=15):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = load_extractor('model.pth.tar-250', device)
    yolov4 = load_yolov4('YOLOv4.pth', device)

    while True:
        primary = io_utils.get_queue_block()
        if len(primary) == 0:
            print('No clips in the queue')
            time.sleep(60)
            continue
        time_prefix = utilities.parse_clip_filename(primary[0][1], data='time')
        timestamp = utilities.frame_timestamp(time_prefix)

        detections = False
        for row in primary:
            video_file = row[1]
            print(f'skimming {video_file}...')
            if detection_skim(video_file, yolov4, device):
                detections = True
                print('detected')
                break
        
        if detections == True:
            for row in primary:
                video_file = row[1]
                print('detecting and embedding...')
                frame_data = detect_embed_pipeline(video_file, yolov4,
                                                stride=stride)
                io_utils.write_detections(frame_data, video_file)

                print('tracking...')
                trk_data, span = track(video_file, stride=stride)
                io_utils.write_trk_data(video_file, trk_data, span)
            
            identification_pipeline(time_prefix)

            io_utils.post_events_to_webapp(time_prefix)
            io_utils.update_queue(action='clear_section', datetime=timestamp)
            delete_files(time_prefix)

        else:
            io_utils.update_queue(action='clear_section', datetime=timestamp)
            delete_files(time_prefix)            
        break

if __name__ == '__main__':
    main(stride=3)
