import torch
import io_utils
from inference import InferencePipeline
from tracking import tracking_pipeline
import cv2
import time
import utilities
import os


def delete_files(time_prefix, footage_path='../input_files/',
                 intermediate_output='../intermediate_output/'):

    for file in os.listdir(footage_path):
        if file.startswith(time_prefix):
            os.remove(footage_path + file)

    for file in os.listdir(intermediate_output):
        if file.startswith(time_prefix):
            os.remove(intermediate_output + file)


def main(stride=15):
    def _get_hdf5_path(video_file):
        return ('../intermediate_output/' + video_file.split('.')[0]
                + '_embeddings.hdf5')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    yolov4_weights = 'models/YOLOv4.pth'
    osnet_weights = 'models/OSNet.pth.tar-250'
    weights_paths = [yolov4_weights, osnet_weights]

    while True:
        primary = io_utils.get_queue_block()
    
        if len(primary) == 0:
            print('No clips in the queue')
            time.sleep(60)
            continue

        time_prefix = utilities.parse_clip_filename(primary[0][1], data='time')
        timestamp = utilities.frame_timestamp(time_prefix)
        
        for row in primary:
            video_file = row[1]
            camera = video_file.split('.')[0].split('_')[-1]

            hdf5_path = _get_hdf5_path(video_file)

            inference_pipeline = InferencePipeline(
                video_file, weights_paths, hdf5_path, device,
                track_stride=stride, id_stride=30
            )

            print('detecting and embedding...')
            inference_pipeline.run()

            print('tracking...')
            all_trks = tracking_pipeline(video_file)

            io_utils.save_track_info(time_prefix, camera, all_trks)

        io_utils.post_events_to_webapp(time_prefix)
        io_utils.update_queue(action='clear_section', datetime=timestamp)
        delete_files(time_prefix)

        #     identification_pipeline(time_prefix)

        #     io_utils.post_events_to_webapp(time_prefix)


        # else:
        #     io_utils.update_queue(action='clear_section', datetime=timestamp)
        #     delete_files(time_prefix)            
        # break

if __name__ == '__main__':
    main(stride=3)
