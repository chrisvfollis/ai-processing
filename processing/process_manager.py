import torch
import io_utils
from inference import InferencePipeline
import multiprocessing
from tracking import tracking_pipeline
import cv2
import time
import utilities
import os
import requests
from dotenv import load_dotenv
import boto3


def delete_files(time_prefix, footage_path='../input_files/',
                 intermediate_output='../intermediate_output/'):

    for file in os.listdir(footage_path):
        if file.startswith(time_prefix):
            os.remove(footage_path + file)

    for file in os.listdir(intermediate_output):
        if file.startswith(time_prefix):
            os.remove(intermediate_output + file)


def process_row(row, credentials, weights_paths, device, stride, time_prefix):
    def _get_hdf5_path(video_file):
        return ('../intermediate_output/' + video_file.split('.')[0]
                + '_embeddings.hdf5')

    def _download_from_s3(s3_key, credentials, bucket_name='ivakt-footage'):
        s3 = boto3.client(
            's3',
            aws_access_key_id=credentials[0],
            aws_secret_access_key=credentials[1],
            region_name='us-west-1'
        )

        try:
            local_path = '../input_files/' + s3_key 
            s3.download_file(bucket_name, s3_key, local_path)
            print(f'Downloaded {s3_key}')
            return True
        except Exception as e:
            print(f"Failed to download {s3_key}: {e}")
            return False

    def _delete_from_s3(s3_key, credentials, bucket_name='ivakt-footage'):
        s3 = boto3.client(
            's3',
            aws_access_key_id=credentials[0],
            aws_secret_access_key=credentials[1],
            region_name='us-west-1'
        )

        try:
            s3.delete_object(Bucket=bucket_name, Key=s3_key)
            print(f'Deleted {s3_key} from S3')
            return True
        except Exception as e:
            print(f"Failed to delete {s3_key} from S3: {e}")
            return False

    video_file = row[0]
    camera = video_file.split('.')[0].split('_')[-1]

    if not _download_from_s3(video_file, credentials):
        if os.path.exists('../input_files/' + video_file):
            os.remove('../input_files/' + video_file)
        return f"Failed to download {video_file}"

    hdf5_path = _get_hdf5_path(video_file)

    inference_pipeline = InferencePipeline(
        video_file, weights_paths, hdf5_path, device,
        track_stride=stride, id_stride=30
    )

    print(f"Detecting and embedding for {video_file}...")
    result = inference_pipeline.run()

    if not result:
        print(f"No detections in {video_file}")
        return f"No detections in {video_file}"

    print(f"Tracking for {video_file}...")
    all_trks = tracking_pipeline(video_file)

    io_utils.save_track_info(time_prefix, camera, all_trks)

    _delete_from_s3(video_file, credentials)

    return f"Processed {video_file} successfully"


def main(stride=15):
    multiprocessing.set_start_method("spawn")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    yolov4_weights = 'models/YOLOv4.pth'
    osnet_weights = 'models/OSNet.pth.tar-250'
    weights_paths = [yolov4_weights, osnet_weights]

    load_dotenv()
    credentials = [os.environ.get('AWS_ACCESS_KEY'),
                   os.environ.get('AWS_SECRET_KEY')]
    INTERNAL_API_KEY = os.environ.get('INTERNAL_API_KEY')
    headers = {
        'X-Custom-Api-Key': INTERNAL_API_KEY,
        'Content-Type': 'application/json'
    }
    base_url = 'https://ivaktvision-fe27c015e5ff.herokuapp.com/'
    queue_block_url = base_url + 'api/service/get_queue_block/'
    update_queue_url = base_url + 'api/service/update_queue/'

    while True:
        try:
            response = requests.get(queue_block_url, headers=headers)
            data = response.json()

            queue_block = data.get('results', [])
            if len(queue_block) == 0:
                print('No clips in the queue')
                time.sleep(60)
                continue
        except requests.exceptions.RequestException as e:
            print(f'Error making request: {e}')
            time.sleep(60)
            continue
        except Exception as e:
            print(f'Unexpected error: {e}')
            time.sleep(60)
            continue

        time_prefix = utilities.parse_clip_filename(queue_block[0][0], data='time')
        timestamp = utilities.frame_timestamp(time_prefix)
        
        with multiprocessing.Pool(processes=4) as pool:
            tasks = [
                (row, credentials, weights_paths, device, stride, time_prefix)
                for row in queue_block
            ]
            pool.starmap(process_row, tasks)

        io_utils.post_events_to_webapp(time_prefix)
        response = requests.post(
            update_queue_url, json={
                'action': 'clear_section', 'timestamp': timestamp.isoformat()},
            headers=headers
        )
        delete_files(time_prefix)

        #     identification_pipeline(time_prefix)

        #     io_utils.post_events_to_webapp(time_prefix)


        # else:
        #     io_utils.update_queue(action='clear_section', datetime=timestamp)
        #     delete_files(time_prefix)            
        # break

if __name__ == '__main__':
    main(stride=3)
