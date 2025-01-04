import torch
import io_utils
import multiprocessing
import time
import utilities
import os
import requests
from dotenv import load_dotenv
import boto3
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def delete_local_files(identifier, file_types='any',
                 paths=['../input_files', '../intermediate_output',
                        '../output_files/event_imgs']):
    def _parse_name_and_extension(file):
        file_parts = [x for x in file.rsplit('.', 1)]

        name = file_parts[0]
        extension = (
            file_parts[-1] if len(file_parts) == 2 else ''
        )
    
        return name, extension

    n_deleted = 0
    for path in paths:
        if not os.path.exists(path):
            print(f'Skipping non-existent path: {path}')
            continue
        for result in os.listdir(path):
            full_path = os.path.join(path, result)
            if not os.path.isfile(full_path):
                continue
            
            file_name, file_extension = _parse_name_and_extension(result)
            if (
                ((identifier == 'all') or (file_name.startswith(identifier))) and
                ((file_types == 'any') or (file_extension in file_types))
            ):
                try:
                    os.remove(full_path)
                    n_deleted += 1
                except Exception as e:
                    print(f'Error deleting {full_path}: {e}')

    print(f'Deleted {n_deleted} files')
    return True


def delete_from_s3(object_key, credentials, bucket_name='ivakt-footage'):
    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials[0],
        aws_secret_access_key=credentials[1],
        region_name='us-west-1'
    )

    try:
        s3.delete_object(Bucket=bucket_name, Key=object_key)
        print(f'Deleted {object_key} from S3')
        return True
    except Exception as e:
        print(f"Failed to delete {object_key} from S3: {e}")
        return False


def process_row(row, credentials, model_paths, device, stride, time_prefix):
    def _download_from_s3(object_key, credentials, bucket_name='ivakt-footage'):
        s3 = boto3.client(
            's3',
            aws_access_key_id=credentials[0],
            aws_secret_access_key=credentials[1],
            region_name='us-west-1'
        )

        try:
            local_path = os.path.join('../input_files', object_key)
            s3.download_file(bucket_name, object_key, local_path)
            print(f'Downloaded {object_key}')
            return True
        except Exception as e:
            print(f"Failed to download {object_key}: {e}")
            return False

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
        except RuntimeError as e:
            print(f"Error configuring TensorFlow GPU memory: {e}")

    from inference import InferencePipeline
    from tracking import tracking_pipeline

    video_file = row[0]
    camera = video_file.split('.')[0].split('_')[-1]

    if not _download_from_s3(video_file, credentials):
        if os.path.exists('../input_files/' + video_file):
            os.remove('../input_files/' + video_file)
        print(f"Failed to download {video_file}")
        return False

    inference_pipeline = InferencePipeline(video_file, model_paths, device,
                                           stride=stride)

    print(f"Running inference pipeline for {video_file}...")
    result = inference_pipeline.run()

    if not result:
        print(f"No detections in {video_file}")
        return False

    print(f"Running tracking pipeline for {video_file}...")
    all_trks = tracking_pipeline(video_file)
    io_utils.save_track_info(time_prefix, camera, all_trks)

    print(f"Processed {video_file} successfully")
    return True


def main(stride=3):
    def _qb_time_info(queue_block):
        time_prefix = utilities.parse_clip_filename(queue_block[0][0],
                                                    data='time')
        timestamp = utilities.frame_timestamp(time_prefix)
        return time_prefix, timestamp

    multiprocessing.set_start_method("spawn")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    yolov4_weights = 'models/YOLOv4.pth'
    osnet_weights = 'models/OSNet.pth.tar-250'
    movenet_dir = 'models/movenet'
    model_paths = [yolov4_weights, osnet_weights, movenet_dir]

    load_dotenv()
    credentials = [os.environ.get('AWS_ACCESS_KEY'),
                   os.environ.get('AWS_SECRET_KEY')]

    io_utils.clear_track_info('all')
    delete_local_files('all')

    while True:
        queue_block = io_utils.get_queue_block()

        if not queue_block:
            time.sleep(60)
            continue

        time_prefix, timestamp = _qb_time_info(queue_block)
        tasks = [(row, credentials, model_paths, device, stride,
                  time_prefix) for row in queue_block]
        
        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(process_row, tasks)

        if io_utils.post_events_to_webapp(time_prefix):
            video_files = [row[0] for row in queue_block]
            for video_file in video_files:
                delete_from_s3(video_file, credentials)

            io_utils.clear_queue_block(timestamp)

        delete_local_files(time_prefix)


if __name__ == '__main__':
    main(stride=3)
