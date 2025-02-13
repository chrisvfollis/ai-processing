import torch
import multiprocessing
import time
import os
from dotenv import load_dotenv
import signal
import sys
from utilities import io_utils

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def process_video(row, credentials, model_info, device, time_prefix):
    video_file = row[0]
    camera = video_file.split('.')[0].split('_')[-1]

    if not io_utils.download_s3_footage(video_file, credentials):
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


    from pipelines.inference import InferencePipeline

    try:
        inf_pipeline = InferencePipeline(video_file, model_info, device)
    except ValueError:
        print(f'Issue with {video_file}. Skipping...')
        return False
    if not inf_pipeline.skim():
        io_utils.delete_s3_footage(video_file, credentials)
        return False
    else:
        inf_pipeline.run()
        inf_pipeline.save_inference_data()

    print(f"Processed {video_file}")

    return True


def run_processing():
    def _prepare():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        yolov4_weights = '../models/weights/YOLOv4.pth'
        osnet_weights = '../models/weights/OSNet.pth.tar-250'
        movenet_dir = '../models/weights/movenet'
        face_models = ['Facenet512', 'retinaface']

        model_info = [yolov4_weights, osnet_weights, movenet_dir,
                      face_models]

        load_dotenv()
        credentials = [os.environ.get('AWS_ACCESS_KEY'),
                    os.environ.get('AWS_SECRET_KEY')]

        return credentials, model_info, device

    multiprocessing.set_start_method("spawn")
    start_vars = _prepare()

    qb_results = io_utils.get_queue_block()
    if qb_results:
        q_block, t_prefix = qb_results[:2]

        tasks = [(row, *start_vars, t_prefix) for row in q_block]
        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(
                process_video, tasks
            )


if __name__ == '__main__':
    run_processing()
