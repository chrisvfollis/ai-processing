import torch
import multiprocessing
import time
import os
from dotenv import load_dotenv
import signal
import sys
from utilities import io_utils
from utilities import utilities as utils
import threading

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def generate_inf_data(video_file, credentials, model_info, device,
                      location='../files/input'):
    if location == 's3':
        print('Attempting download...')
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

    inf_pipeline.run()
    inf_pipeline.save_inference_data()

    print(f"Processed {video_file}")

    return True


def run_process(vid_files='input_dir', inf_params=None):
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

    if vid_files == 'input_dir':
        results = os.listdir('../files/input')
        vid_files = [f for f in results if f.endswith('.mp4')]
        tasks = [(vid_file, *start_vars) for vid_file in vid_files]
    elif vid_files == 'queue':
        qb_results = io_utils.get_queue_block()
        if qb_results:
            q_block = qb_results[0]
            tasks = [(row[0], *start_vars, 's3') for row in q_block]
    else:
        return None
    
    start_time = time.time()
    stop_event = threading.Event()

    time_logging = threading.Thread(
        target=utils.log_elapsed_time, args=(start_time, stop_event), daemon=True
    )
    time_logging.start()

    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(
            generate_inf_data, tasks
        )
    
    stop_event.set()


if __name__ == '__main__':
    all_args = sys.argv

    if len(all_args) == 2:
        run_process(vid_files=sys.argv[1])
    else:
        run_process()
