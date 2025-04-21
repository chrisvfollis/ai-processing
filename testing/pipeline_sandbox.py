# standard dependencies
import os
import sys
import multiprocessing
import threading
import pickle
import time

# 3rd-party dependencies
from dotenv import load_dotenv
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# internal dependencies
from utilities import io_utils
from utilities import general_utils as utils


# ----------------------------------------------------------------------------
# Inference:


def generate_inf_data(video_file, credentials, model_info, device, params=None,
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
        if not params:
            inf_pipeline = InferencePipeline(video_file, model_info, device)
        else:
            inf_pipeline = InferencePipeline(video_file, model_info, device,
                                             **params)
    except ValueError:
        print(f'Issue with {video_file}. Skipping...')
        return False

    inf_pipeline.run()
    inf_pipeline.save_inference_data()

    print(f"Processed {video_file}")

    return True


def run_inference(vid_files='input_dir', params=None):
    def _prepare():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        yolov4_weights = '../models/weights/YOLOv4.pth'
        osnet_weights = '../models/weights/OSNet.pth.tar-250'
        movenet_dir = '../models/weights/movenet'
        face_models = ['Facenet512', 'centerface']

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
        vid_files = sorted([f for f in results if f.endswith('.mp4')])
        tasks = [(vid_file, *start_vars, params) for vid_file in vid_files]
    elif vid_files == 'queue':
        qb_results = io_utils.get_queue_block()
        if qb_results:
            q_block = qb_results[0]
            tasks = [(row[0], *start_vars, params, 's3') for row in q_block]
    else:
        return None
    
    start_time = time.time()
    stop_event = threading.Event()

    time_logging = threading.Thread(
        target=utils.log_elapsed_time, args=(start_time, stop_event, 300, True), daemon=True
    )
    time_logging.start()

    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(
            generate_inf_data, tasks
        )
    
    stop_event.set()
    time_logging.join()


# ----------------------------------------------------------------------------
# Tracking:


def process_inf_output(video_file, device, output_dir='../files/output'):
    file_prefix = video_file.split('.')[0]
    t_prefix = utils.parse_clip_filename(video_file, data='time')
    filename = io_utils.get_latest_file(
        output_dir, f'{file_prefix}_inference_data.pkl'
    )
    data_path = os.path.join(output_dir, filename)

    with open(data_path, 'rb') as f:
        inference_output = pickle.load(f)

    from pipelines.tracking import TrackingPipeline

    trk_pipeline = TrackingPipeline(
        video_file, t_prefix, *inference_output, device, continuity=False
    )

    trk_pipeline.run()
    trk_pipeline.save_runtime_data()

    trk_pipeline.generate_output_vid()

    print(f"Processed {video_file}")

    return True


def run_tracking():
    multiprocessing.set_start_method("spawn")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = os.listdir('../files/input')
    vid_files = sorted([f for f in results if f.endswith('.mp4')])

    tasks = [(vid_file, device) for vid_file in vid_files]

    start_time = time.time()
    stop_event = threading.Event()

    time_logging = threading.Thread(
        target=utils.log_elapsed_time, args=(start_time, stop_event, 30, True), daemon=True
    )
    time_logging.start()

    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(
            process_inf_output, tasks
        )
    
    stop_event.set()
    time_logging.join()


if __name__ == '__main__':
    all_args = sys.argv

    if sys.argv[1] == 'inference':
        params = {
            'yolo_params': {'conf_thresh': 0.5}
        }

        if len(all_args) == 3:
            run_inference(vid_files=sys.argv[1], params=params)
        else:
            run_inference(params=params)
    elif sys.argv[1] == 'tracking':
        run_tracking()
