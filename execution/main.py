import torch
import multiprocessing
import time
import os
from dotenv import load_dotenv
import signal
import sys
from utilities import io_utils
from utilities import utilities as utils
import tracemalloc
import threading

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import gc

tracemalloc.start()

def handle_sigterm(signum, frame):
    print("Received SIGTERM. Cleaning up...")

    io_utils.clear_track_info('all')
    io_utils.delete_local_files('all')

    sys.exit(0)


def monitor_memory(interval=5):
    '''
    Logs the top 5 memory-consuming lines every `interval` seconds.
    '''
    while True:
        time.sleep(interval)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        print("\n[MEMORY MONITOR] Top memory-consuming lines:")
        for stat in top_stats[:5]:
            print(stat)


def process_video(row, credentials, model_info, device, time_prefix):
    video_file = row[0]
    camera = video_file.split('.')[0].split('_')[-1]

    memory_monitoring = threading.Thread(
        target=monitor_memory,
        args=(2,),
        daemon=True
    )
    memory_monitoring.start()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    K = tf.keras.backend
    K.clear_session()
    gc.collect()

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
    from pipelines.tracking import TrackingPipeline

    try:
        inf_pipeline = InferencePipeline(video_file, model_info, device)
    except ValueError:
        print(f'Issue with {video_file}. Skipping...')
        return False
    if not inf_pipeline.skim():
        io_utils.delete_s3_footage(video_file, credentials)
        return False
    else:
        inference_output = inf_pipeline.run()

    trk_pipeline = TrackingPipeline(
        video_file, time_prefix, *inference_output, device
    )   

    trk_pipeline.run()

    all_trks = trk_pipeline.all_trks
    fps = inf_pipeline.fps

    io_utils.save_track_info(
        time_prefix, camera, all_trks, fps=fps
    )

    print(f"Processed {video_file}")

    K.clear_session()
    gc.collect()

    return True


def run_processing_cycle():
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

        io_utils.clear_track_info('all')
        io_utils.delete_local_files('all')

        return credentials, model_info, device
    
    def _finalize(queue_block, time_prefix, timestamp, credentials):
        io_utils.post_events_to_webapp(time_prefix)

        video_files = [row[0] for row in queue_block]
        for video_file in video_files:
            io_utils.delete_s3_footage(video_file, credentials)

        io_utils.clear_queue_block(timestamp)
        io_utils.delete_local_files(time_prefix)

    signal.signal(signal.SIGTERM, handle_sigterm)
    multiprocessing.set_start_method('spawn')
    start_vars = _prepare()

    while True:
        qb_results = io_utils.get_queue_block()
        if qb_results:
            q_block, t_prefix = qb_results[:2]
        else:
            time.sleep(60)
            continue

        # start_time = time.time()
        # stop_event = threading.Event()
        # cycle_time_logging = threading.Thread(
        #     target=utils.log_elapsed_time, args=(start_time, stop_event),
        #     daemon=True
        # )
        # cycle_time_logging.start()

        tasks = [(row, *start_vars, t_prefix) for row in q_block]
        with multiprocessing.Pool(processes=1) as pool:
            pool.starmap(
                process_video, tasks
            )
            pool.close()
            pool.join()
        
        _finalize(*qb_results, start_vars[0])

        # stop_event.set()
        # cycle_time_logging.join()
        

if __name__ == '__main__':
    # stop_memory_monitoring = threading.Event()

    run_processing_cycle()

    # stop_memory_monitoring.set()
    # memory_monitoring.join()
