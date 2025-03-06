import os
import multiprocessing
import torch
import time
from dotenv import load_dotenv
import sys
from utilities import io_utils
from utilities import utilities as utils
import threading
import gc

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

multiprocessing.set_start_method('spawn')


def handle_early_termination(signum, frame):
    print(f'Received {signum}. Cleaning up...')

    io_utils.clear_track_info('all')
    io_utils.delete_local_files('all')

    io_utils.cleanup_semaphores()

    sys.exit(0)


def process_video(row, model_info, device):
    gc.set_debug(gc.DEBUG_SAVEALL)

    K = tf.keras.backend
    torch.cuda.empty_cache()
    K.clear_session()
    gc.collect()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
        except RuntimeError as e:
            print(f"Error configuring TensorFlow GPU memory: {e}")

    from pipelines.tracking import TrackingPipeline

    video_file = row[0]
    time_prefix, camera = utils.parse_clip_filename(video_file)

    if not io_utils.download_s3_footage(video_file):
        return False

    try:
        from pipelines.inference import InferencePipeline
        inf_pipeline = InferencePipeline(video_file, model_info, device)
    except ValueError:
        print(f'Issue with {video_file}. Skipping...')
        return False
    if not inf_pipeline.skim():
        io_utils.delete_s3_footage(video_file)
        return False
    
    inference_output = inf_pipeline.run()

    trk_pipeline = TrackingPipeline(
        video_file, time_prefix, *inference_output, device
    )

    del inf_pipeline
    del inference_output

    torch.cuda.empty_cache()
    K.clear_session()
    gc.collect()

    trk_pipeline.run()

    all_trks = trk_pipeline.all_trks
    fps = inf_pipeline.fps

    io_utils.save_track_info(
        time_prefix, camera, all_trks, fps=fps
    )

    print(f"Processed {video_file}")

    del inference_output
    del trk_pipeline

    torch.cuda.empty_cache()
    K.clear_session()
    gc.collect()


def run_pipeline(device, model_info):
    def _clear_data():
        io_utils.clear_track_info('all')
        io_utils.delete_local_files('all')

    def _setup_logging():
        stop_event = threading.Event()
        time_logger = threading.Thread(
            target=utils.log_elapsed_time, args=(time.time(), stop_event),
            daemon=True
        )
        return time_logger, stop_event

    def _finalize(queue_block):
        time_prefix = utils.parse_clip_filename(queue_block[0][0], data='time')
        timestamp = utils.frame_timestamp(time_prefix)

        io_utils.post_events_to_webapp(time_prefix)

        video_files = [row[0] for row in queue_block]
        for video_file in video_files:
            io_utils.delete_s3_footage(video_file)

        io_utils.clear_queue_block(timestamp)
        io_utils.delete_local_files(time_prefix)

    while True:
        io_utils.cleanup_semaphores()

        queue_block = io_utils.get_queue_block()
    
        if not queue_block:
            time.sleep(60)
            continue

        cycle_time_logger, stop_event = _setup_logging()
        cycle_time_logger.start()

        tasks = [(row, model_info, device) for row in queue_block]
        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(
                process_video, tasks
            )
        
        _finalize(queue_block)
        stop_event.set()
        cycle_time_logger.join()
        

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model_info = [
        '../models/weights/YOLOv4.pth', '../models/weights/OSNet.pth.tar-250',
        '../models/weights/movenet', ('Facenet512','centerface')
    ]

    oom_watchdog = threading.Thread(target=utils.monitor_memory, daemon=True)
    oom_watchdog.start()

    run_pipeline(device, model_info)
