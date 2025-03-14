import os
import multiprocessing
import torch
import time
import sys
from utilities import io_utils
from utilities import utilities as utils
import gc
import signal
import traceback

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def handle_early_termination(signum, frame):
    print(f'Received {signum}. Cleaning up...')

    io_utils.clear_track_info('all')
    io_utils.delete_local_files('all')

    sys.exit(0)


def run_processing_pipelines(row, model_info, device, credentials):
    gc.set_debug(gc.DEBUG_SAVEALL)

    io_utils.clear_memory()

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

    object_key = row[0]
    video_file = object_key.split('/')[-1]

    time_prefix, camera = utils.parse_clip_filename(video_file)

    if not io_utils.download_s3_footage(object_key, credentials):
        return False

    try:
        inference_pipeline = InferencePipeline(
            video_file, model_info,
            device
        )
        valid = inference_pipeline.skim()
        if not valid:
            io_utils.delete_s3_footage(object_key, credentials)
            return False
        
        inference_output = inference_pipeline.run()

        tracking_pipeline = TrackingPipeline(
            video_file, time_prefix,
            *inference_output,
            device,
            credentials
        )

        del inference_pipeline, inference_output
        io_utils.clear_memory()

        tracking_pipeline.run()

        io_utils.save_track_info(
            time_prefix, camera, tracking_pipeline.inactive_trks,
            fps=tracking_pipeline.fps
        )
        print(f"Processed {video_file}")
        return True
    except Exception as e:
        print(f'Error: {e}')
        print(traceback.format_exc())
        return False


def run_master_process(device, model_info, shop_id, credentials):
    def _clear_local_data():
        io_utils.clear_track_info('all')
        io_utils.delete_local_files('all')

    def _finalize(shop_id, queue_block):
        time_prefix = utils.parse_clip_filename(
            queue_block[0][0].split('/')[-1], data='time'
        )
        timestamp = utils.frame_timestamp(time_prefix)

        io_utils.post_events_to_webapp(time_prefix)

        object_keys = [row[0] for row in queue_block]
        for object_key in object_keys:
            io_utils.delete_s3_footage(object_key, credentials)

        io_utils.clear_queue_block(shop_id, timestamp)
        io_utils.delete_local_files(time_prefix)
    
    signal.signal(signal.SIGTERM, handle_early_termination)
    signal.signal(signal.SIGINT, handle_early_termination)

    _clear_local_data()

    while True:
        io_utils.cleanup_semaphores()

        queue_block = io_utils.get_queue_block(shop_id)

        if not queue_block:
            time.sleep(60)
            continue

        time_logger, stop_timing = utils.observability_thread('elapsed_time')
        time_logger.start()

        tasks = [(row, model_info, device, credentials) for row in queue_block]
        with multiprocessing.Pool(processes=3) as pool:
            time.sleep(1)   # Ensure workers have enough time to start
            initial_pids = {p.pid for p in pool._pool if p.is_alive()}

            async_results = pool.starmap_async(
                run_processing_pipelines, tasks
            )

            worker_monitor = utils.observability_thread(
                'failed_workers', args=(pool, initial_pids, async_results)
            )
            worker_monitor.start()
            
            async_results.get()

        _finalize(shop_id, queue_block)
        stop_timing.set()
        time_logger.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    device = torch.device('cuda:0')
    model_info = [
        '../models/weights/YOLOv4.pth', '../models/weights/OSNet.pth.tar-250'
        , ('Facenet512', 'centerface_gpu')
    ]

    credentials = io_utils.get_aws_creds()
    shop_id = io_utils.get_shop('../files/data.db')

    memory_monitor, _ = utils.observability_thread('low_memory')
    memory_monitor.start()

    run_master_process(device, model_info, shop_id, credentials)
