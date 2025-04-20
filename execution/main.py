# standard dependencies
import os
import sys
import traceback
import signal
import gc
import multiprocessing
import time

# 3rd-party dependencies
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# internal dependencies
from utilities import io_utils
from utilities import utilities as utils
from utilities.logger import get_logger


logger = get_logger(__name__)


def handle_early_termination(signum, frame):
    logger.info(f'Received {signum}. Cleaning up...')

    io_utils.clear_track_info('all')
    io_utils.delete_local_files('all')

    sys.exit(0)


def run_processing_pipelines(
        row, model_info, device, credentials, save_all_data=False
    ):
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
            logger.exception(f'Error configuring TensorFlow GPU memory: {e}')

    from pipelines.inference import InferencePipeline
    from pipelines.tracking import TrackingPipeline

    object_key = row[0]
    video_file = object_key.split('/')[-1]

    time_prefix, camera = utils.parse_clip_filename(video_file)

    if not io_utils.download_s3_footage(object_key, credentials):
        logger.warning(f'Failed to download footage: {object_key}')
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
        if save_all_data:
            inference_pipeline.save_inference_data()

        tracking_pipeline = TrackingPipeline(
            video_file, time_prefix,
            *inference_output,
            device,
            credentials
        )

        del inference_pipeline, inference_output
        io_utils.clear_memory()

        tracking_pipeline.run()
        if save_all_data:
            pass    # tracking pipeline data is already saved for continuation

        io_utils.save_track_info(
            time_prefix, camera, tracking_pipeline.inactive_trks,
            fps=tracking_pipeline.fps
        )
        if save_all_data:
            logger.info('Uploading data...')
            io_utils.upload_data(credentials)
        
        logger.info(f'Processed {video_file}')
        return True
    except Exception as e:
        logger.exception(f'Error occurred while processing {video_file}')   # logs the traceback automatically, so
        return False                                                        # no need for traceback.format_exc()


def run_master_process(
        device: torch.device,
        model_info: list,
        shop_id: str,
        credentials: dict,
        retain_footage: bool = False,
        save_all_data: bool = False
    ):
    def _clear_local_data():
        io_utils.clear_track_info('all')
        io_utils.delete_local_files('all')

    def _finalize(shop_id, queue_block, retain_footage=False):
        time_prefix = utils.parse_clip_filename(
            queue_block[0][0].split('/')[-1], data='time'
        )
        timestamp = utils.frame_timestamp(time_prefix)

        io_utils.post_events_to_webapp(time_prefix)

        object_keys = [row[0] for row in queue_block]
        if not retain_footage:
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

        tasks = [
            (row, model_info, device, credentials, save_all_data)
            for row in queue_block
        ]
        with multiprocessing.Pool(processes=3) as pool:
            time.sleep(1)   # ensure workers have enough time to start
            initial_pids = {p.pid for p in pool._pool if p.is_alive()}

            async_results = pool.starmap_async(
                run_processing_pipelines, tasks
            )

            worker_monitor = utils.observability_thread(
                'failed_workers', args=(pool, initial_pids, async_results)
            )
            worker_monitor.start()

            async_results.get()

        _finalize(shop_id, queue_block, retain_footage=retain_footage)
        stop_timing.set()
        time_logger.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    retain_footage = '--retain-footage' in sys.argv
    save_all_data = '--save-all-data' in sys.argv

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_info = [
        '../models/weights/YOLOv4.pth', '../models/weights/OSNet.pth.tar-250'
        , ('Facenet512', 'centerface_gpu')
    ]

    credentials = io_utils.get_aws_creds()
    shop_id, _ = io_utils.get_shop('../files/data.db')

    memory_monitor, _ = utils.observability_thread('low_memory')
    memory_monitor.start()

    run_master_process(
        device, model_info, shop_id, credentials,
        retain_footage=retain_footage,
        save_all_data=save_all_data
    )
