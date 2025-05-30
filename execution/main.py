# standard dependencies
import os
import sys
import signal
import multiprocessing
import time
import argparse
from datetime import datetime
import textwrap

# 3rd-party dependencies
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch

# internal dependencies
from utilities import io_utils, log_utils, conn_utils
from utilities import general_utils as utils


logger = log_utils.get_logger(__name__)


def handle_early_termination(signum, frame):
    logger.info(f'Received {signum}. Cleaning up...')

    io_utils.clear_track_info('all')
    io_utils.delete_local_files('all')

    sys.exit(0)


def run_processing_pipelines(
        row, model_info, device, credentials, log_level=0, save_all_data=False
    ):
    log_utils.configure_logging(log_level=log_level)
    io_utils.clear_memory()

    from pipelines import InferencePipeline, TrackingPipeline

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
            inference_pipeline.save_pipeline_state()

        tracking_pipeline = TrackingPipeline(
            video_file, time_prefix,
            *inference_output,
            credentials,
            device=device,
            log_level=log_level
        )

        del inference_pipeline, inference_output
        
        tracking_pipeline.run()
        if save_all_data:
            pass    # tracking pipeline data is already saved for continuation

        io_utils.save_track_info(
            time_prefix, camera, tracking_pipeline.inactive_trks,
            fps=tracking_pipeline.fps
        )
        
        logger.info(f'Processed {video_file}')
        return True
    except Exception as e:
        logger.exception(f'Error occurred while processing {video_file}')   # logs the traceback automatically, so
        return False                                                        # no need for traceback.format_exc()
    finally:
        io_utils.clear_memory()


def run_master_process(
        device: torch.device,
        model_info: list,
        shop_id: str,
        credentials: dict,
        log_level: int = 0,
        retain_footage: bool = False,
        save_all_data: bool = False,
        start_from=None,
        priority_cam=None,
    ):
    def _clear_local_data():
        io_utils.clear_track_info('all')
        io_utils.delete_local_files('all')

    def _finalize(shop_id, queue_block, retain_footage=False):
        logger.info(textwrap.dedent(f'''
            finalizing queue block...
            retain_footage = {retain_footage}
        '''))

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
    
    log_utils.configure_logging(log_level=log_level)

    logger.info(textwrap.dedent(f'''
        master process args:
        retain_footag = {retain_footage}
        save_all_data = {save_all_data}
        start_from = {start_from}
        priority_cam = {priority_cam}
        log_level = {log_level}
    '''))

    signal.signal(signal.SIGTERM, handle_early_termination)
    signal.signal(signal.SIGINT, handle_early_termination)

    _clear_local_data()

    while True:
        io_utils.cleanup_semaphores()

        queue_block = io_utils.get_queue_block(
            shop_id,
            start_from=start_from,
            priority_camera=priority_cam
        )

        if not queue_block:
            time.sleep(60)
            continue

        time_logger, stop_timing = log_utils.observability_thread(
            'elapsed_time', logger=logger
        )
        time_logger.start()

        tasks = [
            (row, model_info, device, credentials, log_level, save_all_data)
            for row in queue_block
        ]
        with multiprocessing.Pool(processes=3) as pool:
            time.sleep(1)   # ensure workers have enough time to start
            initial_pids = {p.pid for p in pool._pool if p.is_alive()}

            async_results = pool.starmap_async(
                run_processing_pipelines, tasks
            )

            worker_monitor = log_utils.observability_thread(
                'failed_workers', args=(pool, initial_pids, async_results),
                logger=logger
            )
            worker_monitor.start()

            async_results.get()

        _finalize(shop_id, queue_block, retain_footage=retain_footage)
        if save_all_data:
            logger.info('Uploading data...')
            io_utils.upload_data(credentials)
        stop_timing.set()
        time_logger.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--retain-footage', action='store_true')
    parser.add_argument('--save-all-data', action='store_true')
    parser.add_argument('--start-from', type=str, help='Comma-separated datetime')
    parser.add_argument('--priority-cam', type=str)
    parser.add_argument('--debug-level', type=int)

    args = parser.parse_args()

    retain_footage = args.retain_footage
    save_all_data = args.save_all_data
    priority_camera = args.priority_cam

    log_level = args.log_level or 0
    log_utils.configure_logging(log_level=log_level)

    start_from = None
    if args.start_from:
        try:
            parts = [int(x) for x in args.start_from.split(',')]
            start_from = datetime(*parts)
        except Exception as e:
            logger.error(f'Invalid --start-from value: {args.start_from} ({e})')
            sys.exit(1)
    
    device = utils.get_default_device()
    model_info = [
        'YOLOv4.pth', 'OSNet.pth.tar-250', ('Facenet512', 'centerface_gpu')
    ]

    credentials = conn_utils.get_aws_credentials()
    shop_id, _ = io_utils.get_shop()

    memory_monitor, _ = log_utils.observability_thread('low_memory', logger=logger)
    memory_monitor.start()

    run_master_process(
        device, model_info, shop_id, credentials,
        retain_footage=retain_footage,
        save_all_data=save_all_data,
        start_from=start_from,
        priority_cam=priority_camera,
        log_level=log_level,
    )
