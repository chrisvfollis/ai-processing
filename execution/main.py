# standard dependencies
import os
import sys
import multiprocessing
import time
import argparse
from datetime import datetime
from typing import Optional

# 3rd-party dependencies
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch

# internal dependencies
from utilities import io_utils, log_utils, conn_utils
from utilities.io_utils import S3DownloadError
from utilities import general_utils as utils
from pipelines import InferencePipeline, TrackingPipeline, IdentificationPipeline


logger = log_utils.get_logger(__name__)


# =============================================================================
#                      - QUEUE SEGMENT MULTIPROCESSING -
# -----------------------------------------------------------------------------


def process_queue_segment(footage_records: list[tuple], process_config: tuple):
    footage_processing_tasks = [
        ((record,) + process_config) for record in footage_records
    ]

    with multiprocessing.Pool(processes=3) as pool:
        time.sleep(1)   # ensure workers have enough time to start
        initial_pids = {p.pid for p in pool._pool if p.is_alive()}

        async_results = pool.starmap_async(
            run_pipelines, footage_processing_tasks
        )
        worker_monitor = log_utils.observability_thread(
            'failed_workers', args=(pool, initial_pids, async_results),
            logger=logger
        )
        worker_monitor.start()
        async_results.get()


def wrap_up_segment(
        shop_id: str,
        segment_filenames: list,
        time_prefix: str,
        credentials: tuple[str],
        retain_footage: bool,
        save_all_data: bool,
    ):
    timestamp = utils.frame_timestamp(time_prefix)

    io_utils.post_event_data(shop_id, time_prefix, delete_data=True)
    io_utils.clear_queue_block(shop_id, timestamp)

    io_utils.clear_local_files(time_prefix)

    if retain_footage == False:
        object_keys = [f'{shop_id}/{filename}' for filename in segment_filenames]
        io_utils.delete_s3_footage(object_keys, credentials)

    if save_all_data == True:
        logger.info('Uploading data...')
        io_utils.upload_data(credentials)


# =============================================================================
#                     - INDIVIDUAL VIDEO PROCESSING -
# -----------------------------------------------------------------------------


def run_pipelines(
        footage_record: tuple,
        model_cfg: dict,
        device: torch.device,
        log_level: int = 0,
        credentials: tuple[str, ...] = None,
        save_all_data=False,
    ) -> bool:

    log_utils.configure_logging(log_level=log_level)
    io_utils.clear_memory()

    shop_id, filename = footage_record[1:3]
    time_prefix, cam_id = utils.decode_vid_filename(filename)

    inference_cfg = {'model_cfg': model_cfg, 'device': device}
    tracking_cfg = {
        'credentials': credentials,
        'device': device,
        'log_level': log_level
    }
    
    process_result = False
    try:
        object_key = f'{shop_id}/{filename}'
        if not os.path.exists(
            os.path.join(io_utils.get_project_root(), 'files/input/', filename)
        ):
            if not io_utils.download_s3_footage(object_key, credentials):
                raise S3DownloadError(f'Failed to download footage: {object_key}')
        
        inference = InferencePipeline(filename, **inference_cfg)
        if inference.skim() == False:
            io_utils.delete_s3_footage(object_key, credentials)
            return process_result
        
        person_detections, face_data = inference.run()
        if save_all_data:
            inference.save_state()

        tracking = TrackingPipeline(filename, person_detections, **tracking_cfg)
        active_trks, inactive_trks = tracking.run()
        tracking.save_state()

        identification = IdentificationPipeline(
            filename, face_data, active_trks, inactive_trks
        )
        identification.run()
        identification.save_id_event_images(overlap_threshold=0.5)

        io_utils.save_track_info(
            time_prefix, cam_id, inactive_trks, tracking.fps
        )
        process_result = True
        logger.info(f'Processed {filename}')
    except Exception:
        logger.exception(f'Error occurred while processing {filename}')
    finally:
        io_utils.clear_memory()

    return process_result


# =============================================================================
#                         - PRIMARY EXECUTION -
# -----------------------------------------------------------------------------


def main(
        shop_id: str,
        model_configs: list[dict],
        device: torch.device,
        log_level: int = 0,
        credentials: tuple[str] = None,
        save_all_data: bool = False,
        retain_footage: bool = False,
        starting_point: Optional[datetime] = None,
        priority_cam: Optional[int] = None,
    ):
    log_utils.configure_logging(log_level=log_level)

    process_cfg = (model_configs, device, log_level, credentials, save_all_data)
    basic_args = {
        'shop_id': shop_id,
        'credentials': credentials,
        'save_all_data': save_all_data,
        'retain_footage': retain_footage,
    }

    io_utils.clear_local_files()
    io_utils.clear_track_info('all')

    while True:
        io_utils.cleanup_semaphores(logger)
        queue_block_records = io_utils.get_queue_block(
            shop_id, starting_point, priority_cam,
        )
        if not queue_block_records:
            time.sleep(60)
            continue
        else:
            filenames = [row[2] for row in queue_block_records]
            time_prefix, _ = utils.decode_vid_filename(filenames[0])

        time_logger, stop_timing  = log_utils.observability_thread(
            target='elapsed_time', logger=logger
        )
        time_logger.start()

        logger.info('Starting processing run for queue block ...')
        process_queue_segment(queue_block_records, process_cfg)

        stop_timing.set()
        time_logger.join()

        logger.info('Finalizing queue block...')
        wrap_up_segment(filenames, time_prefix, **basic_args)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', type=int, default=0)
    parser.add_argument('--retain-footage', action='store_true', default=False)
    parser.add_argument('--save-all-data', action='store_true', default=False)
    parser.add_argument('--start-from', type=str, help='Comma-separated datetime')
    parser.add_argument('--priority-cam', type=str)

    args = parser.parse_args()

    log_utils.configure_logging(log_level=args.log_level)

    starting_point = None
    if args.start_from:
        try:
            parts = [int(x) for x in args.start_from.split(',')]
            starting_point = datetime(*parts)
        except Exception as e:
            logger.error(f'Invalid --start-from value: {args.start_from} ({e})')
            sys.exit(1)
    
    device = utils.get_default_device()

    yolox_cfg = {
        'checkpoint': 'yolox_model_trt.pth',
        'num_classes': 1,
        'depth': 1.33,
        'width': 1.25,
        'input_size': (800, 1440),
        'conf_thresh': 0.05,
        'nms_thresh': 0.7,
        'fp16': True,
        'use_trt': True,
    }
    faces_cfg = {
        'facenet_cfg': {
            'checkpoint': 'facenet512_model_trt.pth',
            'fp16': False,
            'use_trt': True,
        },
        'centerface_cfg': {
            'conf_thresh': 0.65,
            'min_area': (40, 40),
        },
    }
    osnet_cfg = {}

    model_cfgs = {
        'yolox': yolox_cfg,
        'faces': faces_cfg,
        'osnet': osnet_cfg,
    }

    aws_credentials = conn_utils.get_aws_credentials()
    shop_id, _ = io_utils.get_shop()

    memory_monitor, _ = log_utils.observability_thread('low_memory', logger=logger)
    memory_monitor.start()

    run_config = {
        'shop_id': shop_id,
        'model_configs': model_cfgs,
        'device': device,
        'log_level': args.log_level,
        'credentials': aws_credentials,
        'retain_footage': args.retain_footage,
        'save_all_data': args.save_all_data,
        'starting_point': starting_point,
        'priority_cam': args.priority_cam,
    }

    main(**run_config)
