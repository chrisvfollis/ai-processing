# standard dependencies
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Optional

# 3rd-party dependencies
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.multiprocessing as multiprocessing

# internal dependencies
from utilities import io_utils, log_utils, conn_utils
from utilities.io_utils import S3DownloadError
from utilities import general_utils as utils
from pipelines import InferencePipeline, TrackingPipeline
from modules.identification import identify
from modules import render


logger = log_utils.get_logger(__name__)


# =============================================================================
#                      - INDIVIDUAL VIDEO PROCESSING -
# -----------------------------------------------------------------------------


def run_worker_pipeline(
    footage_record: tuple,
    model_cfg: dict,
    id_strategy: str,
    device: torch.device,
    log_level: int = 0,
    credentials: tuple[str, ...] = None,
    save_all_data: bool = False,
    f_cutoff: Optional[int] = None,
) -> bool:
    project_root = io_utils.get_project_root()

    input_dir = os.path.join(project_root, 'files/input/')
    output_dir = os.path.join(project_root, 'files/output/')

    log_utils.configure_logging(log_level=log_level)
    io_utils.clear_memory()

    shop_id, filename = footage_record[1:3]
    time_prefix, cam_id = utils.decode_vid_filename(filename)

    inference_cfg = {'model_cfg': model_cfg, 'device': device}
    if id_strategy == 'local':
        inference_cfg = inference_cfg | {
            'id_freq': '2 Hz',
            'use_features': True,
        }
    elif id_strategy == 'global':
        inference_cfg = inference_cfg | {
            'id_freq': '5 Hz',
            'use_features': False,
        }

    file_prefix = f'{time_prefix}_{cam_id}'
    object_key = f'{shop_id}/{filename}'
    
    process_result = False
    try:
        if not os.path.exists(os.path.join(input_dir, filename)):
            if not io_utils.download_s3_footage(object_key, credentials):
                raise S3DownloadError(f'Failed to download footage: {object_key}')

        inference = InferencePipeline(filename, **inference_cfg)
        if inference.skim(f_cutoff) == False:
            io_utils.delete_s3_footage(object_key, credentials)
            return process_result
        
        person_detections, face_data = inference.run(f_cutoff=f_cutoff)

        tracking = TrackingPipeline(filename, person_detections)

        active_trks, inactive_trks = tracking.run()
        tracking.filter_tracks()
        trk_detections, _ = tracking.format_track_data()

        if id_strategy == 'local':
            try:
                active_trks, inactive_trks = identify.identify_local_tracks(
                    face_data, active_trks, inactive_trks, trk_detections,
                    filename, credentials,
                )
                if save_all_data:
                    if face_data is not None and not face_data.empty:
                        face_data.to_csv(os.path.join(
                            output_dir, f'{file_prefix}_faces.csv'
                        ))
                    tracking.generate_output_vid(face_data=face_data)
            except Exception:
                logger.exception('Error during local ID')

            io_utils.save_track_info(
                time_prefix, cam_id, inactive_trks, tracking.fps
            )
        elif id_strategy == 'global':
            output_data = {
                'person_dets': inference.person_detection_df,
                'trk_dets': trk_detections,
                'region_log': pd.DataFrame(inference.region_log),
            }
            if (face_data is not None) and (not face_data.empty):
                output_data['faces'] = face_data

            for data_suffix, data in output_data.items():
                data.to_parquet(os.path.join(
                    output_dir, f'{file_prefix}_{data_suffix}.parquet'
                ))

        inference.save_run_info()
        tracking.save_run_info()

        tracking.save_state()

        logger.info(f'Processed {filename}')
        process_result = True
    except Exception:
        logger.exception(f'Error occurred while processing {filename}')
    finally:
        io_utils.clear_memory()

    return process_result


# =============================================================================
#                      - QUEUE SEGMENT PROCESSING -
# -----------------------------------------------------------------------------


def queue_segment_multiprocess(footage_records: list[tuple], process_config: tuple):
    logger.info('Starting processing run for queue block ...')

    footage_processing_tasks = [
        ((record,) + process_config) for record in footage_records
    ]

    with multiprocessing.Pool(processes=4) as pool:
        time.sleep(1)       # give workers a moment to start
        initial_pids = {p.pid for p in pool._pool if p.is_alive()}
        async_results = pool.starmap_async(
            run_worker_pipeline, footage_processing_tasks
        )
        worker_monitor = log_utils.observability_thread(
            'failed_workers', args=(pool, initial_pids, async_results),
            logger=logger
        )
        worker_monitor.start()
        async_results.get()


def wrap_up_segment(
    segment_filenames: list,
    time_prefix: str,
    shop_id: str,
    credentials: tuple[str],
    retain_footage: bool,
    save_all_data: bool,
):
    timestamp = utils.frame_timestamp(time_prefix)

    io_utils.post_event_data(shop_id, time_prefix, delete_data=True, logger=logger)
    io_utils.clear_queue_block(shop_id, timestamp)

    io_utils.clear_local_files(time_prefix, target_extensions=[
        '.hdf5',
        '.jpg',
        '.jpeg',
        '.png',
    ])
    if save_all_data == True:
        logger.info('Uploading data...')
        io_utils.upload_data(credentials)

    if retain_footage == True:
        io_utils.clear_local_files(time_prefix, skip_suffixes=['.mp4'])
        return
    else:
        object_keys = [f'{shop_id}/{filename}' for filename in segment_filenames]
        io_utils.delete_s3_footage(object_keys, credentials)

    io_utils.clear_local_files(time_prefix)


# =============================================================================
#                         - PRIMARY EXECUTION -
# -----------------------------------------------------------------------------


def main(
    shop_id: str,
    model_configs: list[dict],
    id_strategy: str,
    device: torch.device,
    log_level: int = 0,
    credentials: tuple[str] = None,
    save_all_data: bool = False,
    retain_footage: bool = False,
    starting_point: Optional[datetime] = None,
    priority_cam: Optional[int] = None,
    f_cutoff: Optional[int] = None,
):
    log_utils.configure_logging(log_level=log_level)

    process_cfg = (
        model_configs,
        id_strategy,
        device,
        log_level,
        credentials,
        save_all_data,
        f_cutoff,
    )
    basic_args = {
        'shop_id'        : shop_id,
        'credentials'    : credentials,
        'save_all_data'  : save_all_data,
        'retain_footage' : retain_footage,
    }

    dir_paths = io_utils.get_common_dirs()
    input_dir, output_dir, event_imgs_dir = [
        dir_paths[name] for name in ['input_dir', 'output_dir', 'event_imgs_dir']
    ]

    io_utils.clear_local_files(target_dirs=[output_dir, event_imgs_dir])
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

        time_logger, stop_timing = log_utils.observability_thread(
            target='elapsed_time', logger=logger
        )
        time_logger.start()

        queue_segment_multiprocess(queue_block_records, process_cfg)
        if id_strategy == 'global':
            logger.info('Running global identification...')
            results = identify.global_identification(
                time_prefix,
                output_dir,
                match_cutoff          = 0.25,
                mismatch_threshold    = 0.90,
                confidence_weight     = 0.45,
                distance_score_weight = 0.55,
                n_matches             = 1,
                min_score             = 0.50,
                reliability_scale     = 0.70,
                fp_rate               = 0.20,
                prior_presence        = 0.05,
                recall_est            = 0.65,
                # bias_score_boundary   = 0.75,
                bias_score_boundary   = 0.70,
                penalty_biases        = (0.5, 1.25),
                decay_window          = 0.9,
                boost_range           = (3.0, 5.0),
                max_decay             = 0.8,
                max_boost             = 0.7,
                boost_per_neighbor    = 0.05,
            )
            presence_df, filtered_faces, trk_dets = results
            logger.info('Finished global identification')

            if 'identity' in presence_df.columns and (
                presence_df['identity'].notna().any()
            ):
                event_imgs_df = io_utils.save_global_id_event_imgs(
                    time_prefix, presence_df, filtered_faces, trk_dets, credentials
                )
                io_utils.save_attendance_info(time_prefix, presence_df, event_imgs_df)
            else:
                logger.warning(
                    'Skipping event image generation — no valid identities found'
                )

            if save_all_data:
                id_results_paths = [
                    os.path.join(output_dir, f'{time_prefix}_{suffix}.csv')
                    for suffix in [
                        'presence_summary', 'filtered_faces', 'trk_dets'
                    ]
                ]
                presence_df.to_csv(id_results_paths[0], index=False)
                filtered_faces.to_csv(id_results_paths[1], index=False)
                trk_dets.to_csv(id_results_paths[2], index=False)

                raw_people, raw_faces, raw_trks, raw_regions = io_utils.load_raw_detection_data(time_prefix, output_dir)
                for filename in filenames:
                    if not filename.endswith('.mp4'):
                        continue

                    _, cam_id = utils.decode_vid_filename(filename)
                    person_dets_data = raw_people.loc[raw_people['cam_id'] == cam_id]
                    face_data = raw_faces.loc[raw_faces['cam_id'] == cam_id]
                    trk_dets_data = raw_trks.loc[raw_trks['cam_id'] == cam_id]
                    region_data = raw_regions.loc[raw_regions['cam_id'] == cam_id]

                    if face_data.empty and trk_dets_data.empty:
                        continue

                    try:
                        logger.info('Rendering video annotations...')
                        render.video.global_id_output(
                            filename,
                            person_df=person_dets_data,
                            face_df=face_data,
                            trk_df=trk_dets_data,
                            region_df=region_data,
                            f_cutoff=f_cutoff,
                        )
                        logger.info(f'Annotated video saved')
                    except Exception as e:
                        logger.exception(f'Failed to render annotated video for {filename}: {e}')

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
    parser.add_argument('--id-strategy', type=str, default='local')
    parser.add_argument('--f-cutoff', type=int, default=None)
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
        'checkpoint'  : 'yolox_model_trt.pth',
        'num_classes' : 1,
        'depth'       : 1.33,
        'width'       : 1.25,
        'input_size'  : (800, 1440),
        'conf_thresh' : 0.05,
        'nms_thresh'  : 0.7,
        'fp16'        : True,
        'use_trt'     : True,
    }
    faces_cfg = {
        'facenet_cfg': {
            'checkpoint' : 'facenet512_model_trt.pth',
            'fp16'       : False,
            'use_trt'    : True,
        },
        'centerface_cfg': {
            # 'conf_thresh' : 0.40,
            'conf_thresh' : 0.35,
            'min_area'    : (16, 16),
        },
        # 'clearface_cfg': {
        #     'checkpoint': '90000_G.pth',
        # }
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
        'shop_id'        : shop_id,
        'model_configs'  : model_cfgs,
        'id_strategy'    : args.id_strategy,
        'device'         : device,
        'log_level'      : args.log_level,
        'credentials'    : aws_credentials,
        'retain_footage' : args.retain_footage,
        'save_all_data'  : args.save_all_data,
        'starting_point' : starting_point,
        'priority_cam'   : args.priority_cam,
        'f_cutoff'       : args.f_cutoff,
    }

    main(**run_config)
