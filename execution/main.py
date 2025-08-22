# standard dependencies
import os
import sys
import time
from datetime import datetime
from typing import Optional

# 3rd-party dependencies
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.multiprocessing as multiprocessing

# internal dependencies
from execution import configure
from utilities import utils, io_utils, log_utils, conn_utils
from utilities.io_utils import S3DownloadError
from pipelines import InferencePipeline, TrackingPipeline
from modules.identification import identify
from modules.identification.data_structures import AssessIdPresenceParams
from modules import results
from modules.results import render


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
    credentials: tuple[str, str] = None,
    save_all_data: bool = False,
    f_cutoff: Optional[int] = None,
) -> bool:
    project_root = io_utils.get_project_root()

    input_dir = os.path.join(project_root, 'files/input/')
    output_dir = os.path.join(project_root, 'files/output/')

    log_utils.configure_logging(log_level=log_level)
    io_utils.clear_memory()

    shop_uuid, filename = footage_record[1:3]
    time_segment, cam_id = utils.decode_vid_filename(filename)

    inference_cfg = {'model_cfg': model_cfg, 'device': device}
    if id_strategy == 'tracks':
        strategy_params = {
            'stride'          : 1,
            'id_inference_Hz' : 2,
            'use_features'    : True,
        }
    elif id_strategy == 'presence':
        strategy_params = {
            # 'stride'          : 1,
            'stride'          : 2,
            'id_inference_Hz' : 5,
            'use_features'    : False,
        }
    inference_cfg = inference_cfg | strategy_params

    file_prefix  = f'{time_segment}_{cam_id}'
    local_path   = os.path.join(input_dir, filename)
    object_key   = f'{shop_uuid}/{filename}'
    footage_info = (local_path, object_key)

    worker_pipeline_result = False
    try:
        status = io_utils.ensure_footage(*footage_info, credentials)
        if status != True:
            if status == 'NoSuchKey':
                return worker_pipeline_result
            raise S3DownloadError(f'Failed to download {object_key}')

        inference = InferencePipeline(filename, **inference_cfg)
        
        if inference.skim(f_cutoff) == False:
            io_utils.delete_s3_footage(object_key, credentials)
            return worker_pipeline_result
        
        person_detections, face_data = inference.run(batch_size=16, f_cutoff=f_cutoff)

        tracking = TrackingPipeline(filename, person_detections)

        active_trks, inactive_trks = tracking.run()
        tracking.filter_tracks()
        trk_detections, _ = tracking.format_track_data()

        if id_strategy == 'tracks':
            try:
                active_trks, inactive_trks = identify.assign_track_identities(
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

            results.records.save_tracks(
                time_segment, cam_id, inactive_trks, tracking.fps
            )
        elif id_strategy == 'presence':
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
        worker_pipeline_result = True
    except Exception:
        logger.exception(f'Error occurred while processing {filename}')
    finally:
        try:
            del inference, tracking
            del person_detections, face_data, trk_detections
        except NameError:
            pass
        io_utils.clear_memory()

    return worker_pipeline_result


# =============================================================================
#                    - QUEUED TIME SEGMENT PROCESSING -
# -----------------------------------------------------------------------------


def process_records(footage_records: list[tuple], process_config: tuple):
    logger.info('Processing time segment records...')

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
    time_segment: str,
    shop_uuid: str,
    credentials: tuple[str],
    retain_footage: bool,
    save_all_data: bool,
):
    logger.info('Finalizing time segment...')
    
    results.records.post_event_records(shop_uuid, time_segment)

    io_utils.clear_event_records(time_segment)
    io_utils.dequeue_segment(shop_uuid, time_segment)

    io_utils.clear_local_files(time_segment, target_extensions=[
        '.hdf5',
        '.png',
        '.jpg',
        '.jpeg',
    ])
    if save_all_data == True:
        logger.info('Uploading data...')
        io_utils.upload_data(credentials)

    if retain_footage == True:
        io_utils.clear_local_files(time_segment, skip_suffixes=['.mp4'])
        return
    else:
        object_keys = [f'{shop_uuid}/{filename}' for filename in segment_filenames]
        io_utils.delete_s3_footage(object_keys, credentials)

    io_utils.clear_local_files(time_segment)


# =============================================================================
#                         - PRIMARY EXECUTION -
# -----------------------------------------------------------------------------


def main(
    id_strategy: str,
    model_configs: list[dict],
    device: torch.device,
    shop_uuid: str,
    credentials: tuple[str] = None,
    log_level: int = 0,
    save_all_data: bool = False,
    retain_footage: bool = False,
    start_from: Optional[datetime] = None,
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
        'shop_uuid'      : shop_uuid,
        'credentials'    : credentials,
        'save_all_data'  : save_all_data,
        'retain_footage' : retain_footage,
    }

    dir_paths = io_utils.get_common_dirs()
    output_dir, event_imgs_dir = [
        dir_paths[name] for name in ['output_dir', 'event_imgs_dir']
    ]

    io_utils.clear_local_files(target_dirs=[output_dir, event_imgs_dir])
    io_utils.clear_event_records('all')

    while True:
        io_utils.cleanup_semaphores(logger)
        segment_records = io_utils.get_next_queue_segment(
            shop_uuid, start_from, priority_cam,
        )
        if not segment_records:
            time.sleep(60)
            continue
        
        time_segment, filenames = utils.get_segment_info(segment_records)
        elapsed_time_logs, stop_timing = log_utils.observability_thread(
            target='elapsed_time', logger=logger
        )
        elapsed_time_logs.start()

        process_records(segment_records, process_cfg)

        if id_strategy == 'presence':
            try:
                processing_output = io_utils.load_processing_output(time_segment)
                face_data, trk_dets = processing_output[2:]
            except ValueError:
                logger.info('No results to process')

                stop_timing.set()
                elapsed_time_logs.join()
                
                wrap_up_segment(filenames, time_segment, **basic_args)
                continue
                
            logger.info('Running global identification...')
            identity_presence_params = AssessIdPresenceParams(
                match_cutoff          = 0.25,
                mismatch_threshold    = 0.90,
                distance_score_weight = 0.55,
                confidence_weight     = 0.45,
                n_matches             = 2,
                min_score             = 0.45,
                reliability_scale     = 0.725,
                fp_rate               = 0.20,
                presence_prior        = 0.05,
                bias_score_boundary   = 0.75,
                penalty_biases        = (0.5, 1.25),
                decay_window          = 0.9,
                boost_range           = (3.0, 5.0),
                max_decay             = 0.6,
                max_boost             = 0.9,
                boost_per_neighbor    = 0.075,
                fallback_recall_est   = 0.60,
                presence_thresh       = 0.55,
            )
            identity_presence_params = identity_presence_params.as_dict()

            presence_df, filtered_faces = identify.assess_present_identities(
                face_data, **identity_presence_params
            )
            # subsegment_results = identify.subsegment_identity_sweep(
            #     face_data, presence_df, identity_presence_params,
            #     full_duration=300, sub_duration=60
            # )
            logger.info('Finished global identification')

            if (
                ('identity' in presence_df.columns) and
                (presence_df['identity'].notna().any()) and
                ('cam_id' in trk_dets.columns)
            ):
                logger.info('Generating event images...')
                event_imgs_df = results.images.global_id_event_imgs(
                    time_segment, presence_df, filtered_faces, trk_dets,
                    credentials, min_frame_delta=60
                )
                results.records.save_attendance(time_segment, presence_df, event_imgs_df)
            else:
                logger.warning(
                    'Skipping event image generation — no valid identities found'
                )

            if save_all_data:
                id_results_paths = [
                    os.path.join(output_dir, f'{time_segment}_{suffix}.csv')
                    for suffix in [
                        'presence_summary', 'filtered_faces', 'trk_dets'
                    ]
                ]
                presence_df.to_csv(id_results_paths[0], index=False)
                filtered_faces.to_csv(id_results_paths[1], index=False)
                trk_dets.to_csv(id_results_paths[2], index=False)
                
                person_det_data, region_log_data = processing_output[:2]
                for filename in filenames:
                    if not filename.endswith('.mp4'):
                        continue

                    _, cam_id = utils.decode_vid_filename(filename)
                    cam_person_det_data = person_det_data.loc[person_det_data['cam_id'] == cam_id]
                    cam_region_log_data = region_log_data.loc[region_log_data['cam_id'] == cam_id]
                    cam_face_data = face_data.loc[face_data['cam_id'] == cam_id]
                    cam_trk_dets  = trk_dets.loc[trk_dets['cam_id'] == cam_id]
                    
                    if cam_face_data.empty and cam_trk_dets.empty:
                        continue

                    # try:
                    #     logger.info('Rendering video annotations...')
                    #     render.video.global_id_output(
                    #         filename,
                    #         person_df = cam_person_det_data,
                    #         face_df   = cam_face_data,
                    #         trk_df    = cam_trk_dets,
                    #         region_df = cam_region_log_data,
                    #         f_cutoff  = f_cutoff,
                    #     )
                    #     logger.info(f'Annotated video saved')
                    # except Exception as e:
                    #     logger.exception(f'Failed to render annotated video for {filename}: {e}')

        stop_timing.set()
        elapsed_time_logs.join()

        wrap_up_segment(filenames, time_segment, **basic_args)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    parser = configure.make_parser()
    args = parser.parse_args()

    log_utils.configure_logging(log_level=args.log_level)

    start_from = None
    if args.start_from:
        try:
            datetime_pieces = [int(x) for x in args.start_from.split(',')]
            start_from = datetime(*datetime_pieces)
        except Exception as e:
            logger.error(f'Invalid --start-from value: {args.start_from} ({e})')
            sys.exit(1)
    
    memory_monitor, _ = log_utils.observability_thread('low_memory', logger=logger)
    memory_monitor.start()

    main(
        id_strategy    = args.id_strategy,
        model_configs  = configure.package_model_cfgs(),
        device         = utils.get_default_device(),
        shop_uuid      = io_utils.get_shop()[0],
        log_level      = args.log_level,
        retain_footage = args.retain_footage,
        save_all_data  = args.save_all_data,
        credentials    = conn_utils.get_aws_credentials(),
        start_from     = start_from,
        priority_cam   = args.priority_cam,
        f_cutoff       = args.f_cutoff,
    )
