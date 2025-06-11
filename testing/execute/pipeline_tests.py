# standard dependencies
import os
import argparse
import pickle

# 3rd-party dependencies
pass

# internal dependencies
from utilities import io_utils, log_utils, conn_utils
import utilities.general_utils as utils

logger = log_utils.get_logger(__name__)


def run_inference_pipeline(
        video_file: str,
        log_level: int = 0
    ):
    '''
    Runs the inference pipeline on a video file and saves the pipeline state.

    Args:
        video_file (str): Name of the local video file.
        model_info: Model paths or identifiers required by InferencePipeline.
        device (torch.device): The device on which to run the model.
        log_level (int): Logging verbosity level.
    '''

    log_utils.configure_logging(log_level=log_level)
    io_utils.clear_memory()

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

    from pipelines import InferencePipeline

    try:
        inference_pipeline = InferencePipeline(
            video_file=video_file,
            device=utils.get_default_device(),
            yolo_cfg=yolox_cfg,
        )

        person_dets, face_data = inference_pipeline.run(batch_size=16)
        inference_pipeline.save_pipeline_state()
        logger.info(f'Inference complete and state saved for {video_file}')
        return True

    except Exception as e:
        logger.exception(f'Error processing {video_file}')
        return False

    finally:
        io_utils.clear_memory()


def run_tracking_pipeline(video_file, inference_data=None):
    project_root = io_utils.get_project_root()
    output_dir = os.path.join(project_root, 'files/output/')

    file_prefix = video_file.split('.')[0]
    time_prefix, _ = utils.decode_vid_filename(video_file)

    if not inference_data:
        filename = io_utils.get_latest_file(
            output_dir, f'{file_prefix}_inference_pipeline.pkl'
        )
        data_path = os.path.join(output_dir, filename)

        with open(data_path, 'rb') as f:
            inference_data = pickle.load(f)
        
    credentials = conn_utils.get_aws_credentials()

    from pipelines import TrackingPipeline

    trk_pipeline = TrackingPipeline(
        video_file,
        time_prefix,
        inference_data['object_detections'],
        inference_data['face_detections'],
        credentials,
        continuous_mode=False,
    )

    trk_pipeline.run(filter_tracks=False)
    trk_pipeline.generate_output_vid()

    print(f'Processed {video_file}')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, default='inference')
    parser.add_argument('--video', type=str)
    
    args = parser.parse_args()

    pipeline = args.pipeline
    video = args.video

    if pipeline == 'inference':
        run_inference_pipeline(video)
    elif pipeline == 'tracking':
        run_tracking_pipeline(video)
    elif pipeline == 'both':
        run_inference_pipeline(video)
        run_tracking_pipeline(video)
