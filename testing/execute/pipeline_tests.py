# standard dependencies
import os
import argparse
import pickle

# 3rd-party dependencies
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# internal dependencies
from utilities import io_utils, log_utils, conn_utils
import utilities.general_utils as utils

logger = log_utils.get_logger(__name__)


def run_inference_pipeline(
        video_file: str,
        model_info,
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

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
        except RuntimeError as e:
            logger.exception(f'Error configuring TensorFlow GPU memory: {e}')

    from pipelines import InferencePipeline

    try:
        inference_pipeline = InferencePipeline(
            video_file=video_file,
            model_info=model_info,
            device=utils.get_default_device()
        )

        inference_pipeline.run()
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
    time_prefix = utils.parse_clip_filename(video_file, data='time')

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
    trk_pipeline.save_runtime_data()

    trk_pipeline.generate_output_vid()

    print(f'Processed {video_file}')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str)
    parser.add_argument('--video', type=str)

    parser.add_argument('--yolov4', type=str)
    parser.add_argument('--osnet', type=str)
    
    args = parser.parse_args()

    pipeline = args.pipeline
    video = args.video

    yolov4_weights = args.yolov4 or 'YOLOv4.pth'
    osnet_weights = args.osnet or 'OSNet.pth.tar-250'

    model_info = [
        yolov4_weights, osnet_weights,
        ('Facenet512', 'centerface_gpu')
    ]

    if pipeline == 'inference':
        run_inference_pipeline(video, model_info)
    elif pipeline == 'tracking':
        run_tracking_pipeline(video)
    elif pipeline == 'both':
        run_inference_pipeline(video, model_info)
        run_tracking_pipeline(video)
