import torch
import multiprocessing
import time
import os
from dotenv import load_dotenv
import signal
import sys
from utilities import io_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def handle_sigterm(signum, frame):
    print("Received SIGTERM. Cleaning up...")

    io_utils.clear_track_info('all')
    io_utils.delete_local_files('all')

    sys.exit(0)


def process_video(row, credentials, model_info, device, time_prefix):
    video_file = row[0]
    camera = video_file.split('.')[0].split('_')[-1]

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


    inf_pipeline = InferencePipeline(video_file, model_info, device)
    if not inf_pipeline.skim():
        io_utils.delete_s3_footage(video_file, credentials)
        return False
    else:
        inference_output = inf_pipeline.run()

    trk_pipeline = TrackingPipeline(
        video_file, time_prefix, *inference_output, device
    )   

    trk_pipeline.run()
    trk_pipeline.assign_identities()

    all_trks = trk_pipeline.all_trks
    fps = inf_pipeline.fps

    io_utils.save_track_info(
        time_prefix, camera, all_trks, fps=fps
    )
    print(f"Processed {video_file}")

    return True


def run_processing_cycle():
    def _prepare():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        yolov4_weights = '../models/weights/YOLOv4.pth'
        osnet_weights = '../models/weights/OSNet.pth.tar-250'
        movenet_dir = '../models/weights/movenet'
        face_models = ['Facenet512', 'retinaface']

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
    multiprocessing.set_start_method("spawn")
    start_vars = _prepare()

    while True:
        qb_results = io_utils.get_queue_block()
        if qb_results:
            q_block, t_prefix = qb_results[:2]
        else:
            time.sleep(60)
            continue

        tasks = [(row, *start_vars, t_prefix) for row in q_block]
        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(
                process_video, tasks
            )

        _finalize(*qb_results, start_vars[0])


if __name__ == '__main__':
    run_processing_cycle()
