import torch
import multiprocessing
import time
import os
from dotenv import load_dotenv
import signal
import sys
from utilities import io_utils

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def handle_sigterm(signum, frame):
    print("Received SIGTERM. Cleaning up...")

    io_utils.clear_track_info('all')
    io_utils.delete_local_files('all')

    sys.exit(0)


def process_inf_data(row, device, inference_output, time_prefix):
    video_file = row[0]
    camera = video_file.split('.')[0].split('_')[-1]

    from pipelines.tracking import TrackingPipeline

    trk_pipeline = TrackingPipeline(
        video_file, time_prefix, *inference_output, device
    )

    trk_pipeline.run()

    all_trks = trk_pipeline.all_trks
    fps = trk_pipeline.fps

    io_utils.save_track_info(
        time_prefix, camera, all_trks, fps=fps
    )

    trk_pipeline.generate_output_vid()

    print(f"Processed {video_file}")

    return True


def run_processing():
    def _prepare():
        

        load_dotenv()
        credentials = [os.environ.get('AWS_ACCESS_KEY'),
                       os.environ.get('AWS_SECRET_KEY')]

        io_utils.clear_track_info('all')
        io_utils.delete_local_files('all')

        return credentials, device

    signal.signal(signal.SIGTERM, handle_sigterm)
    multiprocessing.set_start_method("spawn")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    while True:
        qb_results = io_utils.get_queue_block()
        if qb_results:
            q_block, t_prefix, _ = qb_results
        else:
            time.sleep(60)
            continue

        tasks = [(row, device, t_prefix) for row in q_block]
        with multiprocessing.Pool(processes=3) as pool:
            pool.starmap(
                process_inf_output, tasks
            )

         io_utils.delete_local_files(t_prefix)


if __name__ == '__main__':
    run_processing()
