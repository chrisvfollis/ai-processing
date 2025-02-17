import torch
import multiprocessing
import os
from dotenv import load_dotenv
from utilities import io_utils
from utilities import utilities as utils
import pickle
import time
import threading

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def process_inf_output(video_file, device):
    file_prefix = video_file.split('.')[0]
    data_path = os.path.join('../files/output', f'{file_prefix}_inference_data.pkl')
    with open(data_path, 'rb') as f:
        inference_output = pickle.load(f)

    t_prefix = utils.parse_clip_filename(video_file, data='time')

    from pipelines.tracking import TrackingPipeline

    trk_pipeline = TrackingPipeline(
        video_file, t_prefix, *inference_output, device, continuity=False
    )

    trk_pipeline.run()

    all_trks = trk_pipeline.all_trks

    trk_pipeline.generate_output_vid()

    print(f"Processed {video_file}")

    return True


def run_processing():
    multiprocessing.set_start_method("spawn")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = os.listdir('../files/input')
    vid_files = sorted([f for f in results if f.endswith('.mp4')])

    tasks = [(vid_file, device) for vid_file in vid_files]

    start_time = time.time()
    stop_event = threading.Event()

    time_logging = threading.Thread(
        target=utils.log_elapsed_time, args=(start_time, stop_event), daemon=True
    )
    time_logging.start()

    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(
            process_inf_output, tasks
        )
    
    stop_event.set()


if __name__ == '__main__':
    run_processing()
