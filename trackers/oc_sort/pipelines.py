# standard dependencies
pass

# 3rd-party dependencies
import numpy as np
import cv2

# internal dependencies
pass


def inference_pipeline(predictor, video_path, output_detections_path,
                       batch_size: int = 1):
    cap = cv2.VideoCapture(video_path)

    detections = {}

    f_start_num = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        
        if len(frames) == batch_size:
            batched_output = predictor.inference(frames)
            
            for i, output in enumerate(batched_output):
                frame_idx = f_start_num + i
                if output is not None:
                    detections[f'frame_{frame_idx}'] = output.cpu().numpy()
                else:
                    detections[f'frame_{frame_idx}'] = np.empty((0, 5))

            frames = []
            f_start_num += batch_size

    if frames:
        batched_output = predictor.inference(frames)
        for i, output in enumerate(batched_output):
            frame_idx = f_start_num + i
            if output is not None:
                detections[f'frame_{frame_idx}'] = output.cpu().numpy()
            else:
                detections[f'frame_{frame_idx}'] = np.empty((0, 5))

    cap.release()
    np.savez_compressed(output_detections_path, **detections)


def run_tracking_only(tracker, detections_path, img_hw, input_size):
    data = np.load(detections_path)
    tracker_results = []

    for frame_idx in sorted(
        data.files, key=lambda x: int(x.replace('frame_', ''))
    ):
        dets = data[frame_idx]
        online_targets = tracker.update(dets, img_hw, input_size)

        tracker_results.append((frame_idx, online_targets))

    return tracker_results
