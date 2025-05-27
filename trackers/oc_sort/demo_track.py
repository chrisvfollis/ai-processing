# standard dependencies
import argparse
import os
import math

# 3rd-party dependencies
import numpy as np
import cv2
import torch

# internal dependencies
import utilities.general_utils as utils
from utilities import io_utils
from models import YoloX
from trackers import OCSort


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, f_num, ids2=None):
    image = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    font = cv2.FONT_HERSHEY_PLAIN
    vert_offset = int(15 * text_scale)

    cv2.putText(
        image,
        f'frame: {f_num} num: {len(tlwhs)}',
        (0, vert_offset),
        font,
        text_scale,
        (0, 0, 255),
        text_thickness
    )

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        xyxy_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))

        obj_id = int(obj_ids[i])
        box_color = get_color(abs(obj_id))
        cv2.rectangle(
            image,
            xyxy_box[0:2],
            xyxy_box[2:4],
            color=box_color,
            thickness=line_thickness
        )
        id_text = f'{int(obj_id)}'
        if ids2 is not None:
            id_text = id_text + f', {int(ids2[i])}'
        cv2.putText(
            image,
            id_text,
            (xyxy_box[0], xyxy_box[1]),
            font,
            text_scale,
            (0, 0, 255),
            text_thickness
        )

    return image


def run_demo(predictor, tracker, args):
    # paths:
    project_root = io_utils.get_project_root()

    input_dir = os.path.join(project_root, 'files/input/')
    input_vid_path = os.path.join(input_dir, args.input_video)

    output_vid_dir = os.path.join(project_root, 'files/output/', 'videos/')
    output_vid_path = io_utils.get_unique_path(
        output_vid_dir, f'{args.input_video.split('.')[0]}_boxes.mp4'
    )

    # input video:
    cap = cv2.VideoCapture(input_vid_path)
    video_info = utils.get_video_info(cap, release=False)
    width, height = video_info[0]
    fps = video_info[2]
    tot_frames = video_info[3]

    # output video:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_dims = (int(width), int(height))
    out = cv2.VideoWriter(output_vid_path, fourcc, fps, vid_dims)

    progress_interval = tot_frames // 4
    f_num = 0

    while True:
        if f_num % progress_interval == 0:
            progress = int(round((f_num / tot_frames) * 100), 0)
            print(f'Percent complete: {progress}%')

        ret, frame = cap.read()
        if not ret:
            break
    
        batched_output = predictor.inference(frame)
        output = batched_output[0]

        if output is not None:
            img_hw = frame.shape[:2]

            online_targets = tracker.update(output, img_hw, predictor.input_size)
    
            online_boxes = []
            online_ids = []
            for t in online_targets:
                trk_id, box = t[4], utils.xywh_xyxy(t[:4], out='xywh')

                valid_ratio = (box[2] / box[3]) <= args.aspect_ratio_thresh
                valid_area = math.prod(box[2:4]) > args.min_box_area

                if not (valid_ratio and valid_area):
                    continue
    
                online_boxes.append(box)
                online_ids.append(trk_id)

            frame = plot_tracking(
                frame, online_boxes, online_ids, f_num
            )

        out.write(frame)
        f_num += 1

    out.release()
    cap.release()


def main(args):
    predictor_config = {
        'checkpoint': args.ckpt,
        'num_classes': 1,
        'depth': 1.33,
        'width': 1.25,
        'input_size': (800, 1440),
        'conf_thresh': args.conf,
        'nms_thresh': args.nms,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'fp16': args.fp16,
    }
    predictor = YoloX(**predictor_config)

    if args.fuse:
        print("\tFusing model...")
        predictor.fuse()

    tracker_config = {
        'det_thresh': args.det_thresh,
        'max_age': args.max_age,
        'min_hits': args.min_hits,
        'iou_threshold': args.iou_thresh,
        'delta_t': args.dt,
        'asso_func': args.asso,
        'inertia': args.inertia,
        'use_byte': args.use_byte,
    }
    tracker = OCSort(**tracker_config)

    run_demo(predictor, tracker, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("OC-SORT parameters")

    # demo video args:
    parser.add_argument("--input-video", type=str)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument('--min-box-area', type=float, default=100,
                        help='filter out tiny boxes')

    # model args:
    parser.add_argument("-c", "--checkpoint", type=str,
                        default='ocsort_x_mot17.pth.tar')
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--fuse", default=False, action="store_true")
    parser.add_argument("--conf", default=0.05, type=float)
    parser.add_argument("--nms", default=0.7, type=float)

    # tracking args:
    parser.add_argument("--det-thresh", type=float, default=0.6)
    parser.add_argument("--max-age", type=int, default=30,
                        help="num frames to keep lost tracks")
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--dt", "--delta-t", type=int, default=3)
    parser.add_argument('--asso', default="iou")
    parser.add_argument("--inertia", type=float, default=0.2)
    parser.add_argument("--use-byte", default=False, action="store_true")
    
    args = parser.parse_args()

    main(args)
