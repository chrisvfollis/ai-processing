# standard dependencies
import argparse
import os
import time
import math

# 3rd-party dependencies
import numpy as np
import cv2
import torch

# internal dependencies
import utilities.general_utils as utils
from models import YoloX
from trackers import OCSort
from trackers.oc_sort.args import make_parser


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, ids2=None):
    img = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    font = cv2.FONT_HERSHEY_PLAIN
    vert_offset = int(15 * text_scale)

    cv2.putText(
        img,
        f'frame: {frame_id} num: {len(tlwhs)}',
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
            img,
            xyxy_box[0:2],
            xyxy_box[2:4],
            color=box_color,
            thickness=line_thickness
        )

        id_text = f'{int(obj_id)}'
        if ids2 is not None:
            id_text = id_text + f', {int(ids2[i])}'

        cv2.putText(
            img,
            id_text,
            (xyxy_box[0], xyxy_box[1]),
            font,
            text_scale,
            (0, 0, 255),
            text_thickness
        )
    
    return img


def run_demo(predictor, tracker, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    save_folder = os.path.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = args.out_path
    print(f"video save_path is {save_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_dims = (int(width), int(height))

    vid_writer = cv2.VideoWriter(save_path, fourcc, fps, vid_dims)

    f_num = 0
    while True:
        if f_num % 20 == 0:
            print(f'Processing frame {f_num}')

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
                frame, online_boxes, online_ids, frame_id=f_num
            )

        vid_writer.write(frame)
        f_num += 1


def main(args):
    output_dir = './YOLOX_outputs'
    vis_folder = os.path.join(output_dir, "track_vis")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = YoloX(
        args.ckpt,
        num_classes=1,
        depth=1.33,
        width=1.25,
        input_size=(800, 1440),
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        device=device,
        fp16=args.fp16,
    )
    if args.fuse:
        print("\tFusing model...")
        predictor.fuse()
    
    tracker = OCSort(
        det_thresh=args.track_thresh,
        iou_threshold=args.iou_thresh,
        use_byte=args.use_byte,
    )

    current_time = time.localtime()

    run_demo(predictor, tracker, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    main(args)
