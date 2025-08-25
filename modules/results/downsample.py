# standard dependencies
import os
from fractions import Fraction
from typing import Optional
from datetime import datetime
import argparse

# 3rd-party dependencies
import av
import cv2
import numpy as np

# internal dependencies
from utilities import io_utils


def downsample_video(
    input_path: str,
    output_path: str,
    sample_every_s: float = 15.0,
    still_duration_s: float = 2.0,
    output_fps: int = 1,
    max_dim: int = 1920,
    crf: int = 23,                   # quality: lower = better (H.264 CRF)
    preset: str = 'veryfast',        # encoder speed/efficiency
):
    in_container = av.open(input_path)
    vstream = next(s for s in in_container.streams if s.type == 'video')

    fallback_rate = None
    if vstream.average_rate is not None:
        fallback_rate = float(vstream.average_rate)
    elif vstream.guessed_rate is not None:
        fallback_rate = float(vstream.guessed_rate)

    sampled_images = []
    next_sample_t = 0.0
    frame_idx = 0

    def _infer_frame_time(frame, stream, idx, fallback_rate=None):
        if frame.pts is not None and stream.time_base is not None:
            return float(frame.pts * stream.time_base)
        if fallback_rate:
            return idx / float(fallback_rate)
        return None

    def _resize_keep_aspect(img_bgr, target_width=None, target_height=None, max_dim=None):
        h, w = img_bgr.shape[:2]
        if max_dim:
            scale = max_dim / max(h, w)
            if scale >= 1.0:
                return img_bgr
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
        elif target_width:
            if w <= target_width:
                return img_bgr
            scale = target_width / float(w)
            new_w, new_h = target_width, int(round(h * scale))
        elif target_height:
            if h <= target_height:
                return img_bgr
            scale = target_height / float(h)
            new_w, new_h = int(round(w * scale)), target_height
        else:
            return img_bgr

        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    for frame in in_container.decode(vstream):
        t = _infer_frame_time(frame, vstream, frame_idx, fallback_rate=fallback_rate)
        frame_idx += 1
        if t is None:
            continue

        if t + 1e-6 >= next_sample_t:
            img = frame.to_ndarray(format='bgr24')
            if max_dim is not None:
                img = _resize_keep_aspect(img, max_dim=max_dim)
            sampled_images.append(img)
            next_sample_t += sample_every_s

    in_container.close()

    if not sampled_images:
        raise RuntimeError('No frames sampled; check input timestamps or parameters.')

    out_container = av.open(output_path, mode='w')
    out_stream = out_container.add_stream('h264', rate=output_fps)

    h, w = sampled_images[0].shape[:2]
    out_stream.width = w
    out_stream.height = h
    out_stream.pix_fmt = 'yuv420p'

    try:
        out_stream.codec_context.options = {
            'crf': str(crf),
            'preset': preset,
        }
    except Exception:
        pass

    still_frames = max(1, int(round(still_duration_s * output_fps)))
    time_base = Fraction(1, output_fps)
    pts = 0

    for img_bgr in sampled_images:
        for _ in range(still_frames):
            frm = av.VideoFrame.from_ndarray(img_bgr, format='bgr24')
            frm = frm.reformat(width=w, height=h, format='yuv420p')
            frm.pts = pts
            frm.time_base = time_base
            for packet in out_stream.encode(frm):
                out_container.mux(packet)
            pts += 1

    for packet in out_stream.encode():
        out_container.mux(packet)

    out_container.close()


def bulk_downsample(
    start_time: Optional[datetime | list] = None,
    end_time: Optional[datetime | list] = None,
    cam_ids: Optional[int | list[int]] = None,
    sample_every_s: float = 15.0,
    still_duration_s: float = 2.0,
    output_fps: int = 1,
    max_dim: int = 1920,
    crf: int = 23,
    preset: str = 'veryfast',
) -> list[str]:
    project_root = io_utils.get_project_root()
    input_dir = os.path.join(project_root, 'files/input/')
    output_dir = os.path.join(project_root, 'files/output/videos/')

    footage = io_utils.find_relevant_footage(start_time, end_time, cam_ids)

    for filename in footage:
        downsample_video(
            input_path       = os.path.join(input_dir, filename),
            output_path      = os.path.join(output_dir, f'ds_{filename}'),
            sample_every_s   = sample_every_s,
            still_duration_s = still_duration_s,
            output_fps       = output_fps,
            max_dim          = max_dim,
            crf              = crf,
            preset           = preset,
        )


if __name__ == '__main__':
    bulk_downsample()
