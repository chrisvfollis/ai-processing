# standard dependencies
import os
from fractions import Fraction
from typing import Optional
from datetime import datetime
import argparse

# 3rd-party dependencies
import av

# internal dependencies
from utilities import utils, io_utils



def downsample_video(
    input_path: str,
    output_path: str,
    sample_interval: float = 20.0,  # seconds between samples
    frame_duration: float = 2.0,    # length in seconds of output frames
    max_dim: int = 1920,
    crf: int = 25,
    preset: str = 'veryfast',
):
    ic = av.open(input_path)
    vstream = next(s for s in ic.streams if s.type == 'video')
    try:
        vstream.thread_type = 'AUTO'
    except Exception:
        pass

    fps = (Fraction(1, 1) / Fraction(frame_duration).limit_denominator(1000))

    oc = av.open(output_path, mode='w')
    ostream = oc.add_stream('h264', rate=fps)
    ostream.pix_fmt = 'yuv420p'
    try:
        ostream.codec_context.options = {
            'crf': str(crf),
            'preset': preset,
            'tune': 'stillimage'
        }
    except Exception:
        pass

    tb = Fraction(fps.denominator, fps.numerator)
    pts = 0
    dims_set = False

    def fit(w, h, max_dim):
        if not max_dim:
            return w, h
        s = max_dim / max(w, h)
        return (w, h) if s >= 1.0 else (int(round(w * s)), int(round(h * s)))

    if vstream.time_base is None:
        target = 0.0
        t_per_frame = (
            float(1.0 / float(vstream.average_rate))
            if vstream.average_rate else None
        )
        for frame in ic.decode(vstream):
            if t_per_frame is None:
                break
            if target <= 0.0 + 1e-6:
                w_out, h_out = fit(frame.width, frame.height, max_dim)
                of = frame.reformat(width=w_out, height=h_out, format='yuv420p')
                if not dims_set:
                    ostream.width, ostream.height = w_out, h_out
                    dims_set = True
                of.pts, of.time_base = pts, tb
                for pkt in ostream.encode(of):
                    oc.mux(pkt)
                pts += 1
                target = sample_interval

        for pkt in ostream.encode():
            oc.mux(pkt)
        oc.close(); ic.close()
        return

    tb_in = float(vstream.time_base)
    t = 0.0
    consecutive_failures = 0
    max_failures = 3
    tol = max(0.5, 0.5) / max(1.0, float(vstream.average_rate or 20.0))

    while True:
        target_ts = int(t / tb_in)
        try:
            ic.seek(target_ts, any_frame=False, backward=True, stream=vstream)
        except av.AVError:
            break

        got = None
        decoded = 0
        cap = int(2 * (float(vstream.average_rate or 20.0)) + 10)
        for frame in ic.decode(vstream):
            if frame.pts is None:
                continue
            ftime = float(frame.pts * vstream.time_base)
            decoded += 1
            if ftime + 1e-6 >= t - tol:
                got = frame
                break
            if decoded >= cap:
                break

        if got is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                break
            t += sample_interval
            continue

        consecutive_failures = 0
        w_out, h_out = fit(got.width, got.height, max_dim)
        of = got.reformat(width=w_out, height=h_out, format='yuv420p')
        if not dims_set:
            ostream.width, ostream.height = w_out, h_out
            dims_set = True

        of.pts, of.time_base = pts, tb
        for pkt in ostream.encode(of):
            oc.mux(pkt)
        pts += 1

        t += sample_interval

    for pkt in ostream.encode():
        oc.mux(pkt)
    oc.close()
    ic.close()


def bulk_downsample(
    start_time: Optional[datetime | list] = None,
    end_time: Optional[datetime | list] = None,
    cam_ids: Optional[int | list[int]] = None,
    sample_interval: float = 20.0,
    frame_duration: float = 2.0,
    max_dim: int = 1920,
    crf: int = 25,
    preset: str = 'veryfast',
) -> list[str]:
    project_root = io_utils.get_project_root()

    input_dir = os.path.join(project_root, 'files/input/')
    output_dir = os.path.join(project_root, 'files/output/videos/')

    footage = io_utils.find_relevant_footage(start_time, end_time, cam_ids)

    for filename in footage:
        print(f'[{datetime.now()}] Downsampling {filename}...')
        downsample_video(
            input_path       = os.path.join(input_dir, filename),
            output_path      = os.path.join(output_dir, f'ds_{filename}'),
            sample_interval  = sample_interval,
            frame_duration   = frame_duration,
            max_dim          = max_dim,
            crf              = crf,
            preset           = preset,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start-time', type=str, default=None)
    parser.add_argument('--end-time', type=str, default=None)
    parser.add_argument('--cam-ids', type=str, default=None)

    args = parser.parse_args()

    bulk_downsample(
        start_time = utils.convert_to_datetime(args.start_time),
        end_time   = utils.convert_to_datetime(args.end_time),
        cam_ids    = list(map(int, args.cam_ids.split(','))) if args.cam_ids else None,
    )
