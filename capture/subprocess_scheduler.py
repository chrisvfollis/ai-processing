import subprocess
import time
from datetime import datetime, timedelta
from rtsp_capture import rtsp_cap
import sys
from edge_logs import dispatch_logger


def run_subprocess(python, script):
    process = subprocess.Popen(
        [python, script, str(delta)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return True


if __name__ == '__main__':
    origin = 'subprocess_scheduler'
    s3_upl = './s3_uploader.py'
    rtsp_capture = './rtsp_capture.py'
    python = sys.executable
    dispatch_logger('activity', origin, 'daemon started')
    try:
        process = subprocess.Popen(
            '/home/ivaktvision/Documents/edgesoftware/utilities/network_scanner.sh',
            shell=True,
            cwd='/home/ivaktvision/Documents/edgesoftware/utilities'
        )
        process.wait()
    except Exception as e:
        dispatch_logger('error', origin, f'network scanner error: {e}')

    while True:
        now = datetime.now()
        opening_time = datetime(year=now.year, month=now.month,
                                day=now.day, hour=7)
        closing_time = datetime(year=now.year, month=now.month,
                                day=now.day, hour=18)

        if now.weekday() < 5:
            if now >= closing_time:
                if now.weekday() == 4:
                    opening_time = opening_time + timedelta(days=3)
                else:
                    opening_time = opening_time + timedelta(days=1)

            elif now >= opening_time:
                delta = (closing_time - now).total_seconds() + 1
                dispatch_logger('activity', origin, f'starting capture')
                rtsp_cap()
                continue

            delta = (opening_time - now).total_seconds() + 1
            dispatch_logger('activity', origin, f'starting uploads')
            run_subprocess(python, s3_upl)
            time.sleep(delta)

        else:
            days = 7 - opening_time.weekday()
            opening_time = opening_time + timedelta(days=days)
            delta = (opening_time - now).total_seconds() + 1
            dispatch_logger('activity', origin, 'starting uploads')
            run_subprocess(python, s3_upl)
            time.sleep(delta)
