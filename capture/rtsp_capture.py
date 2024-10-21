import datetime
import multiprocessing
import subprocess
import shlex
from edge_logs import dispatch_logger
import time
import ffmpeg
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import input_output as io_utils


def get_config():
    with open('./config/cameras.txt', 'r') as camfile:
        addresses = [line.strip() for line in camfile]
    urls = [f'rtsp://admin:abcd1234@{address}:554/rtsp/streaming?channel=01&subtype=0'
            for address in addresses[1:]]
    
    with open('./config/location.txt', 'r') as locfile:
        location = locfile.read().strip()
    return urls, location


def capture_video(rtsp_url, duration, cam, location, origin):
    try:
        time = datetime.datetime.now()
        t_formatted = time.strftime("%Y-%m-%d_%H:%M:%S")
        file = f'{t_formatted}_{str(cam)}.mp4'
        output_path = f'footage/{file}'
        (
            ffmpeg
            .input(rtsp_url, rtsp_transport='tcp', t=duration)
            .output(output_path, vcodec='copy', format='mp4')
            .run()
        )

        io_utils.update_queue('../appdata/data.db', file, time=time, cam=f'c{cam}')

    except Exception as e:
        dispatch_logger('error', origin, f'{e}')
        return False
    return True


def multi_capture(duration, origin):
    try:
        urls, location = get_config()
        streams, i = [], 0
        for url in urls:
            streams.append({"url": url, "duration": duration, "cam": i,
                            "location": location, "origin": origin})
            i += 1
        
        num_processes = len(urls)

        pool = multiprocessing.Pool(processes=num_processes)
        pool.starmap(capture_video, ((stream["url"], stream["duration"],
                                    stream["cam"], stream["location"],
                                    stream["origin"]) for stream in
                                    streams))
        pool.close()
        pool.join()
    except Exception as e:
        dispatch_logger('error', origin, f'{e}')
        return False
    return True


def rtsp_cap():
    origin = 'rtsp_capture'
    closing_time = datetime.time(18)
    dispatch_logger('activity', origin, f'process started. closing time: {closing_time}')
    now = datetime.datetime.now().time()

    while now < closing_time:
        print('New loop')
        try:
            if (datetime.datetime.now().time().minute % 5) == 0:
                multi_capture(300, origin)
            else:
                now = datetime.datetime.now()
                next_5_min = (((now.minute//5)+1)*5)
                if next_5_min >= 60:
                    record_until = datetime.datetime(
                        now.year, now.month, now.day,
                        now.hour + 1, 0, 0)
                else:
                    record_until = datetime.datetime(
                        now.year, now.month, now.day, 
                        now.hour, next_5_min, 0)
                delta = (record_until - now).total_seconds() + 1
                multi_capture(delta, origin)
        except Exception as e:
            dispatch_logger('error', origin, f'{e}')
            time.sleep(60)
        now = datetime.datetime.now().time()
    
    dispatch_logger('activity', origin, 'past closing time.')
