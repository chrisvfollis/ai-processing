from datetime import datetime
import multiprocessing
import subprocess
import shlex
import time
import ffmpeg
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'processing'))
import io_utils
import sqlite3


def update_camera_info():
    def _scan_network():
        while True:
            try:
                process = subprocess.Popen(
                    '/home/ivaktvision/Documents/timemanager/capture/utilities/network_scanner.sh',
                    shell=True,
                    cwd='/home/ivaktvision/Documents/timemanager/capture/utilities'
                )
                process.wait()
                with open('./config/cameras.txt', 'r') as camfile:
                    addresses = [line.strip() for line in camfile]
                    addresses = addresses[1:]
                return addresses
            except Exception as e:
                print('Network scan failed')
                print(e)
                time.sleep(30)

    def _get_current_info(cursor):
        try:
            cursor.execute('''
                SELECT * FROM cameras
            ''')
            results = cursor.fetchall()
            if results and len(results) > 0:
                return results
            else:
                return None
        except sqlite3.OperationalError:
            return None

    def _compare_info(prior_data, new_addresses):
        old_addresses = [row[2] for row in prior_data]

        matched = [address for address in new_addresses if address in
                   old_addresses]
    
    def _add_new_cameras(conn, cursor, addresses):
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera TEXT UNIQUE,
                designation TEXT,
                ip_address TEXT,
                reference_img TEXT
            )
        ''')
        conn.commit()

        for i, address in enumerate(addresses):
            camera = f'c{i}'
            cursor.execute('''
                INSERT INTO cameras (camera, ip_address)
                VALUES (?, ?)
                ON CONFLICT(camera) DO UPDATE SET ip_address = excluded.ip_address
            ''', (camera, address))
        conn.commit()

    conn = sqlite3.connect('../appdata/data.db')
    cursor = conn.cursor()

    new_ip_addresses = _scan_network()
    print(new_ip_addresses)
    # if not _get_current_info(cursor):
    #     _add_new_cameras(conn, cursor, new_ip_addresses)
    _add_new_cameras(conn, cursor, new_ip_addresses)


def get_stream_info(fps={'primary': 30, 'secondary': 15}, db_path='../appdata/data.db'):
    def _create_rtsp_url(ip_address, format=0, default_creds=0, credentials=None):
        default_credentials = [('admin', 'abcd1234'), ('admin', 'admin')]
        if credentials:
            user, passwd = credentials
        else:
            user, passwd = default_credentials[default_creds]
        formats = [f'rtsp://{user}:{passwd}@{ip_address}:554/rtsp/streaming?channel=01&subtype=0',
                f'rtsp://{user}:{passwd}@{ip_address}:554/ch01/0']
        return formats[format]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM cameras')

        results = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError:
        conn.close()
        return None
    
    stream_info = {}
    for result in results:
        camera = result[1].strip('c')
        try:
            frame_rate = fps[result[2]]
        except KeyError:
            frame_rate = 30
        url = _create_rtsp_url(result[3])

        stream_info[camera] = {'url': url, 'frame_rate': frame_rate}

    return stream_info


def rtsp_capture(rtsp_url, duration, cam, frame_rate):
    try:
        time = datetime.now()
        t_formatted = time.strftime("%Y-%m-%d_%H-%M-%S")
        file = f'{t_formatted}_{str(cam)}.mp4'
        output_path = f'../input_files/{file}'
        (
            ffmpeg
            .input(rtsp_url, rtsp_transport='tcp', t=duration)
            .output(output_path, vcodec='copy', format='mp4', r=frame_rate)
            .run()
        )

        io_utils.update_queue(action='add', video_file=file, datetime=time, cam=f'c{cam}')

    except Exception as e:
        print(e)
        return False
    return True


def run_capture_cycle(stream_info, interval=5, min_seconds=3):
    def _multi_capture(stream_info, duration):
        try:
            streams = []
            for cam, data in stream_info.items():
                url = data['url']
                frame_rate = data['frame_rate']
                streams.append({"url": url, "duration": duration, "cam": cam,
                                "frame_rate": frame_rate})
            
            num_processes = len(stream_info.keys())

            pool = multiprocessing.Pool(processes=num_processes)
            pool.starmap(rtsp_capture, ((stream["url"], stream["duration"],
                                        stream["cam"], stream["frame_rate"])
                                        for stream in streams))
            pool.close()
            pool.join()
        except Exception as e:
            print(e)
            return False
        return True
    
    duration = interval * 60
    while True:
        now = datetime.now()
        if now.minute % interval == 0:
            _multi_capture(stream_info, duration)
        else:
            next_interval = (((now.minute//interval)+1)*interval)
            if next_interval >= 60:
                record_until = datetime(now.year, now.month, now.day,
                                        now.hour + 1, 0, 0)
            else:
                record_until = datetime(now.year, now.month, now.day, 
                                        now.hour, next_interval, 0)
            now = datetime.now()
            delta = (record_until - now).total_seconds() + 1
            if delta > min_seconds:
                _multi_capture(stream_info, delta)


if __name__ == '__main__':
    while True:
        update_camera_info()
        stream_info = get_stream_info()
        if not stream_info:
            print('No stream info')
            time.sleep(30)
            continue
        else:
            run_capture_cycle(stream_info, interval=5)
