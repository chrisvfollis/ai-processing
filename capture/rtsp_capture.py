from datetime import datetime
import multiprocessing
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
import ffmpeg
import sqlite3
import os
from dotenv import load_dotenv
import requests
import boto3
import signal


def worker_initializer():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def handle_sigterm(signum, frame):
    print(f'Received SIGTERM in process {os.getpid()}. Initiating shutdown...')
    global shutdown_flag
    shutdown_flag = True


shutdown_flag = False
signal.signal(signal.SIGTERM, handle_sigterm)

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


def get_stream_info(db_path='../appdata/data.db'):
    def _create_rtsp_url(ip_address, format=2, credentials=2):
        
        default_credentials = [
            ('admin', 'abcd1234'),
            ('admin', 'admin'),
            ('admin', '123456')
        ]

        if isinstance(credentials, int):
            user, passwd = default_credentials[credentials]
        else:
            user, passwd = credentials
        
        formats = [
            f'rtsp://{user}:{passwd}@{ip_address}:554/rtsp/streaming?channel=01&subtype=0',
            f'rtsp://{user}:{passwd}@{ip_address}:554/ch01/0',
            f'rtsp://{user}:{passwd}@{ip_address}:554/Streaming/Channels/101'
        ]

        if isinstance(format, int):
            return formats[format]
        else:
            return format


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
        url = _create_rtsp_url(result[3])

        stream_info[camera] = {'url': url}

    return stream_info


def upload_and_post(cap_info):
    def _upload_to_s3(file_path, s3_key, credentials,
                      bucket_name='ivakt-footage'):
        s3 = boto3.client(
            's3',
            aws_access_key_id=credentials[0],
            aws_secret_access_key=credentials[1],
            region_name='us-west-1'
        )
        try:
            s3.upload_file(file_path, bucket_name, s3_key)
            print(f'Uploaded {s3_key}')
            return True
        except Exception as e:
            print(f'Upload failed for {file_path}: {s3_key}')
            return False
        
    load_dotenv()
    credentials = [os.environ.get('AWS_ACCESS_KEY'),
                   os.environ.get('AWS_SECRET_KEY')]
    INTERNAL_API_KEY = os.environ.get('INTERNAL_API_KEY')
    print(f'Internal API key: {INTERNAL_API_KEY}')
    url = ('https://ivaktvision-fe27c015e5ff.herokuapp.com/'
           + 'api/service/update_queue/')

    data = {'action': 'add', 'filenames': [], 'timestamps': [], 'cameras': []}
    for row in cap_info:
        file_path = f'../input_files/{row[0]}'
        s3_key = row[0]

        if _upload_to_s3(file_path, s3_key, credentials):
            data['filenames'].append(row[0])
            data['timestamps'].append(row[1].isoformat())
            data['cameras'].append(row[2])

        if os.path.exists(file_path):
            os.remove(file_path)

    headers = {
        'X-Custom-Api-Key': INTERNAL_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print('Success')
        else:
            print('Error:')
            print(response.text)
            print(response.status_code)
    except Exception as e:
        print(f'Request failed: {e}')


def rtsp_capture(rtsp_url, duration, cam):
    try:
        time = datetime.now()
        t_formatted = time.strftime("%Y-%m-%d_%H-%M-%S")
        timestamp = datetime.strptime(t_formatted, "%Y-%m-%d_%H-%M-%S")
        file = f'{t_formatted}_{str(cam)}.mp4'
        output_path = f'../input_files/{file}'
        process = (
            ffmpeg
            .input(rtsp_url, rtsp_transport='tcp', t=duration)
            .output(output_path, vcodec='copy', format='mp4', an=None)
            .run_async(pipe_stdin=True)
        )

        start_time = time.time()
        while process.poll() is None:
            if shutdown_flag:
                print(f'Terminating capture for camera {cam}')
                try:
                    process.stdin.write(b'q')
                    process.stdin.close()
                except Exception as e:
                    print(f'Error while stopping FFmpeg for camera {cam}: {e}')
                    break
            if time.time() - start_time >= duration:
                break
        process.wait()
    except Exception as e:
        print(f'Error: {e}')
        return None
    return (file, timestamp, f'c{cam}')


def run_capture_cycle(stream_info, interval=1, min_seconds=3):
    def _multi_capture(stream_info, duration):
        try:
            streams = [{'url': data['url'], 'duration': duration, 'cam': cam}
                       for cam, data in stream_info.items()]
            
            num_processes = len(stream_info.keys())

            pool = multiprocessing.Pool(processes=num_processes)
            cap_info = pool.starmap(rtsp_capture, (
                (stream["url"], stream["duration"], stream["cam"])
                for stream in streams
            ))
            pool.close()
            pool.join()

            return [row for row in cap_info if row is not None]

        except Exception as e:
            print(f'Error: {e}')
            return None
    
    upload_queue = ThreadPoolExecutor(max_workers=1)

    duration = interval * 60
    while True:
        now = datetime.now()
        if now.minute % interval == 0:
            cap_info = _multi_capture(stream_info, duration)
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
                cap_info = _multi_capture(stream_info, delta)
        
        if cap_info:
            upload_queue.submit(upload_and_post, cap_info)
        else:
            print("No valid captures to upload")


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