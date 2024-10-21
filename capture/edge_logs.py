import requests
import json
from datetime import datetime
import sys
from dotenv import load_dotenv
import os
import subprocess


def log_to_server(api_key, category, origin, content):
    url = 'https://ivaktvision-fe27c015e5ff.herokuapp.com/api/edgedevice/logs/'

    data = {
        'datetime': str(datetime.now()),
        'category': category,
        'origin': origin,
        'content': content
        }

    json_data = json.dumps(data)

    headers = {
        'Content-Type': 'application/json',
        'X-Custom-API-Key': api_key
    }

    response = requests.post(url, data=json_data, headers=headers)

    if response.status_code != 200:
        server_response = str(response.json)
        file_path = f"./error_logs/{data['datetime']}_{data['origin']}_error.txt"
        with open(file_path, 'w') as file:
            for key in data.keys():
                file.write(data[key])
            file.write(f'\n{server_response}')

    return True


def dispatch_logger(category, origin, content):
    python = sys.executable
    edge_logs = './edge_logs.py'
    process = subprocess.Popen(
        [python, edge_logs, category, origin, content],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    return process.poll()


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get("EDGE_API_KEY")
    category = (sys.argv[1] if sys.argv[1] is not None else '')
    origin = (sys.argv[2] if sys.argv[2] is not None else '')
    content = (sys.argv[3] if sys.argv[3] is not None else '')

    log_to_server(api_key, category, origin, content)
