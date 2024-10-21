from datetime import datetime, timedelta
import sys
import os
import boto3
from dotenv import load_dotenv
from boto3.s3.transfer import TransferConfig
from edge_logs import dispatch_logger


def upload_to_s3(origin, client, file_name, bucket, object_name, config=None):
    try:
        client.upload_file(file_name, bucket, object_name, Config=config)
        os.remove(file_name)
    except Exception as e:
        dispatch_logger('error', origin, f'error uploading {object_name}: {e}')


if __name__ == '__main__':
    load_dotenv()
    MB = 1024**2
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    origin = 's3_uploader'
    dispatch_logger('notice', origin, 's3_uploader started')
    try:
        duration = float(sys.argv[1]) - 1200
        stop_time = datetime.now() + timedelta(duration)
        config = TransferConfig()
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name='us-west-1')
        files = sorted(os.listdir('./footage'))
    except Exception as e:
        dispatch_logger('error', origin, f'error: {e}')

    if len(files) == 0:
        dispatch_logger('notice', origin, 'footage directory is empty')
    for file in files:
        if datetime.now() < stop_time:
            upload_to_s3(origin, s3, 'footage/' + file,
                        'unprocessed-ivakt-footage',
                        file, config)
        else:
            break
