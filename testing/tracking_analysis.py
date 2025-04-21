# standard dependencies
import pickle
import os
import re


# 3rd-party dependencies
import pandas as pd
import boto3


# internal dependencies
from utilities import io_utils


def download_tracking_pkls(
        bucket_name='visionservice-data',
        local_dir='../files/output/'
    ):
    credentials = io_utils.get_aws_creds()

    s3 = boto3.client(
        's3',
        aws_access_key_id=credentials[0],
        aws_secret_access_key=credentials[1],
        region_name='us-west-1'
    )

    os.makedirs(local_dir, exist_ok=True)

    pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d+\.pkl')
    # example: 2025-04-18_09-30-00_3.pkl

    response = s3.list_objects_v2(Bucket=bucket_name)

    if 'Contents' not in response:
        print(f"No objects found in bucket {bucket_name}")
        return

    for obj in response['Contents']:
        key = obj['Key']
        if pattern.fullmatch(os.path.basename(key)):
            local_path = os.path.join(local_dir, os.path.basename(key))
            print(f"Downloading {key} to {local_path}")
            s3.download_file(bucket_name, key, local_path)

