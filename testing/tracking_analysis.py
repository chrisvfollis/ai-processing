# standard dependencies
import pickle
import os
import re
import argparse


# 3rd-party dependencies
import numpy as np
import pandas as pd
import boto3
import matplotlib.pyplot as plt

# internal dependencies
from utilities import io_utils


def download_tracking_pkls(
        shop_id: str = None,
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
    # pattern match example: 2025-04-18_09-30-00_3.pkl

    prefix = f'{shop_id}/' if shop_id else ''

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in page_iterator:
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)

            if pattern.fullmatch(filename):
                local_path = os.path.join(local_dir, filename)
                if os.path.exists(local_path):
                    continue
                print(f'Downloading {key} to {local_path}')
                s3.download_file(bucket_name, key, local_path)


def analyze_lengths(pkl_dir='../files/output/'):
    track_lengths = []
    all_fps = []

    for fname in os.listdir(pkl_dir):
        if fname.endswith('.pkl') and (not fname.endswith('inference_data.pkl')):
            path = os.path.join(pkl_dir, fname)
            try:
                with open(path, 'rb') as f:
                    pipeline = pickle.load(f)

                all_tracks = pipeline.all_trks
                fps = pipeline.fps

                for trk in all_tracks.values():
                    if hasattr(trk, 'span') and isinstance(trk.span, list):
                        start, end = trk.span
                        length = max(0, end - start)
                        track_lengths.append(length)
                        all_fps.append(fps)

            except Exception as e:
                print(f'Failed to process {fname}: {e}')

    if not track_lengths:
        return 'No valid track lengths found'

    stats = {
        'average_length': np.mean(track_lengths),
        'median_length': np.median(track_lengths),
        'min_length': np.min(track_lengths),
        'max_length': np.max(track_lengths),
    }

    # Plot histogram
    plt.hist(track_lengths, bins=30, edgecolor='black')
    plt.title('Track Length Distribution')
    plt.xlabel('Track Length (frames)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(pkl_dir, 'track_length_histogram.png')
    plt.savefig(plot_path)
    plt.close()

    stats['histogram_path'] = plot_path

    return pd.DataFrame([stats])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track-lengths', action='store_true')
    args = parser.parse_args()
    analyze_track_lengths = args.track_lengths

    shop_id = io_utils.get_shop()
    download_tracking_pkls(shop_id=shop_id)

    if analyze_track_lengths:
        analyze_lengths()
