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
import seaborn as sns


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
    durations = []
    identities = []

    for fname in os.listdir(pkl_dir):
        if fname.endswith('.pkl') and not fname.endswith('inference_data.pkl'):
            path = os.path.join(pkl_dir, fname)
            try:
                with open(path, 'rb') as f:
                    pipeline = pickle.load(f)

                fps = pipeline.fps
                all_tracks = pipeline.all_trks

                for trk in all_tracks.values():
                    if hasattr(trk, 'span') and isinstance(trk.span, list):
                        start, end = trk.span
                        length_sec = max(0, end - start) / fps
                        durations.append(length_sec)
                        identities.append(bool(getattr(trk, 'identity', None)))

            except Exception as e:
                print(f'Failed to process {fname}: {e}')

    if not durations:
        return 'No valid track durations found'

    df = pd.DataFrame({
        'duration_sec': durations,
        'has_identity': identities
    })

    stats = {
        'average_duration_sec': np.mean(durations),
        'median_duration_sec': np.median(durations),
        'min_duration_sec': np.min(durations),
        'max_duration_sec': np.max(durations),
        'with_identity': df['has_identity'].sum(),
        'without_identity': len(df) - df['has_identity'].sum(),
        'total_tracks': len(df)
    }

    plt.figure()
    plt.hist(durations, bins=30, edgecolor='black')
    plt.yscale('log')
    plt.title('Track Duration Distribution (Log Scale)')
    plt.xlabel('Track Duration (seconds, log scale)')
    plt.ylabel('Frequency')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    hist_path = os.path.join(pkl_dir, 'track_duration_histogram.png')
    plt.savefig(hist_path)
    plt.close()

    plt.figure()
    sns.boxplot(x='has_identity', y='duration_sec', data=df)
    plt.yscale('log')
    plt.title('Track Duration by Identity Assignment')
    plt.xlabel('Has Identity')
    plt.ylabel('Track Duration (seconds)')
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    box_path = os.path.join(pkl_dir, 'duration_vs_identity_boxplot.png')
    plt.savefig(box_path)
    plt.close()

    df['duration_bin'] = pd.cut(
        df['duration_sec'],
        bins=[0, 5, 10, 30, 60, 120, 300, 600],
        include_lowest=True,
        right=False
    )
    bin_summary = df.groupby('duration_bin')['has_identity'].mean().reset_index()

    plt.figure()
    sns.barplot(x='duration_bin', y='has_identity', data=bin_summary)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title('Identity Assignment Rate by Track Duration Bin')
    plt.ylabel('Fraction with Identity')
    plt.xlabel('Track Duration Bin (seconds)')
    plt.tight_layout()
    bar_path = os.path.join(pkl_dir, 'identity_assignment_by_duration_bin.png')
    plt.savefig(bar_path)
    plt.close()

    stats['binned_identity_rate_path'] = bar_path

    return stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track-lengths', action='store_true')
    args = parser.parse_args()
    analyze_track_lengths = args.track_lengths

    download_tracking_pkls()

    if analyze_track_lengths:
        length_stats = analyze_lengths()
        print(length_stats)
