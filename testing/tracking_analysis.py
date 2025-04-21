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
import statsmodels.api as sm


# internal dependencies
from utilities import io_utils
import utilities.general_utils as utils


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

    X = np.log(df['duration_sec'].values + 1)  # add 1 to avoid log(0)
    X = sm.add_constant(X)  # adds intercept
    y = df['has_identity'].astype(int)

    model = sm.Logit(y, X).fit()
    print(model.summary())

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
    plt.title('Track Duration Distribution')
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

    bin_max = utils.logceil_round(np.max(durations))
    bins = [bin_max // i for i in range(10, 0, -1)]
    
    df['duration_bin'] = pd.cut(
        df['duration_sec'],
        bins=sorted(set([0]+ bins)),
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

    return stats


def analyze_bbox_areas(pkl_dir='../files/output/'):
    side_lengths = []
    has_identity = []

    for fname in os.listdir(pkl_dir):
        if fname.endswith('.pkl') and not fname.endswith('inference_data.pkl'):
            path = os.path.join(pkl_dir, fname)
            try:
                with open(path, 'rb') as f:
                    pipeline = pickle.load(f)

                for trk in pipeline.all_trks.values():
                    detections = trk.object_detections.values()
                    if not detections:
                        continue

                    areas = [d[2] * d[3] for d in detections if len(d) >= 4]
                    if not areas:
                        continue
                    avg_area = np.mean(areas)
                    avg_side_length = np.sqrt(avg_area)

                    side_lengths.append(avg_side_length)
                    has_identity.append(bool(getattr(trk, 'identity', None)))

            except Exception as e:
                print(f"Failed to process {fname}: {e}")

    if not side_lengths:
        return 'No valid bbox areas found'

    df = pd.DataFrame({
        'avg_bbox_side_length': side_lengths,
        'has_identity': has_identity
    })

    stats = {
        'mean_side_length': np.mean(side_lengths),
        'median_side_length': np.median(side_lengths),
        'min_side_length': np.min(side_lengths),
        'max_side_length': np.max(side_lengths),
        'with_identity': df['has_identity'].sum(),
        'without_identity': len(df) - df['has_identity'].sum(),
        'total_tracks': len(df)
    }

    plt.figure()
    plt.hist(side_lengths, bins=30, edgecolor='black')
    plt.title('Average BBox Side Length Distribution')
    plt.xlabel('(Average Area) — Side Length (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(pkl_dir, 'bbox_side_length_histogram.png'))
    plt.close()

    plt.figure()
    sns.boxplot(x='has_identity', y='avg_bbox_side_length', data=df)
    plt.yscale('log')
    plt.title('Avg BBox Side Length by Identity Assignment')
    plt.xlabel('Has Identity')
    plt.ylabel('(Avg Area) — Side Length (pixels)')
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(pkl_dir, 'bbox_side_length_vs_identity_boxplot.png'))
    plt.close()

    bin_edges = np.linspace(min(side_lengths), max(side_lengths), num=10)
    df['length_bin'] = pd.cut(df['avg_bbox_side_length'], bins=bin_edges)

    bin_proportions = (
        df.groupby('length_bin')['has_identity']
        .agg(['sum', 'count'])
        .rename(columns={'sum': 'with_id', 'count': 'total'})
    )
    bin_proportions['proportion'] = bin_proportions['with_id'] / bin_proportions['total']

    plt.figure()
    bin_centers = [interval.mid for interval in bin_proportions.index]
    plt.plot(bin_centers, bin_proportions['proportion'], marker='o')
    plt.ylim(0, 1.05)
    plt.title('Proportion of Tracks with Identity by Side Length Bin')
    plt.xlabel('Avg BBox Side Length (pixels)')
    plt.ylabel('Proportion with Identity')
    plt.grid(True, ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(pkl_dir, 'identity_proportion_by_side_length.png'))
    plt.close()

    X = np.array(side_lengths)
    y = np.array(has_identity, dtype=int)

    X_sm = sm.add_constant(X)
    model = sm.Logit(y, X_sm)
    result = model.fit(disp=False)

    side_len_range = np.linspace(min(X), max(X), 500)
    X_plot = sm.add_constant(side_len_range)
    probs = result.predict(X_plot)

    plt.figure()
    plt.plot(side_len_range, probs, label='Logit fit')
    plt.scatter(X, y, alpha=0.2, s=10, c='gray', label='Data (jittered)')
    plt.title('Logistic Regression: Identity vs Side Length')
    plt.xlabel('Avg BBox Side Length (pixels)')
    plt.ylabel('P(Identity Assigned)')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pkl_dir, 'logit_identity_vs_side_length.png'))
    plt.close()

    return stats, df



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--track-lengths', action='store_true')
    parser.add_argument('--bbox-areas', action='store_true')

    args = parser.parse_args()

    analyze_track_lengths = args.track_lengths
    analyze_track_bbox_areas = args.bbox_areas

    download_tracking_pkls()

    if analyze_track_lengths:
        length_stats = analyze_lengths()
        print('\n')
        print('========== Track Lengths (Duration) vs Identification ==========')
        print(length_stats)
        print('\n')
    if analyze_track_bbox_areas:
        print('\n')
        print('========== Track Bbox Areas vs Identification ==========')
        bbox_area_stats, _ = analyze_bbox_areas()
        print(bbox_area_stats)
        print('\n')
