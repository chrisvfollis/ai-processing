# standard dependencies
import pickle
import os
import re
import argparse
from typing import Union

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


def prepare_tracking_data(
        pkl_dir='../files/output/',
        min_duration_sec: Union[float, int] = 1.0,
        var_percentile: int = 5
    ):
    '''
    Loads and cleans pickled tracking data.

    Args:
        min_duration_sec: Filters short-lived tracks likely caused by transient
                          object detector noise. Tracks with durations below this
                          threshold are discarded.

        var_percentile: Filters stationary objects erroneously tracked as people
                        for extended periods. A track's average [x, y, w, h] variance
                        must fall above this percentile within the distribution of
                        all track-wise variances.

    Returns:
        List of (fps, cleaned_tracks) tuples.
    '''

    all_variances = []
    loaded_data = []

    len_filtered = 0
    var_filtered = 0

    # load and compute variances
    for fname in os.listdir(pkl_dir):
        if fname.endswith('.pkl') and not fname.endswith('inference_data.pkl'):
            path = os.path.join(pkl_dir, fname)
            try:
                with open(path, 'rb') as f:
                    pipeline = pickle.load(f)

                fps = pipeline.fps
                track_variances = []

                for trk in pipeline.all_trks.values():
                    if not hasattr(trk, 'span') or not isinstance(trk.span, list):
                        continue
                    start, end = trk.span
                    duration = max(0, end - start) / fps
                    if duration <= min_duration_sec:
                        len_filtered += 1
                        continue  # filter short, irrelevant tracks

                    boxes = list(trk.object_detections.values())
                    if len(boxes) < 2:
                        continue

                    arr = np.array([[box[0], box[1], box[2], box[3]]
                                    for box in boxes if len(box) >= 4])
                    if arr.shape[0] < 2:
                        continue

                    var = np.var(arr, axis=0).mean()
                    track_variances.append((trk, var))

                if track_variances:
                    loaded_data.append((fps, track_variances))
                    all_variances.extend([v for _, v in track_variances])

            except Exception as e:
                print(f'Failed to process {fname}: {e}')

    if not all_variances:
        return []
    
    # filter using variance threshold
    var_cutoff = np.percentile(all_variances, var_percentile)
    cleaned_data = []
    for fps, trk_var_pairs in loaded_data:
        cleaned_tracks = {}
        for trk, var in trk_var_pairs:
            if var > var_cutoff:
                cleaned_tracks[trk.track_id] = trk
            else:
                var_filtered += 1
        if cleaned_tracks:
            cleaned_data.append((fps, cleaned_tracks))
    
    print(f'{len_filtered} short-lived tracks filtered')
    print(f'{var_filtered} tracked stationary objects filtered')

    return cleaned_data


def analyze_lengths(data, pkl_dir='../files/output/'):
    '''
    Models relationships between track length (duration) and other attributes or
    outcomes.
    '''
    def _plot_duration_histogram(durations):
        plt.figure()
        plt.hist(durations, bins=30, edgecolor='black')
        plt.yscale('log')
        plt.title('Track Duration Distribution')
        plt.xlabel('Track Duration (seconds, log scale)')
        plt.ylabel('Frequency')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'track_duration_histogram.png'))
        plt.close()

    def _plot_boxplot(df):
        plt.figure()
        sns.boxplot(x='has_identity', y='duration_sec', data=df)
        plt.yscale('log')
        plt.title('Track Duration by Identity Assignment')
        plt.xlabel('Has Identity')
        plt.ylabel('Track Duration (seconds)')
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'duration_vs_identity_boxplot.png'))
        plt.close()

    def _plot_identity_bins(df):
        bin_max = utils.logceil_round(np.max(df['duration_sec']))
        bins = sorted(set([0] + [bin_max // i for i in range(10, 0, -1)]))

        df['duration_bin'] = pd.cut(df['duration_sec'], bins=bins, include_lowest=True, right=False)
        bin_summary = df.groupby('duration_bin')['has_identity'].mean().reset_index()

        plt.figure()
        sns.barplot(x='duration_bin', y='has_identity', data=bin_summary)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.title('Identity Assignment Rate by Track Duration Bin')
        plt.ylabel('Fraction with Identity')
        plt.xlabel('Track Duration Bin (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'identity_assignment_by_duration_bin.png'))
        plt.close()

    def _plot_heatmap(df):
        heatmap_df = df.copy()

        heatmap_df['duration_bin'] = pd.cut(df['duration_sec'], bins=10)
        heatmap_df['face_frame_bin'] = pd.cut(df['num_face_frames'], bins=10)

        row_labels = [f"{int(b.left)}-{int(b.right)}" for b in heatmap_df['duration_bin'].cat.categories]
        col_labels = [f"{int(b.left)}-{int(b.right)}" for b in heatmap_df['face_frame_bin'].cat.categories]

        pivot = (
            heatmap_df
            .groupby(['duration_bin', 'face_frame_bin'])['avg_min_cos_dist']
            .mean()
            .unstack()
        )

        pivot.index = row_labels
        pivot.columns = col_labels

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot, annot=True, cmap='coolwarm', fmt=".3f",
            cbar_kws={'label': 'Mean Min Cosine Distance'}
        )
        plt.title('Mean Cosine Distance by Duration and Face Detection Count')
        plt.xlabel('Face Frame Count (binned)')
        plt.ylabel('Track Duration (binned)')
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'cos_dist_heatmap.png'))
        plt.close()

    durations = []
    identities = []
    num_face_frames = []
    avg_min_cos_dists = []

    for fps, all_tracks in data:
        for trk in all_tracks.values():
            if hasattr(trk, 'span') and isinstance(trk.span, list):
                start, end = trk.span
                duration = max(0, end - start) / fps

                face_frames = trk.face_detections.keys()
                num_faces = len(face_frames)

                dists = []
                for df in trk.face_detections.values():
                    if 'distance' in df:
                        dists.append(df['distance'].min())

                mean_min_cos_dist = np.mean(dists) if dists else np.nan

                durations.append(duration)
                identities.append(bool(getattr(trk, 'identity', None)))
                num_face_frames.append(num_faces)
                avg_min_cos_dists.append(mean_min_cos_dist)

    if not durations:
        return 'No valid track durations found'

    df = pd.DataFrame({
        'duration_sec': durations,
        'has_identity': identities,
        'num_face_frames': num_face_frames,
        'avg_min_cos_dist': avg_min_cos_dists
    })

    X = np.log(df['duration_sec'].values + 1)[:, None]
    X = sm.add_constant(X)
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

    _plot_duration_histogram(durations)
    _plot_boxplot(df)
    _plot_identity_bins(df)
    _plot_heatmap(df)

    return stats, df


def analyze_bbox_areas(data, pkl_dir='../files/output/'):

    def _chart_area_histogram(df, pkl_dir):
        plt.figure()
        plt.hist(df['avg_bbox_area'], bins=30, edgecolor='black')
        plt.yscale('log')
        plt.title('Avg. Bounding Box Area Distribution')
        plt.xlabel('Average BBox Area (pixels²)')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'bbox_area_histogram.png'))
        plt.close()

    def _chart_area_boxplot(df, pkl_dir):
        plt.figure()
        sns.boxplot(x='has_identity', y='avg_bbox_area', data=df)
        plt.yscale('log')
        plt.title('Avg. BBox Area by Identity Assignment')
        plt.xlabel('Has Identity')
        plt.ylabel('Average BBox Area (pixels²)')
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'bbox_area_boxplot.png'))
        plt.close()

    def _chart_area_bins(df, pkl_dir):
        bin_max = utils.logceil_round(df['avg_bbox_area'].max())
        bins = sorted(set([0] + [bin_max // i for i in range(10, 0, -1)]))

        df['area_bin'] = pd.cut(
            df['avg_bbox_area'],
            bins=bins,
            include_lowest=True,
            right=False
        )

        bin_summary = df.groupby('area_bin')['has_identity'].mean().reset_index()

        plt.figure()
        sns.barplot(x='area_bin', y='has_identity', data=bin_summary)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.title('Identity Assignment Rate by Avg. BBox Area Bin')
        plt.ylabel('Fraction with Identity')
        plt.xlabel('Avg. BBox Area Bin (pixels²)')
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'identity_assignment_by_area_bin.png'))
        plt.close()
    
    def _chart_area_vs_duration(df, pkl_dir):
        X = np.log(df['avg_bbox_side_length'] + 1).values
        y = np.log(df['duration_sec'] + 1).values
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        print(model.summary())

        x_vals = np.linspace(min(df['avg_bbox_side_length']), max(df['avg_bbox_side_length']), 100)
        X_plot = sm.add_constant(np.log(x_vals + 1))
        y_pred = model.predict(X_plot)

        plt.figure()
        plt.scatter(df['avg_bbox_side_length'], df['duration_sec'], alpha=0.3, label='Data')
        plt.plot(x_vals, np.exp(y_pred) - 1, color='red', label='Log-log OLS fit')
        plt.xlabel('Avg BBox Side Length (pixels)')
        plt.ylabel('Track Duration (seconds)')
        plt.yscale('log')
        plt.xscale('log')
        plt.title('Track Duration vs BBox Side Length')
        plt.legend()
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'duration_vs_bbox_size_regression.png'))
        plt.close()

    def _chart_face_bbox_area_boxplot(data, pkl_dir):
        face_bbox_areas = []
        has_identity_flags = []

        for _, all_tracks in data:
            for trk in all_tracks.values():
                if not hasattr(trk, 'face_detections'):
                    continue

                identity_assigned = bool(getattr(trk, 'identity', None))

                for frame in trk.face_detections.keys():
                    box = trk.object_detections.get(frame, None)
                    if box and len(box) >= 4:
                        area = box[2] * box[3]
                        face_bbox_areas.append(area)
                        has_identity_flags.append(identity_assigned)

        if not face_bbox_areas:
            print('No face-associated bounding boxes found.')
            return

        df = pd.DataFrame({
            'face_bbox_area': face_bbox_areas,
            'has_identity': has_identity_flags
        })

        plt.figure()
        sns.boxplot(x='has_identity', y='face_bbox_area', data=df)
        plt.yscale('log')
        plt.title('BBox Area in Face-Detected Frames by Identity Assignment')
        plt.xlabel('Has Identity')
        plt.ylabel('BBox Area (pixels²)')
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(os.path.join(pkl_dir, 'face_bbox_area_boxplot.png'))
        plt.close()

    avg_areas = []
    side_lengths = []
    identities = []
    durations = []

    for fps, all_tracks in data:
        for trk in all_tracks.values():
            boxes = list(trk.object_detections.values())
            areas = [box[2] * box[3] for box in boxes if len(box) >= 4]
            if not areas:
                continue
            
        if not hasattr(trk, 'span') or not isinstance(trk.span, list):
            continue

        start, end = trk.span
        duration = max(0, end - start) / fps
        durations.append(duration)

        avg_area = np.mean(areas)
        side_length = np.sqrt(avg_area)

        avg_areas.append(avg_area)
        side_lengths.append(side_length)
        identities.append(bool(getattr(trk, 'identity', None)))

    if not avg_areas:
        return 'No valid track data found'

    df = pd.DataFrame({
        'avg_bbox_area': avg_areas,
        'avg_bbox_side_length': side_lengths,
        'duration_sec': durations,
        'has_identity': identities
    })

    _chart_area_histogram(df, pkl_dir)
    _chart_area_boxplot(df, pkl_dir)
    _chart_area_bins(df, pkl_dir)
    _chart_area_vs_duration(df, pkl_dir)
    _chart_face_bbox_area_boxplot(data, pkl_dir)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--all', action='store_true')
    parser.add_argument('--track-lengths', action='store_true')
    parser.add_argument('--bbox-areas', action='store_true')

    parser.add_argument('--min-length', type=float)
    parser.add_argument('--var-percentile', type=int)

    args = parser.parse_args()

    analyze_track_lengths = args.all or args.track_lengths
    analyze_track_bbox_areas = args.all or args.bbox_areas

    min_length = args.min_length or 1.0
    var_percentile = args.var_percentile or 5

    download_tracking_pkls()

    data = prepare_tracking_data(
        min_duration_sec=min_length,
        var_percentile=var_percentile
    )

    if analyze_track_lengths:
        length_stats, _ = analyze_lengths(data)
        print('\n')
        print('========== Track Lengths (Duration) vs Identification ==========')
        print(length_stats)
        print('\n')
    if analyze_track_bbox_areas:
        print('\n')
        print('========== Track Bbox Areas vs Identification ==========')
        bbox_area_stats = analyze_bbox_areas(data)
        print(bbox_area_stats)
        print('\n')
