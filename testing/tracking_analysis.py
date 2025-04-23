# standard dependencies
import os
import argparse

# 3rd-party dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# internal dependencies
from utilities import io_utils
import utilities.general_utils as utils
from utilities import test_utils



def analyze_lengths(data, file_dir='../files/output/'):
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
        plt.savefig(os.path.join(file_dir, 'track_duration_histogram.png'))
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
        plt.savefig(os.path.join(file_dir, 'duration-vs-ids_boxplot.png'))
        plt.close()

    def _plot_identity_bins(df):
        bin_max = utils.logceil_round(np.max(df['duration_sec']))
        bins = sorted(set([0] + [bin_max // i for i in range(10, 0, -1)]))

        df['duration_bin'] = pd.cut(
            df['duration_sec'],
            bins=bins,
            include_lowest=True,
            right=False
        )
        bin_summary = (
            df.groupby('duration_bin', observed=False)['has_identity']
            .mean().reset_index()
        )

        plt.figure()
        sns.barplot(x='duration_bin', y='has_identity', data=bin_summary)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.title('Identity Assignment Rate by Track Duration Bin')
        plt.ylabel('Fraction with Identity')
        plt.xlabel('Track Duration Bin (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, 'duration-vs-ids_barchart.png'))
        plt.close()

    def _plot_heatmap(df):
        heatmap_df = df.copy()

        heatmap_df['duration_bin'] = pd.cut(df['duration_sec'], bins=10)
        heatmap_df['face_frame_bin'] = pd.cut(df['num_face_frames'], bins=10)

        row_labels = [
            f"{int(b.left)}-{int(b.right)}" for b in
            heatmap_df['duration_bin'].cat.categories
        ]
        col_labels = [
            f"{int(b.left)}-{int(b.right)}" for b in
            heatmap_df['face_frame_bin'].cat.categories
        ]

        pivot = (
            heatmap_df.groupby(['duration_bin', 'face_frame_bin'], observed=False)
            ['avg_min_cos_dist']
            .mean().unstack()
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
        plt.savefig(os.path.join(file_dir, 'cos_dist_heatmap.png'))
        plt.close()

    durations = []
    identifications = []
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
                identifications.append(bool(getattr(trk, 'identity', None)))
                num_face_frames.append(num_faces)
                avg_min_cos_dists.append(mean_min_cos_dist)

    if not durations:
        return 'No valid track durations found'

    trackwise_stats = pd.DataFrame({
        'duration_sec': durations,
        'has_identity': identifications,
        'num_face_frames': num_face_frames,
        'avg_min_cos_dist': avg_min_cos_dists
    })

    X = np.log(trackwise_stats['duration_sec'].values + 1)[:, None]
    X = sm.add_constant(X)
    y = trackwise_stats['has_identity'].astype(int)

    model = sm.Logit(y, X).fit(disp=0)

    ols_output = pd.DataFrame([{
        'feature': 'duration_sec',
        'coef': model.params[1],
        'intercept': model.params[0],
        'p_value': model.pvalues[1],
        'r_squared': None,
        'model_type': 'Logit',
        'module': 'duration'
    }])

    overall_stats = pd.DataFrame([{
        'average_duration_sec': np.mean(durations),
        'median_duration_sec': np.median(durations),
        'min_duration_sec': np.min(durations),
        'max_duration_sec': np.max(durations),
        'with_identity': trackwise_stats['has_identity'].sum(),
        'without_identity': len(trackwise_stats) - trackwise_stats['has_identity'].sum(),
        'total_tracks': len(trackwise_stats),
        'module': 'duration'
    }])

    _plot_duration_histogram(durations)
    _plot_boxplot(trackwise_stats)
    _plot_identity_bins(trackwise_stats)
    _plot_heatmap(trackwise_stats)

    return trackwise_stats, overall_stats, ols_output


def analyze_bbox_areas(data, file_dir='../files/output/'):

    def _chart_area_histogram(df, file_dir):
        plt.figure()
        plt.hist(df['avg_bbox_area'], bins=30, edgecolor='black')
        plt.yscale('log')
        plt.title('Avg. Bounding Box Area Distribution')
        plt.xlabel('Average BBox Area (pixels²)')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, 'avg-area__histogram.png'))
        plt.close()

    def _chart_area_boxplot(df, file_dir):
        plt.figure()
        sns.boxplot(x='has_identity', y='avg_bbox_area', data=df)
        plt.yscale('log')
        plt.title('Avg. BBox Area by Identity Assignment')
        plt.xlabel('Has Identity')
        plt.ylabel('Average BBox Area (pixels²)')
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, 'avg-area_vs_ids__boxplot.png'))
        plt.close()

    def _chart_area_bins(df, file_dir):
        bin_max = utils.logceil_round(df['avg_bbox_area'].max())
        bins = sorted(set([0] + [bin_max // i for i in range(10, 0, -1)]))

        df['area_bin'] = pd.cut(
            df['avg_bbox_area'],
            bins=bins,
            include_lowest=True,
            right=False
        )

        bin_summary = (
            df.groupby('area_bin', observed=False)['has_identity']
            .mean().reset_index()
        )

        plt.figure()
        sns.barplot(x='area_bin', y='has_identity', data=bin_summary)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.title('Identity Assignment Rate by Avg. BBox Area Bin')
        plt.ylabel('Fraction with Identity')
        plt.xlabel('Avg. BBox Area Bin (pixels²)')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, 'avg-area_vs_ids__barchart.png'))
        plt.close()

    def _chart_area_vs_duration(df, file_dir):
        def _run_loglog_regression(x_column, suffix, records):
            X = np.log(df[x_column] + 1).values
            y = np.log(df['duration_sec'] + 1).values
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            records.append({
                'feature': x_column,
                'coef': model.params[1],
                'intercept': model.params[0],
                'p_value': model.pvalues[1],
                'r_squared': model.rsquared,
                'model_type': 'OLS',
                'module': 'bbox_area'
            })

            x_vals = np.linspace(df[x_column].min(), df[x_column].max(), 100)
            X_plot = sm.add_constant(np.log(x_vals + 1))
            y_pred = model.predict(X_plot)

            plt.figure()
            plt.scatter(df[x_column], df['duration_sec'], alpha=0.3, label='Data')
            plt.plot(x_vals, np.exp(y_pred) - 1, color='red', label='Log-log OLS fit')
            plt.xlabel(f'{x_column.replace("_", " ").title()} (pixels)')
            plt.ylabel('Track Duration (seconds)')
            plt.yscale('log')
            plt.xscale('log')
            plt.title(f'Track Duration vs {x_column.replace("_", " ").title()}')
            plt.legend()
            plt.grid(True, which='both', ls='--')
            plt.tight_layout()
            filename = f'{suffix}-area_vs_duration__regression.png'
            plt.savefig(os.path.join(file_dir, filename))
            plt.close()

            return records
        
        ols_records = []
        ols_records = _run_loglog_regression(
            'avg_bbox_side_length', 'avg', ols_records
        )
        ols_records = _run_loglog_regression(
            'q75_bbox_side_length', 'q75', ols_records
        )

        return pd.DataFrame(ols_records)

    identifications = []
    avg_areas = []
    q75_areas = []
    side_lengths = []
    q75_side_lengths = []
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
            q75_area = np.percentile(areas, 75)

            avg_areas.append(avg_area)
            q75_areas.append(q75_area)
            side_lengths.append(np.sqrt(avg_area))
            q75_side_lengths.append(np.sqrt(q75_area))
            identifications.append(bool(getattr(trk, 'identity', None)))

    if not avg_areas:
        return 'No valid track data found'

    trackwise_stats = pd.DataFrame({
        'avg_bbox_area': avg_areas,
        'q75_bbox_area': q75_areas,
        'avg_bbox_side_length': side_lengths,
        'q75_bbox_side_length': q75_side_lengths,
        'duration_sec': durations,
        'has_identity': identifications
    })

    _chart_area_histogram(trackwise_stats, file_dir)
    _chart_area_boxplot(trackwise_stats, file_dir)
    _chart_area_bins(trackwise_stats, file_dir)
    ols_output = _chart_area_vs_duration(trackwise_stats, file_dir)

    return trackwise_stats, ols_output


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

    test_utils.download_tracking_pkls()

    print(f'min_length={min_length}')
    print(f'var_percentile={var_percentile}')

    data = test_utils.prepare_tracking_data(
        min_duration_sec=min_length,
        var_percentile=var_percentile
    )

    all_trackwise_stats = []
    all_overall_stats = []
    all_ols_output = []

    if analyze_track_lengths:
        len_trackwise, len_overall, len_ols = analyze_lengths(data)

        all_trackwise_stats.append(len_trackwise)
        all_overall_stats.append(len_overall)
        all_ols_output.append(len_ols)

    if analyze_track_bbox_areas:
        bbox_trackwise, bbox_ols = analyze_bbox_areas(data)

        all_trackwise_stats.append(bbox_trackwise)
        all_ols_output.append(bbox_ols)

    test_utils.export_tracking_analysis(
        all_trackwise_stats=all_trackwise_stats,
        all_overall_stats=all_overall_stats,
        all_ols_output=all_ols_output
    )
