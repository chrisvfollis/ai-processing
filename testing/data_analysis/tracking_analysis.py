# standard dependencies
import os
import argparse

# 3rd-party dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import percentileofscore

# internal dependencies
from utilities import io_utils
import utilities.general_utils as utils
from testing import test_utils


def generate_tracking_stats(data, ideal_bbox_area=693):
    '''
    Extracts and computes all shared per-track statistics from tracking data.

    Returns:
        - trackwise_stats: pd.DataFrame with one row per track
        - overall_stats: pd.DataFrame with summary stats
        - ols_output: pd.DataFrame with regression coefficients across models
    '''

    def _run_logit(x_col):
        X = np.log(trackwise_stats[x_col] + 1).values[:, None]
        X = sm.add_constant(X)
        y = trackwise_stats['has_identity'].astype(int)

        model = sm.Logit(y, X).fit(disp=0)
        params = pd.Series(model.params)
        pvalues = pd.Series(model.pvalues)

        return {
            'feature': x_col,
            'coef': params.iloc[1],
            'intercept': params.iloc[0],
            'p_value': pvalues.iloc[1],
            'r_squared': model.prsquared,
            'model_type': 'Logit',
            'module': 'has_identity'
        }

    def _run_ols(y_col, x_col):
        X = np.log(trackwise_stats[x_col] + 1).values[:, None]
        X = sm.add_constant(X)
        y = np.log(trackwise_stats[y_col] + 1).values

        model = sm.OLS(y, X).fit()
        params = pd.Series(model.params)
        pvalues = pd.Series(model.pvalues)

        return {
            'feature': x_col,
            'coef': params.iloc[1],
            'intercept': params.iloc[0],
            'p_value': pvalues.iloc[1],
            'r_squared': model.rsquared,
            'model_type': 'OLS',
            'module': y_col
        }

    durations = []
    identifications = []
    num_face_frames = []
    overall_min_cos_dists = []
    avg_min_cos_dists = []
    median_min_cos_dists = []

    avg_areas = []
    q75_areas = []
    avg_root_areas = []
    avg_side_lengths = []
    q75_side_lengths = []
    median_bbox_areas = []
    median_bbox_side_lengths = []

    for fps, all_tracks in data:
        for trk in all_tracks.values():
            if not hasattr(trk, 'span') or not isinstance(trk.span, list):
                continue

            start, end = trk.span
            duration = max(0, end - start) / fps

            boxes = list(trk.object_detections.values())
            areas = [box[2] * box[3] for box in boxes if len(box) >= 4]
            if not areas:
                continue

            face_frames = trk.face_detections.keys()
            num_faces = len(face_frames)

            dists = []
            for df in trk.face_detections.values():
                if 'distance' in df:
                    dists.append(df['distance'].min())
            overall_min_cos_dist = min(dists) if dists else np.nan
            mean_min_cos_dist = np.mean(dists) if dists else np.nan
            median_cos_dist = np.median(dists) if dists else np.nan

            avg_area = np.mean(areas)
            q75_area = np.percentile(areas, 75)
            avg_root_area = np.sqrt(avg_area)
            median_area = np.median(areas)

            durations.append(duration)
            identifications.append(bool(getattr(trk, 'identity', None)))
            num_face_frames.append(num_faces)
            overall_min_cos_dists.append(overall_min_cos_dist)
            avg_min_cos_dists.append(mean_min_cos_dist)
            median_min_cos_dists.append(median_cos_dist)

            avg_areas.append(avg_area)
            q75_areas.append(q75_area)
            avg_root_areas.append(avg_root_area)
            median_bbox_areas.append(median_area)
            median_bbox_side_lengths.append(np.sqrt(median_area))
            avg_side_lengths.append(np.sqrt(avg_area))
            q75_side_lengths.append(np.sqrt(q75_area))

    if not durations:
        return None, None, None

    trackwise_stats = pd.DataFrame({
        'duration_sec': durations,
        'has_identity': identifications,
        'num_face_frames': num_face_frames,
        'overall_min_cos_dist': overall_min_cos_dists,
        'avg_min_cos_dist': avg_min_cos_dists,
        'median_min_cos_dist': median_min_cos_dists,
        'avg_bbox_area': avg_areas,
        'q75_bbox_area': q75_areas,
        'avg_root_area': avg_root_areas,
        'median_bbox_area': median_bbox_areas,
        'median_bbox_side_lengths': median_bbox_side_lengths,
        'avg_bbox_side_length': avg_side_lengths,
        'q75_bbox_side_length': q75_side_lengths
    })

    # --- Overall stats ---
    overall_stats = pd.DataFrame([{
        'average_duration_sec': np.mean(durations),
        'median_duration_sec': np.median(durations),
        'min_duration_sec': np.min(durations),
        'max_duration_sec': np.max(durations),
        'with_identity': sum(identifications),
        'without_identity': len(identifications) - sum(identifications),
        'total_tracks': len(durations),
        'average_bbox_area': np.mean(avg_areas),
        'median_bbox_area': np.median(median_bbox_areas),
        'ideal_bbox_area_percentile': percentileofscore(
            avg_root_areas, ideal_bbox_area, kind='weak'
        ),
        'average_cos_dist': np.nanmean(avg_min_cos_dists),
        'median_cos_dist': np.nanmedian(median_min_cos_dists),
        'module': 'unified_summary'
    }])

    # --- OLS/Logit models ---
    ols_output = []

    ols_output.append(_run_logit('duration_sec'))
    ols_output.append(_run_ols('duration_sec', 'avg_bbox_side_length'))
    ols_output.append(_run_ols('duration_sec', 'q75_bbox_side_length'))

    ols_output_df = pd.DataFrame(ols_output)

    return (trackwise_stats, overall_stats, ols_output_df)


def track_duration_charts(tracking_stats, file_dir='../../files/output/'):
    '''
    Models relationships between track length (duration) and other attributes or
    outcomes.
    '''
    def _plot_duration_histogram(durations):
        filename = io_utils.get_unique_filename(file_dir, 'duration__histogram.png')
        plt.figure()
        plt.hist(durations, bins=30, edgecolor='black')
        plt.yscale('log')
        plt.title('Distribution of Track Durations')
        plt.xlabel('track duration (seconds)')
        plt.ylabel('frequency')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    def _plot_boxplot(df):
        filename = io_utils.get_unique_filename(file_dir, 'duration-vs-ids_boxplot.png')
        plt.figure()
        sns.boxplot(x='has_identity', y='duration_sec', data=df)
        plt.yscale('log')
        plt.title('Durations of Identified vs Unidentified Tracks')
        plt.xlabel('Has Identity')
        plt.ylabel('Track Duration (seconds)')
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    def _plot_identity_bins(df):
        filename = io_utils.get_unique_filename(file_dir, 'duration-vs-ids__barchart.png')

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
        plt.title('Identification Rate by Track Duration')
        plt.ylabel('identification rate')
        plt.xlabel('track duration (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    def _plot_heatmap(df):
        filename = io_utils.get_unique_filename(file_dir, 'cos_dist__heatmap.png')
        heatmap_df = df.copy()

        heatmap_df['duration_bin'] = pd.cut(df['duration_sec'], bins=5)
        heatmap_df['bbox_area_bin'] = pd.cut(df['median_bbox_side_lengths'], bins=5)

        col_bins = heatmap_df['duration_bin'].cat.categories
        row_bins = heatmap_df['bbox_area_bin'].cat.categories

        col_labels = [f"{int(b.left)}-{int(b.right)}" for b in col_bins]
        row_labels = [f"{int(b.left)}-{int(b.right)}" for b in row_bins]

        pivot = (
            heatmap_df.groupby(['bbox_area_bin', 'duration_bin'], observed=False)
            ['avg_min_cos_dist']
            .mean().unstack()
        )

        pivot.index = row_labels
        pivot.columns = col_labels

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            pivot, annot=True, cmap='crest', fmt=".3f",
            cbar_kws={'label': 'Mean Cosine Distance'}
        )

        plt.title('Mean Cosine Distance by BBox Area and Duration\nwith Trendline Contours')
        plt.xlabel('track duration (seconds)')
        plt.ylabel('median bbox area (√pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    def _duration_vs_cosdist_barchart(df, file_dir):
        filename = io_utils.get_unique_filename(file_dir, 'duration-vs-cosdist__barchart.png')

        bin_max = utils.logceil_round(df['duration_sec'].max())
        bins = sorted(set([0] + [bin_max // i for i in range(10, 0, -1)]))

        df['duration_bin'] = pd.cut(
            df['duration_sec'],
            bins=bins,
            include_lowest=True,
            right=False
        )

        bin_summary = (
            df.groupby('duration_bin', observed=False)['overall_min_cos_dist']
            .mean().reset_index()
        )

        bin_summary['x'] = np.arange(len(bin_summary))

        bin_summary['bin_center'] = [
            (interval.left + interval.right) / 2 for interval in bin_summary['duration_bin']
        ]

        X = sm.add_constant(bin_summary['bin_center'])
        y = bin_summary['overall_min_cos_dist']
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)

        plt.figure()
        sns.barplot(x='x', y='overall_min_cos_dist', data=bin_summary, color='skyblue')
        plt.plot(bin_summary['x'], y_pred, color='black', marker='o', linestyle='-', label='Trendline')

        plt.xticks(
            ticks=bin_summary['x'],
            labels=bin_summary['duration_bin'].astype(str),
            rotation=45
        )
        plt.ylim(0, 1)
        plt.title('Min Cosine Distance by Track Duration')
        plt.ylabel('min cosine distance')
        plt.xlabel('track duration (seconds)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

        print(f"Trend R²: {model.rsquared:.3f}")

    trackwise_stats, _, _ = tracking_stats

    _plot_duration_histogram(trackwise_stats['duration_sec'])
    _plot_boxplot(trackwise_stats)
    _plot_identity_bins(trackwise_stats)
    _plot_heatmap(trackwise_stats)
    _duration_vs_cosdist_barchart(trackwise_stats, file_dir)


def track_bbox_charts(tracking_stats, file_dir='../../files/output/'):

    def _chart_area_histogram(df, file_dir):
        filename = io_utils.get_unique_filename(file_dir, 'avg-area__histogram.png')

        plt.figure()
        plt.hist(df['avg_bbox_side_length'], bins=30, edgecolor='black')
        plt.yscale('log')
        plt.title('Trackwise Avg BBox Areas')
        plt.xlabel('avg bbox area (√pixels)')
        plt.ylabel('frequency')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    def _chart_area_boxplot(df, file_dir):
        filename = io_utils.get_unique_filename(file_dir, 'avg-area_vs_ids__boxplot.png')
        plt.figure()
        sns.boxplot(x='has_identity', y='avg_bbox_side_length', data=df)
        plt.yscale('log')
        plt.title('Avg BBox Area by Identity Assignment')
        plt.xlabel('has identity')
        plt.ylabel('avg bbox area (√pixels)')
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    def _chart_area_bins(df, file_dir):
        filename = io_utils.get_unique_filename(file_dir, 'avg-area_vs_ids__barchart.png')

        bin_max = utils.logceil_round(df['avg_bbox_side_length'].max())
        bins = sorted(set([0] + [bin_max // i for i in range(10, 0, -1)]))

        df['area_bin'] = pd.cut(
            df['avg_bbox_side_length'],
            bins=bins,
            include_lowest=True,
            right=False
        )

        bin_summary = (
            df.groupby('area_bin', observed=False)['has_identity']
            .mean().reset_index()
        )

        bin_centers = [
            (interval.left + interval.right) / 2
            for interval in bin_summary['area_bin']
        ]
        bin_summary['bin_center'] = bin_centers

        X = sm.add_constant(bin_summary['bin_center'])
        y = bin_summary['has_identity']
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)

        # assign ordinal positions for each bin:
        bin_summary['x'] = np.arange(len(bin_summary))

        plt.figure()
        sns.barplot(x='x', y='has_identity', data=bin_summary, color='skyblue')
        plt.plot(bin_summary['x'], y_pred, color='black', marker='o', linestyle='-', label='Trendline')
        plt.xticks(ticks=bin_summary['x'], labels=bin_summary['area_bin'].astype(str), rotation=45)
        plt.ylim(0, 1)
        plt.title('Identification Rate by Avg BBox Area')
        plt.ylabel('identification rate')
        plt.xlabel('avg bbox area (√pixels)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

        print(f"Trend R²: {model.rsquared:.3f}")

    def _chart_area_vs_duration(df, ols_output, file_dir):
        for prefix, column in [('avg', 'avg_bbox_side_length'), ('q75', 'q75_bbox_side_length')]:
            row = ols_output[
                (ols_output['module'] == 'duration_sec') &
                (ols_output['feature'] == column)
            ].squeeze()

            if row.empty:
                continue  # skip if the model isn't present

            coef = row['coef']
            intercept = row['intercept']

            x_vals = np.linspace(df[column].min(), df[column].max(), 100)
            X_plot = np.log(x_vals + 1)
            y_pred = intercept + coef * X_plot

            plt.figure()
            plt.scatter(df[column], df['duration_sec'], alpha=0.3, label='Data')
            plt.plot(x_vals, np.exp(y_pred) - 1, color='red', label='Log-log OLS fit')
            plt.xlabel(f'{prefix} bbox area (√pixels)')
            plt.ylabel('track duration (seconds)')
            plt.yscale('log')
            title_prefix = prefix[0].upper() + prefix[1:]
            plt.title(f'{title_prefix} Bbox Area vs Duration')
            plt.legend()
            plt.grid(True, which='both', ls='--')
            plt.tight_layout()
            filename = io_utils.get_unique_filename(file_dir, f'{prefix}-area_vs_duration__regression.png')
            plt.savefig(os.path.join(file_dir, filename))
            plt.close()

    def _chart_area_vs_cosdist_barchart(df, file_dir):
        filename = io_utils.get_unique_filename(file_dir, 'avg-area_vs_median-cosdist__barchart.png')

        bin_max = utils.logceil_round(df['avg_bbox_side_length'].max())
        bins = sorted(set([0] + [bin_max // i for i in range(10, 0, -1)]))

        df['bbox_bin'] = pd.cut(
            df['avg_bbox_side_length'],
            bins=bins,
            include_lowest=True,
            right=False
        )

        bin_summary = (
            df.groupby('bbox_bin', observed=False)['avg_min_cos_dist']
            .mean().reset_index()
        )

        plt.figure()
        sns.barplot(x='bbox_bin', y='avg_min_cos_dist', data=bin_summary)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.title('Avg Cosine Distance by Avg BBox Area Bin')
        plt.ylabel('mean cosine distance')
        plt.xlabel('avg bbox area (√pixels, binned)')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    trackwise_stats, _, ols_output = tracking_stats

    _chart_area_histogram(trackwise_stats, file_dir)
    _chart_area_boxplot(trackwise_stats, file_dir)
    _chart_area_bins(trackwise_stats, file_dir)
    _chart_area_vs_cosdist_barchart(trackwise_stats, file_dir)
    _chart_area_vs_duration(trackwise_stats, ols_output, file_dir)


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

    print('\n')
    print(f'min_length = {min_length}')
    print(f'var_percentile = {var_percentile}')
    print('\n')
    data = test_utils.prepare_tracking_data(
        min_duration_sec=min_length,
        var_percentile=var_percentile
    )
    tracking_stats = generate_tracking_stats(data)

    if analyze_track_lengths:
        track_duration_charts(tracking_stats)

    if analyze_track_bbox_areas:
        track_bbox_charts(tracking_stats)
    
    test_utils.export_tracking_analysis(*tracking_stats)
