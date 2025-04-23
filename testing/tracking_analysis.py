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
        filename = io_utils.get_unique_filename(file_dir, 'cos_dist_heatmap.png')
        heatmap_df = df.copy()

        heatmap_df['duration_bin'] = pd.cut(df['duration_sec'], bins=5)
        heatmap_df['bbox_area_bin'] = pd.cut(df['median_bbox_area'], bins=5)

        col_labels = [
            f"{int(b.left)}-{int(b.right)}" for b in
            heatmap_df['duration_bin'].cat.categories
        ]
        row_labels = [
            f"{int(b.left)}-{int(b.right)}" for b in
            heatmap_df['bbox_area_bin'].cat.categories
        ]

        pivot = (
            heatmap_df.groupby(['bbox_area_bin', 'duration_bin'], observed=False)
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
        plt.title('Mean Cosine Distance by BBox Area and Duration')
        plt.xlabel('Track Duration (seconds)')
        plt.ylabel('Median BBox Area (pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    durations = []
    identifications = []
    num_face_frames = []
    median_bbox_areas = []
    avg_min_cos_dists = []

    for fps, all_tracks in data:
        for trk in all_tracks.values():
            if hasattr(trk, 'span') and isinstance(trk.span, list):
                start, end = trk.span
                duration = max(0, end - start) / fps

                boxes = list(trk.object_detections.values())
                areas = [box[2] * box[3] for box in boxes if len(box) >= 4]
                side_lengths = np.sqrt(areas)

                median_area = np.median(areas) if areas else np.nan

                face_frames = trk.face_detections.keys()
                num_faces = len(face_frames)

                dists = []
                for df in trk.face_detections.values():
                    if 'distance' in df:
                        dists.append(df['distance'].min())
                mean_min_cos_dist = np.mean(dists) if dists else np.nan

                durations.append(duration)
                identifications.append(bool(getattr(trk, 'identity', None)))
                median_bbox_areas.append(median_area)
                avg_min_cos_dists.append(mean_min_cos_dist)
                num_face_frames.append(num_faces)

    if not durations:
        return 'No valid track durations found'

    trackwise_stats = pd.DataFrame({
        'duration_sec': durations,
        'has_identity': identifications,
        'num_face_frames': num_face_frames,
        'avg_min_cos_dist': avg_min_cos_dists,
        'median_bbox_area': median_bbox_areas,
    })

    X = np.log(trackwise_stats['duration_sec'].values + 1)[:, None]
    X = sm.add_constant(X)
    y = trackwise_stats['has_identity'].astype(int)

    model = sm.Logit(y, X).fit(disp=0)

    params = pd.Series(model.params)
    pvalues = pd.Series(model.pvalues)

    ols_output = pd.DataFrame([{
        'feature': 'duration_sec',
        'coef': params.iloc[1],
        'intercept': params.iloc[0],
        'p_value': pvalues.iloc[1],
        'r_squared': model.prsquared,   #McFadden's pseudo-R^2
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

        plt.figure()
        sns.barplot(x='area_bin', y='has_identity', data=bin_summary)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.title('Identification Rate by Avg BBox Area')
        plt.ylabel('identification rate')
        plt.xlabel('avg bbox area (√pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()

    def _chart_area_vs_duration(df, file_dir):
        def _run_loglog_regression(x_column, prefix, records):
            X = np.log(df[x_column] + 1).values
            y = np.log(df['duration_sec'] + 1).values
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            params = pd.Series(model.params)
            pvalues = pd.Series(model.pvalues)

            records.append({
                'feature': x_column,
                'coef': params.iloc[1],
                'intercept': params.iloc[0],
                'p_value': pvalues.iloc[1],
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
            plt.xlabel(f'{prefix} bbox area (√pixels)')
            plt.ylabel('track duration (seconds)')
            plt.yscale('log')
            # plt.xscale('log')
            title_prefix = prefix[0].upper() + prefix[1:]
            plt.title(f'{title_prefix} Bbox Area vs Duration')
            plt.legend()
            plt.grid(True, which='both', ls='--')
            plt.tight_layout()
            filename = io_utils.get_unique_filename(file_dir, f'{prefix}-area_vs_duration__regression.png')
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
            df.groupby('bbox_bin', observed=False)['median_min_cos_dist']
            .mean().reset_index()
        )

        plt.figure()
        sns.barplot(x='bbox_bin', y='median_min_cos_dist', data=bin_summary)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.title('Median Cosine Distance by Avg BBox Area Bin')
        plt.ylabel('median cosine distance')
        plt.xlabel('avg bbox area (√pixels, binned)')
        plt.tight_layout()
        plt.savefig(os.path.join(file_dir, filename))
        plt.close()


    identifications = []
    avg_areas = []
    q75_areas = []
    avg_side_lengths = []
    q75_side_lengths = []
    durations = []
    median_min_cos_dists = []

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

            dists = []
            for df in trk.face_detections.values():
                if 'distance' in df:
                    dists.append(df['distance'].min())
            median_cos_dist = np.median(dists) if dists else np.nan
            median_min_cos_dists.append(median_cos_dist)


            avg_area = np.mean(areas)
            q75_area = np.percentile(areas, 75)

            avg_areas.append(avg_area)
            q75_areas.append(q75_area)
            avg_side_lengths.append(np.sqrt(avg_area))
            q75_side_lengths.append(np.sqrt(q75_area))
            identifications.append(bool(getattr(trk, 'identity', None)))

    if not avg_areas:
        return 'No valid track data found'

    trackwise_stats = pd.DataFrame({
        'avg_bbox_area': avg_areas,
        'q75_bbox_area': q75_areas,
        'avg_bbox_side_length': avg_side_lengths,
        'q75_bbox_side_length': q75_side_lengths,
        'duration_sec': durations,
        'has_identity': identifications,
        'median_min_cos_dist': median_min_cos_dists
    })

    _chart_area_histogram(trackwise_stats, file_dir)
    _chart_area_boxplot(trackwise_stats, file_dir)
    _chart_area_bins(trackwise_stats, file_dir)
    _chart_area_vs_cosdist_barchart(trackwise_stats, file_dir)
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

    print('\n')
    print(f'min_length = {min_length}')
    print(f'var_percentile = {var_percentile}')
    print('\n')
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
