from pythonGraphingLibrary import plotUtils, bp_utils
import matplotlib.pyplot as plt


def create_summary_boxplot(ax, diff_data, y_label, group_names, metric_names, n_obs_pos):
    num_metrics = len(metric_names)
    # calculations
    num_bars, num_obs, num_groups, box_positions = bp_utils.calc_bar_positions(diff_data, num_metrics)

    # create figure and rudimentary boxplot
    bp = bp_utils.create_rudimentary_bp(ax, diff_data, box_positions)

    # colormap
    color_map = plt.get_cmap('Dark2')

    # now start updating boxplot elements
    bp_utils.update_bp_boxes(bp, num_metrics, color_map)
    bp_utils.update_bp_whiskers(bp, num_metrics, color_map)
    bp_utils.update_bp_caps(bp, num_metrics, color_map)
    bp_utils.update_bp_medians(bp)
    bp_utils.update_bp_fliers(bp, num_metrics, color_map)
    bp_utils.bp_vertical_sep(ax, num_groups, num_metrics)
    bp_utils.update_bp_xticks_groups(ax, num_bars, num_groups, num_metrics, group_names, font_size=12)
    plotUtils.update_yticks(ax, fontsize=10)
    plotUtils.update_spines(ax)
    plotUtils.update_ylabel(ax, y_label, font_size=12)

    # add the number of observations
    props = dict(boxstyle='round', facecolor='none')
    ax.text(n_obs_pos[0], n_obs_pos[1], 'n=' + str(num_obs), transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    return bp


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import sys
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from scipy.stats import ttest_rel
    from matplotlib.backends.backend_pdf import PdfPages
    from biplane_tasks.parameters import read_smoothing_exceptions, marker_smoothing_exceptions
    from biplane_kine.database import create_db
    from biplane_kine.database.vicon_accuracy import BiplaneMarkerSubjectEndpts
    from biplane_kine.graphing.vicon_accuracy_plotters import ViconAccuracySmoothingPlotter
    from biplane_kine.graphing.common_graph_utils import init_graphing
    from biplane_kine.smoothing.kf_filtering_helpers import InsufficientDataError, DoNotUseMarkerError
    from biplane_kine.misc.json_utils import Params
    from .smoothing_effect_marker import add_c3d_helper, marker_accuracy_diff
    import logging
    from logging.config import fileConfig

    # initialize
    config_dir = Path(sys.argv[1])
    params = Params.get_params(config_dir / 'parameters.json')

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    all_exceptions = read_smoothing_exceptions(params.smoothing_exceptions)
    db = create_db(params.accuracy_db_dir, BiplaneMarkerSubjectEndpts)
    add_c3d_helper(db, params.labeled_c3d_dir, params.filled_c3d_dir)

    # create summary database
    summary_db = []
    for (subject_name, trial_name, biplane_trial, c3d_trial) in zip(db['Subject_Name'], db['Trial_Name'],
                                                                    db['Biplane_Marker_Trial'], db['C3D_Trial']):
        for marker in biplane_trial.markers:
            marker_except = marker_smoothing_exceptions(all_exceptions, trial_name, marker)
            try:
                bi_vcn_diff = marker_accuracy_diff(biplane_trial, c3d_trial, marker, marker_except, db.attrs['dt'])
            except InsufficientDataError as e:
                log.error('Insufficient data for trial {} marker {}: {}'.format(trial_name, marker, e))
                continue
            except DoNotUseMarkerError as e:
                log.error('Marker {} for trial {} should not be used: {}'.format(trial_name, marker, e))
                continue
            plotter = ViconAccuracySmoothingPlotter(trial_name, marker, bi_vcn_diff)
            summary_db.append((subject_name, trial_name, marker, *bi_vcn_diff.raw_rms, bi_vcn_diff.raw_rms_scalar,
                               *bi_vcn_diff.raw_mae, bi_vcn_diff.raw_mae_scalar, *bi_vcn_diff.raw_max,
                               bi_vcn_diff.raw_max_scalar, *bi_vcn_diff.smoothed_rms, bi_vcn_diff.smoothed_rms_scalar,
                               *bi_vcn_diff.smoothed_mae, bi_vcn_diff.smoothed_mae_scalar, *bi_vcn_diff.smoothed_max,
                               bi_vcn_diff.smoothed_max_scalar, plotter))

    cols = {
        'Subject_Name': pd.StringDtype(),
        'Trial_Name': pd.StringDtype(),
        'Marker': pd.StringDtype(),
        'Raw_RMS_x': np.float64,
        'Raw_RMS_y': np.float64,
        'Raw_RMS_z': np.float64,
        'Raw_RMS': np.float64,
        'Raw_MAE_x': np.float64,
        'Raw_MAE_y': np.float64,
        'Raw_MAE_z': np.float64,
        'Raw_MAE': np.float64,
        'Raw_Max_x': np.float64,
        'Raw_Max_y': np.float64,
        'Raw_Max_z': np.float64,
        'Raw_Max': np.float64,
        'Smooth_RMS_x': np.float64,
        'Smooth_RMS_y': np.float64,
        'Smooth_RMS_z': np.float64,
        'Smooth_RMS': np.float64,
        'Smooth_MAE_x': np.float64,
        'Smooth_MAE_y': np.float64,
        'Smooth_MAE_z': np.float64,
        'Smooth_MAE': np.float64,
        'Smooth_Max_x': np.float64,
        'Smooth_Max_y': np.float64,
        'Smooth_Max_z': np.float64,
        'Smooth_Max': np.float64,
        'plotter': object
    }
    summary_df = pd.DataFrame.from_records(summary_db, columns=list(cols.keys()))
    summary_df.astype(cols)
    # for some reason the call above does not respect strings
    summary_df['Subject_Name'] = summary_df['Subject_Name'].astype(pd.StringDtype())
    summary_df['Trial_Name'] = summary_df['Trial_Name'].astype(pd.StringDtype())
    summary_df['Marker'] = summary_df['Marker'].astype(pd.StringDtype())
    summary_df.set_index(['Trial_Name', 'Marker'], drop=False, inplace=True, verify_integrity=True)

    # create PDFs
    root_path = Path(params.output_dir)
    for subject_name, subject_df in summary_df.groupby('Subject_Name'):
        log.info('Outputting subject %s', subject_name)
        subject_dir = (root_path / subject_name)
        subject_dir.mkdir(parents=True, exist_ok=True)
        for trial_name, trial_df in subject_df.groupby(level=0):
            log.info('Outputting trial %s', trial_name)
            trial_dir = subject_dir / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_pdf_file = subject_dir / (trial_name + '.pdf')
            with PdfPages(trial_pdf_file) as trial_pdf:
                for marker, plotter in zip(trial_df['Marker'], trial_df['plotter']):
                    log.info('Outputting marker %s', marker)
                    figs = plotter.plot()
                    marker_pdf_file = trial_dir / (marker + '.pdf')
                    with PdfPages(marker_pdf_file) as marker_pdf:
                        for fig_num, fig in enumerate(figs):
                            marker_pdf.savefig(fig)
                            if fig_num in [0, 2, 4]:
                                trial_pdf.savefig(fig)
                            fig.clf()
                            plt.close(fig)

    # create plot of statistics
    init_graphing()
    fig = plt.figure()
    axs = fig.subplots()
    summary_data = summary_df[['Raw_RMS', 'Smooth_RMS', 'Raw_MAE', 'Smooth_MAE', 'Raw_Max', 'Smooth_Max']].to_numpy()
    bp_plt = create_summary_boxplot(axs, summary_data, 'Vicon-Biplane (mm)', ['RMS', 'MAE', 'Max'], ['Raw', 'Smoothed'],
                                    (0.92, 1.0))

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    fig.suptitle('Vicon Marker - Biplane \n (Raw vs Smoothed)', fontsize=12, fontweight='bold')
    plt.subplots_adjust(top=0.92)
    legend = fig.legend(bp_plt['boxes'][0:2], ['Raw', 'Smoothed'], loc='upper left', bbox_to_anchor=(0.01, 1.00),
                        fontsize=11, ncol=2, columnspacing=0.5, handletextpad=0.3, handlelength=1.0)
    legend.set_draggable(True)

    plt.show()

    # print p-values
    _, pvalue_rms = ttest_rel(summary_df['Raw_RMS'].to_numpy(), summary_df['Smooth_RMS'].to_numpy())
    _, pvalue_mae = ttest_rel(summary_df['Raw_MAE'].to_numpy(), summary_df['Smooth_MAE'].to_numpy())
    _, pvalue_max = ttest_rel(summary_df['Raw_Max'].to_numpy(), summary_df['Smooth_Max'].to_numpy())
    print('p-value of paired t-test for {}: {:.5f}'.format('RMS', pvalue_rms))
    print('p-value of paired t-test for {}: {:.5f}'.format('MAE', pvalue_mae))
    print('p-value of paired t-test for {}: {:.5f}'.format('Max', pvalue_max))
