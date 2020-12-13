import numpy as np
from matplotlib import patches
from biplane_kine.graphing import plot_utils


def get_bp_limits(bp):
    min_max = np.zeros((len(bp['fliers']), 3))
    for idx, (flier, lWhisker, uWhisker) in enumerate(zip(bp['fliers'], bp['whiskers'][0::2], bp['whiskers'][1::2])):
        x_data = np.append(flier.get_xdata(), np.append(lWhisker.get_xdata(), uWhisker.get_xdata()))
        unique_x = np.unique(x_data.round(decimals=4))
        assert (len(unique_x) == 1)
        y_data = np.append(flier.get_ydata(), np.append(lWhisker.get_ydata(), uWhisker.get_ydata()))
        min_max[idx, 0] = unique_x
        min_max[idx, 1] = min(y_data)
        min_max[idx, 2] = max(y_data)
    return min_max


def calc_bar_positions(bp_data, num_metrics):
    num_bars = bp_data.shape[1]
    num_obs = bp_data.shape[0]
    num_groups = num_bars // num_metrics
    assert (num_bars % num_metrics == 0)
    box_positions = np.arange(num_bars + num_groups - 1)
    box_positions = np.delete(box_positions, np.arange(num_metrics, num_bars + num_groups - 1, num_metrics + 1)) + 1

    return num_bars, num_obs, num_groups, box_positions


def create_rudimentary_bp(ax, bp_data, box_positions, box_widths=0.8, whiskers=1.5):
    bp = ax.boxplot(bp_data, widths=box_widths, positions=box_positions, patch_artist=True, whis=whiskers)
    return bp


def update_bp_boxes(bp, num_metrics, color_map, line_width=1.5):
    for idx, box in enumerate(bp['boxes']):
        box.set(facecolor=color_map.colors[idx % num_metrics], edgecolor=color_map.colors[idx % num_metrics],
                linewidth=line_width)


def update_bp_whiskers(bp, num_metrics, color_map, line_width=1.5):
    for idx, (lWhisker, uWhisker) in enumerate(zip(bp['whiskers'][0::2], bp['whiskers'][1::2])):
        lWhisker.set(color=color_map.colors[idx % num_metrics], linewidth=line_width)
        uWhisker.set(color=color_map.colors[idx % num_metrics], linewidth=line_width)


def update_bp_caps(bp, num_metrics, color_map, line_width=1.5):
    for idx, (lCap, uCap) in enumerate(zip(bp['caps'][0::2], bp['caps'][1::2])):
        lCap.set(color=color_map.colors[idx % num_metrics], linewidth=line_width)
        uCap.set(color=color_map.colors[idx % num_metrics], linewidth=line_width)


def update_bp_medians(bp, line_width=1.5):
    for median in bp['medians']:
        median.set(color='black', linewidth=line_width, linestyle='-')


def update_bp_fliers(bp, num_metrics, color_map):
    for idx, flier in enumerate(bp['fliers']):
        flier.set(marker='.', markerfacecolor=color_map.colors[idx % num_metrics], fillstyle='full',
                  markeredgecolor=color_map.colors[idx % num_metrics])


def bp_vertical_sep(ax, num_groups, num_metrics, line_color='0.75'):
    for i in range(1, num_groups):
        ax.axvline(i * (num_metrics + 1), color=line_color)


def update_bp_legend_numobs(ax, bp, num_metrics, num_obs, legend_entries, legend_position, fontsize=12,
                            is_draggable=True):
    # create an invisible rectangle
    n_legend_entry = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    legend = ax.legend(
        bp['boxes'][0:num_metrics] + [n_legend_entry], legend_entries + ['(n=' + str(num_obs) + ')'],
        loc='upper left', bbox_to_anchor=legend_position, fontsize=fontsize)
    legend.set_draggable(is_draggable)


def update_bp_legend(ax, bp, num_metrics, legend_entries, legend_position, fontsize=12, ncols=1, is_draggable=True):
    legend = ax.legend(bp['boxes'][0:num_metrics], legend_entries, loc='upper left', bbox_to_anchor=legend_position,
                       fontsize=fontsize, ncol=ncols)
    legend.set_draggable(is_draggable)


def update_bp_xticks_groups(ax, num_bars, num_groups, num_metrics, tick_labels, font_size=12, font_weight='bold'):
    first_tick = np.average(np.arange(num_metrics) + 1)
    ax.set_xticks(np.arange(first_tick, num_bars + num_groups, num_metrics + 1))
    ax.set_xticklabels(tick_labels)
    plot_utils.update_xticks(ax, font_size=font_size, font_weight=font_weight)
    ax.xaxis.set_tick_params(which='both', length=0)


def update_bp_xticks(ax, num_bars, tick_labels, font_size=12, font_weight='bold'):
    ax.set_xticks(np.arange(num_bars))
    ax.set_xticklabels(tick_labels)
    plot_utils.update_xticks(ax, font_size=font_size, font_weight=font_weight)
    ax.xaxis.set_tick_params(which='both', length=0)
