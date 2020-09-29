from pythonGraphingLibrary import plotUtils


def marker_diff_graph(ax, marker_data, y_label, x_label, x_data):
    lines = ax.plot(x_data, marker_data)
    plotUtils.update_spines(ax)
    plotUtils.update_xticks(ax, font_size=8)
    plotUtils.update_yticks(ax, fontsize=8)
    if x_label:
        plotUtils.update_xlabel(ax, x_label, font_size=10)
    plotUtils.update_ylabel(ax, y_label, font_size=10)
    return lines
