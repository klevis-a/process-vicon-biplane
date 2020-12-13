from matplotlib import font_manager


def update_xticks(ax, font_size=12, font_weight='bold'):
    ax.xaxis.set_tick_params(width=2)
    fp = font_manager.FontProperties(weight=font_weight, size=font_size)
    for label in ax.get_xticklabels():
        label.set_fontproperties(fp)


def update_yticks(ax, fontsize=12, fontweight='bold'):
    ax.yaxis.set_tick_params(width=2, pad=0.1)
    fp = font_manager.FontProperties(weight=fontweight, size=fontsize)
    for label in ax.get_yticklabels():
        label.set_fontproperties(fp)


def update_spines(ax, line_width=2):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)


def update_spines_add(ax, line_width=2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_linewidth(line_width)


def update_spines2(ax_l, ax_r, line_width=2):
    ax_l.spines['top'].set_visible(False)
    ax_r.spines['top'].set_visible(False)
    ax_l.spines['left'].set_linewidth(line_width)
    ax_r.spines['right'].set_linewidth(line_width)
    ax_l.spines['bottom'].set_linewidth(line_width)


def update_ylabel(ax, ylabel_text, font_size=12, font_weight='bold', labelpad=2):
    ax.set_ylabel(ylabel_text, fontdict=dict(fontsize=font_size, fontweight=font_weight), labelpad=labelpad)


def update_xlabel(ax, xlabel_text, font_size=12, font_weight='bold'):
    ax.set_xlabel(xlabel_text, fontdict=dict(fontsize=font_size, fontweight=font_weight), labelpad=2)


def update_title(ax, title_text, font_size=12, font_weight='bold', pad=-8):
    ax.set_title(title_text, fontdict=dict(fontsize=font_size, fontweight=font_weight), pad=pad)
