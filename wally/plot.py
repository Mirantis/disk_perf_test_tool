import logging
from typing import List

import numpy

from cephlib.units import unit_conversion_coef_f
from cephlib.plot import PlotParams, provide_plot

from .resources import IOSummary


logger = logging.getLogger("wally")


@provide_plot(eng=False, no_legend=True, grid='y', style_name='ioqd', noadjust=True)
def io_chart(pp: PlotParams,
             legend: str,
             iosums: List[IOSummary],
             iops_log_spine: bool = False,
             lat_log_spine: bool = False) -> None:

    # --------------  MAGIC VALUES  ---------------------
    # IOPS bar width
    width = 0.2

    # offset from center of bar to deviation/confidence range indicator
    err_x_offset = 0.03

    # extra space on top and bottom, comparing to maximal tight layout
    extra_y_space = 0.05

    # additional spine for BW/IOPS on left side of plot
    extra_io_spine_x_offset = -0.1

    # extra space on left and right sides
    extra_x_space = 0.5

    # legend location settings
    legend_location = "center left"
    legend_bbox_to_anchor = (1.1, 0.81)

    # --------------  END OF MAGIC VALUES  ---------------------

    block_size = iosums[0].block_size
    xpos = numpy.arange(1, len(iosums) + 1, dtype='uint')

    coef_mb = unit_conversion_coef_f(iosums[0].bw.units, "MiBps")
    coef_iops = unit_conversion_coef_f(iosums[0].bw.units, "KiBps") / block_size

    iops_primary = block_size < pp.style.large_blocks

    coef = coef_iops if iops_primary else coef_mb
    pp.ax.set_ylabel("IOPS" if iops_primary else "BW (MiBps)")

    vals = [iosum.bw.average * coef for iosum in iosums]

    # set correct x limits for primary IO spine
    min_io = min(iosum.bw.average - iosum.bw.deviation * pp.style.dev_range_x for iosum in iosums)
    max_io = max(iosum.bw.average + iosum.bw.deviation * pp.style.dev_range_x for iosum in iosums)
    border = (max_io - min_io) * extra_y_space
    io_lims = (min_io - border, max_io + border)

    pp.ax.set_ylim(io_lims[0] * coef, io_lims[-1] * coef)
    pp.ax.bar(xpos - width / 2, vals, width=width, color=pp.colors.box_color, label=legend)

    # plot deviation and confidence error ranges
    err1_legend = err2_legend = None
    for pos, iosum in zip(xpos, iosums):
        dev_bar_pos = pos - err_x_offset
        err1_legend = pp.ax.errorbar(dev_bar_pos,
                                     iosum.bw.average * coef,
                                     iosum.bw.deviation * pp.style.dev_range_x * coef,
                                     alpha=pp.colors.subinfo_alpha,
                                     color=pp.colors.suppl_color1)  # 'magenta'

        conf_bar_pos = pos + err_x_offset
        err2_legend = pp.ax.errorbar(conf_bar_pos,
                                     iosum.bw.average * coef,
                                     iosum.bw.confidence * coef,
                                     alpha=pp.colors.subinfo_alpha,
                                     color=pp.colors.suppl_color2)  # 'teal'

    handles1, labels1 = pp.ax.get_legend_handles_labels()

    handles1 += [err1_legend, err2_legend]
    labels1 += ["{}% dev".format(pp.style.dev_perc),
                "{}% conf".format(int(100 * iosums[0].bw.confidence_level))]

    # extra y spine for latency on right side
    ax2 = pp.ax.twinx()

    # plot median and 95 perc latency
    lat_coef_ms = unit_conversion_coef_f(iosums[0].lat.units, "ms")
    ax2.plot(xpos, [iosum.lat.perc_50 * lat_coef_ms for iosum in iosums], label="lat med")
    ax2.plot(xpos, [iosum.lat.perc_95 * lat_coef_ms for iosum in iosums], label="lat 95%")

    for grid_line in ax2.get_ygridlines():
        grid_line.set_linestyle(":")

    # extra y spine for BW/IOPS on left side
    if pp.style.extra_io_spine:
        ax3 = pp.ax.twinx()
        if iops_log_spine:
            ax3.set_yscale('log')

        ax3.set_ylabel("BW (MiBps)" if iops_primary else "IOPS")
        secondary_coef = coef_mb if iops_primary else coef_iops
        ax3.set_ylim(io_lims[0] * secondary_coef, io_lims[1] * secondary_coef)
        ax3.spines["left"].set_position(("axes", extra_io_spine_x_offset))
        ax3.spines["left"].set_visible(True)
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.set_ticks_position('left')
    else:
        ax3 = None

    ax2.set_ylabel("Latency (ms)")

    # legend box
    handles2, labels2 = ax2.get_legend_handles_labels()
    pp.ax.legend(handles1 + handles2, labels1 + labels2, loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)

    # limit and label x spine
    pp.ax.set_xlim(extra_x_space, len(iosums) + extra_x_space)
    pp.ax.set_ylim(bottom=0)
    pp.ax.set_xticks(xpos)
    pp.ax.set_xticklabels(["{0}*{1}={2}".format(iosum.qd, iosum.nodes_count, iosum.qd * iosum.nodes_count)
                          for iosum in iosums],
                          rotation=30 if len(iosums) > 9 else 0)
    pp.ax.set_xlabel("IO queue depth * test node count = total parallel requests")

    # apply log scales for X spines, if set
    if iops_log_spine:
        pp.ax.set_yscale('log')

    if lat_log_spine:
        ax2.set_yscale('log')

    # override some styles
    pp.fig.set_size_inches(*pp.style.qd_chart_inches)
    pp.fig.subplots_adjust(right=pp.style.subplot_adjust_r)

    if pp.style.extra_io_spine:
        ax3.grid(False)

