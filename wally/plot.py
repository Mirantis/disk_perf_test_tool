import logging
from io import BytesIO
from functools import wraps
from typing import Tuple, cast, List, Callable, Optional, Any

import numpy
import scipy.stats
import matplotlib.axis
import matplotlib.style
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# to make seaborn styles available
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn

from cephlib.plot import process_heatmap_data, hmap_from_2d, do_plot_hmap_with_histo

from .hlstorage import ResultStorage
from .utils import unit_conversion_coef
from .statistic import moving_average, moving_dev, hist_outliers_perc, find_ouliers_ts, approximate_curve
from .result_classes import StatProps, DataSource, TimeSeries, NormStatProps
from .report_profiles import StyleProfile, ColorProfile
from .resources import IOSummary


logger = logging.getLogger("wally")


# --------------  PLOT HELPERS FUNCTIONS  ------------------------------------------------------------------------------

def get_emb_image(fig: Figure, file_format: str, **opts) -> bytes:
    bio = BytesIO()
    if file_format == 'svg':
        fig.savefig(bio, format='svg', **opts)
        img_start = "<!-- Created with matplotlib (http://matplotlib.org/) -->"
        return bio.getvalue().decode("utf8").split(img_start, 1)[1].encode("utf8")
    else:
        fig.savefig(bio, format=file_format, **opts)
        return bio.getvalue()


class PlotParams:
    def __init__(self, fig: Figure, ax: Any, title: str,
                 style: StyleProfile, colors: ColorProfile) -> None:
        self.fig = fig
        self.ax = ax
        self.style = style
        self.colors = colors
        self.title = title


def provide_plot(noaxis: bool = False,
                 eng: bool = False,
                 no_legend: bool = False,
                 long_plot: bool = True,
                 grid: Any = None,
                 style_name: str = 'default',
                 noadjust: bool = False) -> Callable[..., Callable[..., str]]:
    def closure1(func: Callable[..., None]) -> Callable[..., str]:
        @wraps(func)
        def closure2(storage: ResultStorage,
                     style: StyleProfile,
                     colors: ColorProfile,
                     path: DataSource,
                     title: Optional[str],
                     *args, **kwargs) -> str:
            fpath = storage.check_plot_file(path)
            if not fpath:

                assert style_name in ('default', 'ioqd')
                mlstyle = style.default_style if style_name == 'default' else style.io_chart_style
                with matplotlib.style.context(mlstyle):
                    file_format = path.tag.split(".")[-1]
                    fig = plt.figure(figsize=style.figsize_long if long_plot else style.figsize)

                    if not noaxis:
                        xlabel = kwargs.pop('xlabel', None)
                        ylabel = kwargs.pop('ylabel', None)
                        ax = fig.add_subplot(111)

                        if xlabel is not None:
                            ax.set_xlabel(xlabel)

                        if ylabel is not None:
                            ax.set_ylabel(ylabel)

                        if grid:
                            ax.grid(axis=grid)
                    else:
                        ax = None

                    if title:
                        fig.suptitle(title, fontsize=style.title_font_size)

                    pp = PlotParams(fig, ax, title, style, colors)
                    func(pp, *args, **kwargs)
                    apply_style(pp, eng=eng, no_legend=no_legend, noadjust=noadjust)

                    fpath = storage.put_plot_file(get_emb_image(fig, file_format=file_format, dpi=style.dpi), path)
                    logger.debug("Plot %s saved to %r", path, fpath)
                    plt.close(fig)
            return fpath
        return closure2
    return closure1


def apply_style(pp: PlotParams, eng: bool = True, no_legend: bool = False, noadjust: bool = False) -> None:

    if (pp.style.legend_for_eng or not eng) and not no_legend:
        if not noadjust:
            pp.fig.subplots_adjust(right=StyleProfile.subplot_adjust_r)
        legend_location = "center left"
        legend_bbox_to_anchor = (1.03, 0.81)

        for ax in pp.fig.axes:
            ax.legend(loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)
    elif not noadjust:
        pp.fig.subplots_adjust(right=StyleProfile.subplot_adjust_r_no_legend)

    if pp.style.tide_layout:
        pp.fig.set_tight_layout(True)


# --------------  PLOT FUNCTIONS  --------------------------------------------------------------------------------------


@provide_plot(eng=True)
def plot_hist(pp: PlotParams, units: str, prop: StatProps) -> None:

    normed_bins = prop.bins_populations / prop.bins_populations.sum()
    bar_width = prop.bins_edges[1] - prop.bins_edges[0]
    pp.ax.bar(prop.bins_edges, normed_bins, color=pp.colors.box_color, width=bar_width, label="Real data")

    pp.ax.set(xlabel=units, ylabel="Value probability")

    if isinstance(prop, NormStatProps):
        nprop = cast(NormStatProps, prop)
        stats = scipy.stats.norm(nprop.average, nprop.deviation)

        new_edges, step = numpy.linspace(prop.bins_edges[0], prop.bins_edges[-1],
                                         len(prop.bins_edges) * 10, retstep=True)

        ypoints = stats.cdf(new_edges) * 11
        ypoints = [nextpt - prevpt for (nextpt, prevpt) in zip(ypoints[1:], ypoints[:-1])]
        xpoints = (new_edges[1:] + new_edges[:-1]) / 2

        pp.ax.plot(xpoints, ypoints, color=pp.colors.primary_color, label="Expected from\nnormal\ndistribution")

    pp.ax.set_xlim(left=prop.bins_edges[0])
    if prop.log_bins:
        pp.ax.set_xscale('log')


@provide_plot(grid='y')
def plot_simple_over_time(pp: PlotParams, tss: List[Tuple[str, numpy.ndarray]], average: bool = False) -> None:
    max_len = 0
    for name, arr in tss:
        if average:
            avg_vals = moving_average(arr, pp.style.avg_range)
            if pp.style.approx_average_no_points:
                time_points = numpy.arange(len(avg_vals))
                avg_vals = approximate_curve(cast(List[int], time_points),
                                             avg_vals,
                                             cast(List[int], time_points),
                                             pp.style.curve_approx_level)
            arr = avg_vals
        pp.ax.plot(arr, label=name)
        max_len = max(max_len, len(arr))
    pp.ax.set_xlim(-5, max_len + 5)


@provide_plot(no_legend=True, grid='x', noadjust=True)
def plot_simple_bars(pp: PlotParams,
                     names: List[str],
                     values: List[float],
                     errs: List[float] = None,
                     x_formatter: Callable[[float, float], str] = None,
                     one_point_zero_line: bool = True) -> None:

    ind = numpy.arange(len(names))
    width = 0.35
    pp.ax.barh(ind, values, width, xerr=errs)

    pp.ax.set_yticks(ind)
    pp.ax.set_yticklabels(names)
    pp.ax.set_xlim(0, max(val + err for val, err in zip(values, errs)) * 1.1)

    if one_point_zero_line:
        pp.ax.axvline(x=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)

    if x_formatter:
        pp.ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))

    pp.fig.subplots_adjust(left=0.2)


@provide_plot(no_legend=True, long_plot=True, noaxis=True)
def plot_hmap_from_2d(pp: PlotParams, data2d: numpy.ndarray, xlabel: str, ylabel: str,
                      bins: numpy.ndarray = None) -> None:
    ioq1d, ranges = hmap_from_2d(data2d)
    heatmap, bins = process_heatmap_data(ioq1d, bin_ranges=ranges, bins=bins)
    bins_populations, _ = numpy.histogram(ioq1d, bins)

    ax, _ = do_plot_hmap_with_histo(pp.fig,
                                    heatmap,
                                    bins_populations,
                                    bins,
                                    cmap=pp.colors.hmap_cmap,
                                    cbar=pp.style.heatmap_colorbar,
                                    histo_grid=pp.style.histo_grid)
    ax.set(ylabel=ylabel, xlabel=xlabel)


@provide_plot(eng=True, grid='y')
def plot_v_over_time(pp: PlotParams, units: str, ts: TimeSeries,
                     plot_avg_dev: bool = True, plot_points: bool = True) -> None:

    min_time = min(ts.times)

    # convert time to ms
    coef = float(unit_conversion_coef(ts.time_units, 's'))
    time_points = numpy.array([(val_time - min_time) * coef for val_time in ts.times])

    outliers_idxs = find_ouliers_ts(ts.data, cut_range=pp.style.outliers_q_nd)
    outliers_4q_idxs = find_ouliers_ts(ts.data, cut_range=pp.style.outliers_hide_q_nd)
    normal_idxs = numpy.logical_not(outliers_idxs)
    outliers_idxs = outliers_idxs & numpy.logical_not(outliers_4q_idxs)
    # hidden_outliers_count = numpy.count_nonzero(outliers_4q_idxs)

    data = ts.data[normal_idxs]
    data_times = time_points[normal_idxs]
    outliers = ts.data[outliers_idxs]
    outliers_times = time_points[outliers_idxs]

    if plot_points:
        alpha = pp.colors.noise_alpha if plot_avg_dev else 1.0
        pp.ax.plot(data_times, data, pp.style.point_shape, color=pp.colors.primary_color, alpha=alpha, label="Data")
        pp.ax.plot(outliers_times, outliers, pp.style.err_point_shape, color=pp.colors.err_color, label="Outliers")

    has_negative_dev = False
    plus_minus = "\xb1"

    if plot_avg_dev and len(data) < pp.style.avg_range * 2:
        logger.warning("Array %r to small to plot average over %s points", pp.title, pp.style.avg_range)
    elif plot_avg_dev:
        avg_vals = moving_average(data, pp.style.avg_range)
        dev_vals = moving_dev(data, pp.style.avg_range)
        avg_times = moving_average(data_times, pp.style.avg_range)

        if (plot_points and pp.style.approx_average) or (not plot_points and pp.style.approx_average_no_points):
            avg_vals = approximate_curve(avg_times, avg_vals, avg_times, pp.style.curve_approx_level)
            dev_vals = approximate_curve(avg_times, dev_vals, avg_times, pp.style.curve_approx_level)

        pp.ax.plot(avg_times, avg_vals, c=pp.colors.suppl_color1, label="Average")

        low_vals_dev = avg_vals - dev_vals * pp.style.dev_range_x
        hight_vals_dev = avg_vals + dev_vals * pp.style.dev_range_x
        if (pp.style.dev_range_x - int(pp.style.dev_range_x)) < 0.01:
            pp.ax.plot(avg_times, low_vals_dev, c=pp.colors.suppl_color2,
                       label="{}{}*stdev".format(plus_minus, int(pp.style.dev_range_x)))
        else:
            pp.ax.plot(avg_times, low_vals_dev, c=pp.colors.suppl_color2,
                       label="{}{}*stdev".format(plus_minus, pp.style.dev_range_x))
        pp.ax.plot(avg_times, hight_vals_dev, c=pp.colors.suppl_color2)
        has_negative_dev = low_vals_dev.min() < 0

    pp.ax.set_xlim(-5, max(time_points) + 5)
    pp.ax.set_xlabel("Time, seconds from test begin")

    if plot_avg_dev:
        pp.ax.set_ylabel("{}. Average and {}stddev over {} points".format(units, plus_minus, pp.style.avg_range))
    else:
        pp.ax.set_ylabel(units)

    if has_negative_dev:
        pp.ax.set_ylim(bottom=0)


@provide_plot(eng=True, no_legend=True, grid='y', noadjust=True)
def plot_lat_over_time(pp: PlotParams, ts: TimeSeries) -> None:
    times = ts.times - min(ts.times)
    step = len(times) / pp.style.lat_samples
    points = [times[int(i * step + 0.5)] for i in range(pp.style.lat_samples)]
    points.append(times[-1])
    bounds = list(zip(points[:-1], points[1:]))
    agg_data = []
    positions = []
    labels = []

    for begin, end in bounds:
        agg_hist = ts.data[begin:end].sum(axis=0)

        if pp.style.violin_instead_of_box:
            # cut outliers
            idx1, idx2 = hist_outliers_perc(agg_hist, pp.style.outliers_lat)
            agg_hist = agg_hist[idx1:idx2]
            curr_bins_vals = ts.histo_bins[idx1:idx2]

            correct_coef = pp.style.violin_point_count / sum(agg_hist)
            if correct_coef > 1:
                correct_coef = 1
        else:
            curr_bins_vals = ts.histo_bins
            correct_coef = 1

        vals = numpy.empty(shape=[numpy.sum(agg_hist)], dtype='float32')
        cidx = 0

        non_zero, = agg_hist.nonzero()
        for pos in non_zero:
            count = int(agg_hist[pos] * correct_coef + 0.5)

            if count != 0:
                vals[cidx: cidx + count] = curr_bins_vals[pos]
                cidx += count

        agg_data.append(vals[:cidx])
        positions.append((end + begin) / 2)
        labels.append(str((end + begin) // 2))

    if pp.style.violin_instead_of_box:
        patches = pp.ax.violinplot(agg_data, positions=positions, showmeans=True, showmedians=True, widths=step / 2)
        patches['cmeans'].set_color("blue")
        patches['cmedians'].set_color("green")
        if pp.style.legend_for_eng:
            legend_location = "center left"
            legend_bbox_to_anchor = (1.03, 0.81)
            pp.ax.legend([patches['cmeans'], patches['cmedians']], ["mean", "median"],
                         loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)
    else:
        pp.ax.boxplot(agg_data, 0, '', positions=positions, labels=labels, widths=step / 4)

    pp.ax.set_xlim(min(times), max(times))
    pp.ax.set_xlabel("Time, seconds from test begin, sampled for ~{} seconds".format(int(step)))
    pp.fig.subplots_adjust(right=pp.style.subplot_adjust_r)


@provide_plot(eng=True, no_legend=True, noaxis=True, long_plot=True)
def plot_histo_heatmap(pp: PlotParams, ts: TimeSeries, ylabel: str, xlabel: str = "time, s") -> None:

    # only histogram-based ts can be plotted
    assert len(ts.data.shape) == 2

    # Find global outliers. As load is expected to be stable during one job
    # outliers range can be detected globally
    total_hist = ts.data.sum(axis=0)
    idx1, idx2 = hist_outliers_perc(total_hist,
                                    bounds_perc=pp.style.outliers_lat,
                                    min_bins_left=pp.style.hm_hist_bins_count)

    # merge outliers with most close non-outliers cell
    orig_data = ts.data[:, idx1:idx2].copy()
    if idx1 > 0:
        orig_data[:, 0] += ts.data[:, :idx1].sum(axis=1)

    if idx2 < ts.data.shape[1]:
        orig_data[:, -1] += ts.data[:, idx2:].sum(axis=1)

    bins_vals = ts.histo_bins[idx1:idx2]

    # rebin over X axis
    # aggregate some lines in ts.data to plot ~style.hm_x_slots x bins
    agg_idx = float(len(orig_data)) / pp.style.hm_x_slots
    if agg_idx >= 2:
        idxs = list(map(int, numpy.round(numpy.arange(0, len(orig_data) + 1, agg_idx))))
        assert len(idxs) > 1
        data = numpy.empty([len(idxs) - 1, orig_data.shape[1]], dtype=numpy.float32)  # type: List[numpy.ndarray]
        for idx, (sidx, eidx) in enumerate(zip(idxs[:-1], idxs[1:])):
            data[idx] = orig_data[sidx:eidx,:].sum(axis=0) / (eidx - sidx)
    else:
        data = orig_data

    # rebin over Y axis
    # =================

    # don't using rebin_histogram here, as we need apply same bins for many arrays
    step = (bins_vals[-1] - bins_vals[0]) / pp.style.hm_hist_bins_count
    new_bins_edges = numpy.arange(pp.style.hm_hist_bins_count) * step + bins_vals[0]
    bin_mapping = numpy.clip(numpy.searchsorted(new_bins_edges, bins_vals) - 1, 0, len(new_bins_edges) - 1)

    # map origin bins ranges to heatmap bins, iterate over rows
    cmap = []
    for line in data:
        curr_bins = [0] * pp.style.hm_hist_bins_count
        for idx, count in zip(bin_mapping, line):
            curr_bins[idx] += count
        cmap.append(curr_bins)
    ncmap = numpy.array(cmap)

    histo = ncmap.sum(axis=0).reshape((-1,))
    ax, _ = do_plot_hmap_with_histo(pp.fig, ncmap, histo, new_bins_edges,
                                    cmap=pp.colors.hmap_cmap,
                                    cbar=pp.style.heatmap_colorbar, avg_labels=True)
    ax.set(ylabel=ylabel, xlabel=xlabel)


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

    coef_mb = float(unit_conversion_coef(iosums[0].bw.units, "MiBps"))
    coef_iops = float(unit_conversion_coef(iosums[0].bw.units, "KiBps")) / block_size

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
    lat_coef_ms = float(unit_conversion_coef(iosums[0].lat.units, "ms"))
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
    pp.fig.subplots_adjust(right=StyleProfile.subplot_adjust_r)

    if pp.style.extra_io_spine:
        ax3.grid(False)

