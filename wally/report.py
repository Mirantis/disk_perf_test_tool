import os
import abc
import logging
import warnings
from io import BytesIO
from functools import wraps
from collections import defaultdict
from typing import Dict, Any, Iterator, Tuple, cast, List, Callable, Set, Optional, Union

import numpy
import scipy.stats
import matplotlib.style
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
from statsmodels.tsa.stattools import adfuller

from cephlib.common import float2str
from cephlib.plot import plot_hmap_with_y_histo, hmap_from_2d
import xmlbuilder3

import wally

from . import html
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .hlstorage import ResultStorage
from .utils import b2ssize, b2ssize_10, STORAGE_ROLES, unit_conversion_coef
from .statistic import (calc_norm_stat_props, calc_histo_stat_props, moving_average, moving_dev,
                        hist_outliers_perc, find_ouliers_ts, approximate_curve)
from .result_classes import (StatProps, DataSource, TimeSeries, NormStatProps, HistoStatProps, SuiteConfig)
from .suits.io.fio import FioTest, FioJobConfig
from .suits.io.fio_job import FioJobParams
from .suits.job import JobConfig
from .data_selectors import (get_aggregated, AGG_TAG, summ_sensors, find_sensors_to_2d, find_nodes_by_roles,
                             get_ts_for_time_range)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import seaborn


logger = logging.getLogger("wally")


# ----------------  CONSTS ---------------------------------------------------------------------------------------------


DEBUG = False


# ----------------  PROFILES  ------------------------------------------------------------------------------------------


# this is default values, real values is loaded from config

class ColorProfile:
    primary_color = 'b'
    suppl_color1 = 'teal'
    suppl_color2 = 'magenta'
    suppl_color3 = 'orange'
    box_color = 'y'
    err_color = 'red'

    noise_alpha = 0.3
    subinfo_alpha = 0.7

    imshow_colormap = None  # type: str
    hmap_cmap = "Blues"


default_format = 'svg'
io_chart_format = 'svg'


class StyleProfile:
    default_style = 'seaborn-white'
    io_chart_style = 'classic'

    dpi = 80
    grid = True
    tide_layout = False
    hist_boxes = 10
    hist_lat_boxes = 25
    hm_hist_bins_count = 25
    hm_x_slots = 25
    min_points_for_dev = 5

    x_label_rotation = 35

    dev_range_x = 2.0
    dev_perc = 95

    point_shape = 'o'
    err_point_shape = '*'

    avg_range = 20
    approx_average = True

    curve_approx_level = 6
    curve_approx_points = 100
    assert avg_range >= min_points_for_dev

    # figure size in inches
    figsize = (8, 4)
    figsize_long = (8, 4)
    qd_chart_inches = (16, 9)

    subplot_adjust_r = 0.75
    subplot_adjust_r_no_legend = 0.9
    title_font_size = 12

    extra_io_spine = True

    legend_for_eng = True
    # heatmap_interpolation = '1d'
    heatmap_interpolation = None
    heatmap_interpolation_points = 300
    outliers_q_nd = 3.0
    outliers_hide_q_nd = 4.0
    outliers_lat = (0.01, 0.9)

    violin_instead_of_box = True
    violin_point_count = 30000

    heatmap_colorbar = False

    min_iops_vs_qd_jobs = 3

    qd_bins = [0, 1, 2, 4, 6, 8, 12, 16, 20, 26, 32, 40, 48, 56, 64, 96, 128]
    iotime_bins = list(range(0, 1030, 50))
    block_size_bins = [0, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048]
    large_blocks = 256


DefColorProfile = ColorProfile()
DefStyleProfile = StyleProfile()


# ----------------  STRUCTS  -------------------------------------------------------------------------------------------


# TODO: need to be revised, have to user StatProps fields instead
class StoragePerfSummary:
    def __init__(self, name: str) -> None:
        self.direct_iops_r_max = 0  # type: int
        self.direct_iops_w_max = 0  # type: int

        # 64 used instead of 4k to faster feed caches
        self.direct_iops_w64_max = 0  # type: int

        self.rws4k_10ms = 0  # type: int
        self.rws4k_30ms = 0  # type: int
        self.rws4k_100ms = 0  # type: int
        self.bw_write_max = 0  # type: int
        self.bw_read_max = 0  # type: int

        self.bw = None  # type: float
        self.iops = None  # type: float
        self.lat = None  # type: float
        self.lat_50 = None  # type: float
        self.lat_95 = None  # type: float


class IOSummary:
    def __init__(self,
                 qd: int,
                 block_size: int,
                 nodes_count:int,
                 bw: NormStatProps,
                 lat: HistoStatProps) -> None:

        self.qd = qd
        self.nodes_count = nodes_count
        self.block_size = block_size

        self.bw = bw
        self.lat = lat


# --------------  AGGREGATION AND STAT FUNCTIONS  ----------------------------------------------------------------------

iosum_cache = {}  # type: Dict[Tuple[str, str]]


def make_iosum(rstorage: ResultStorage, suite: SuiteConfig, job: FioJobConfig, nc: bool = False) -> IOSummary:
    key = (suite.storage_id, job.storage_id)
    if not nc and key in iosum_cache:
        return iosum_cache[key]

    lat = get_aggregated(rstorage, suite, job, "lat")
    io = get_aggregated(rstorage, suite, job, "bw")

    res = IOSummary(job.qd,
                    nodes_count=len(suite.nodes_ids),
                    block_size=job.bsize,
                    lat=calc_histo_stat_props(lat, rebins_count=StyleProfile.hist_boxes),
                    bw=calc_norm_stat_props(io, StyleProfile.hist_boxes))

    if not nc:
        iosum_cache[key] = res

    return res


def is_sensor_numarray(sensor: str, metric: str) -> bool:
    """Returns True if sensor provides one-dimension array of numeric values. One number per one measurement."""
    return True


LEVEL_SENSORS = {("block-io", "io_queue"),
                 ("system-cpu", "procs_blocked"),
                 ("system-cpu", "procs_queue")}


def is_level_sensor(sensor: str, metric: str) -> bool:
    """Returns True if sensor measure level of any kind, E.g. queue depth."""
    return (sensor, metric) in LEVEL_SENSORS


def is_delta_sensor(sensor: str, metric: str) -> bool:
    """Returns True if sensor provides deltas for cumulative value. E.g. io completed in given period"""
    return not is_level_sensor(sensor, metric)


cpu_load_cache = {}  # type: Dict[Tuple[int, Tuple[str, ...], Tuple[int, int]], Dict[str, TimeSeries]]


def get_cluster_cpu_load(rstorage: ResultStorage, roles: List[str],
                         time_range: Tuple[int, int], nc: bool = False) -> Dict[str, TimeSeries]:

    key = (id(rstorage), tuple(roles), time_range)
    if not nc and key in cpu_load_cache:
        return cpu_load_cache[key]

    cpu_ts = {}
    cpu_metrics = "idle guest iowait sirq nice irq steal sys user".split()
    for name in cpu_metrics:
        cpu_ts[name] = summ_sensors(rstorage, roles, sensor='system-cpu', metric=name, time_range=time_range)

    it = iter(cpu_ts.values())
    total_over_time = next(it).data.copy()  # type: numpy.ndarray
    for ts in it:
        if ts is not None:
            total_over_time += ts.data

    total = cpu_ts['idle'].copy(no_data=True)
    total.data = total_over_time
    cpu_ts['total'] = total

    if not nc:
        cpu_load_cache[key] = cpu_ts

    return cpu_ts


# --------------  PLOT HELPERS FUNCTIONS  ------------------------------------------------------------------------------

def get_emb_image(fig: Figure, format: str, **opts) -> bytes:
    bio = BytesIO()
    if format == 'svg':
        fig.savefig(bio, format='svg', **opts)
        img_start = "<!-- Created with matplotlib (http://matplotlib.org/) -->"
        return bio.getvalue().decode("utf8").split(img_start, 1)[1].encode("utf8")
    else:
        fig.savefig(bio, format=format, **opts)
        return bio.getvalue()


def provide_plot(func: Callable[..., None]) -> Callable[..., str]:
    @wraps(func)
    def closure1(storage: ResultStorage,
                 path: DataSource,
                 *args, **kwargs) -> str:
        fpath = storage.check_plot_file(path)
        if not fpath:
            format = path.tag.split(".")[-1]
            fig = plt.figure(figsize=StyleProfile.figsize)
            plt.style.use(StyleProfile.default_style)
            func(fig, *args, **kwargs)
            fpath = storage.put_plot_file(get_emb_image(fig, format=format, dpi=DefStyleProfile.dpi), path)
            logger.debug("Plot %s saved to %r", path, fpath)
            plt.close(fig)
        return fpath
    return closure1


def apply_style(fig: Figure, title: str, style: StyleProfile, eng: bool = True,
                no_legend: bool = False) -> None:

    for ax in fig.axes:
        ax.grid(style.grid)

    if (style.legend_for_eng or not eng) and not no_legend:
        fig.subplots_adjust(right=StyleProfile.subplot_adjust_r)
        legend_location = "center left"
        legend_bbox_to_anchor = (1.03, 0.81)
        for ax in fig.axes:
            ax.legend(loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)
    else:
        fig.subplots_adjust(right=StyleProfile.subplot_adjust_r_no_legend)

    if style.tide_layout:
        fig.set_tight_layout(True)

    fig.suptitle(title, fontsize=style.title_font_size)


# --------------  PLOT FUNCTIONS  --------------------------------------------------------------------------------------


@provide_plot
def plot_hist(fig: Figure, title: str, units: str,
              prop: StatProps,
              colors: ColorProfile = DefColorProfile,
              style: StyleProfile = DefStyleProfile) -> None:

    ax = fig.add_subplot(111)

    # TODO: unit should came from ts
    normed_bins = prop.bins_populations / prop.bins_populations.sum()
    bar_width = prop.bins_edges[1] - prop.bins_edges[0]
    ax.bar(prop.bins_edges, normed_bins, color=colors.box_color, width=bar_width, label="Real data")

    ax.set(xlabel=units, ylabel="Value probability")

    dist_plotted = False
    if isinstance(prop, NormStatProps):
        nprop = cast(NormStatProps, prop)
        stats = scipy.stats.norm(nprop.average, nprop.deviation)

        new_edges, step = numpy.linspace(prop.bins_edges[0], prop.bins_edges[-1],
                                         len(prop.bins_edges) * 10, retstep=True)

        ypoints = stats.cdf(new_edges) * 11
        ypoints = [next - prev for (next, prev) in zip(ypoints[1:], ypoints[:-1])]
        xpoints = (new_edges[1:] + new_edges[:-1]) / 2

        ax.plot(xpoints, ypoints, color=colors.primary_color, label="Expected from\nnormal\ndistribution")
        dist_plotted = True

    ax.set_xlim(left=prop.bins_edges[0])
    if prop.log_bins:
        ax.set_xscale('log')

    apply_style(fig, title, style, eng=True, no_legend=not dist_plotted)


@provide_plot
def plot_simple_over_time(fig: Figure,
                          tss: List[Tuple[str, numpy.ndarray]],
                          title: str,
                          ylabel: str,
                          xlabel: str = "time, s",
                          average: bool = False,
                          colors: ColorProfile = DefColorProfile,
                          style: StyleProfile = DefStyleProfile) -> None:
    ax = fig.add_subplot(111)
    for name, arr in tss:
        if average:
            avg_vals = moving_average(arr, style.avg_range)
            if style.approx_average:
                time_points = numpy.arange(len(avg_vals))
                avg_vals = approximate_curve(time_points, avg_vals, time_points, style.curve_approx_level)
            arr = avg_vals
        ax.plot(arr, label=name)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    apply_style(fig, title, style, eng=True)


@provide_plot
def plot_simple_bars(fig: Figure,
                     title: str,
                     names: List[str],
                     values: List[float],
                     errs: List[float] = None,
                     colors: ColorProfile = DefColorProfile,
                     style: StyleProfile = DefStyleProfile) -> None:

    ax = fig.add_subplot(111)
    ind = numpy.arange(len(names))
    width = 0.35
    ax.barh(ind, values, width, xerr=errs)

    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(val + err for val, err in zip(values, errs)) * 1.1)

    apply_style(fig, title, style, no_legend=True)
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    fig.subplots_adjust(left=0.2)


@provide_plot
def plot_hmap_from_2d(fig: Figure,
                      data2d: numpy.ndarray,
                      title: str, ylabel: str, xlabel: str = 'time, s', bins: numpy.ndarray = None,
                      colors: ColorProfile = DefColorProfile, style: StyleProfile = DefStyleProfile) -> None:
    fig.set_size_inches(*style.figsize_long)
    ioq1d, ranges = hmap_from_2d(data2d)
    ax, _ = plot_hmap_with_y_histo(fig, ioq1d, ranges, bins=bins, cmap=colors.hmap_cmap)
    ax.set(ylabel=ylabel, xlabel=xlabel)
    apply_style(fig, title, style, no_legend=True)


@provide_plot
def plot_v_over_time(fig: Figure,
                     title: str,
                     units: str,
                     ts: TimeSeries,
                     plot_avg_dev: bool = True,
                     plot_points: bool = True,
                     colors: ColorProfile = DefColorProfile,
                     style: StyleProfile = DefStyleProfile) -> None:

    min_time = min(ts.times)

    # convert time to ms
    coef = float(unit_conversion_coef(ts.time_units, 's'))
    time_points = numpy.array([(val_time - min_time) * coef for val_time in ts.times])

    outliers_idxs = find_ouliers_ts(ts.data, cut_range=style.outliers_q_nd)
    outliers_4q_idxs = find_ouliers_ts(ts.data, cut_range=style.outliers_hide_q_nd)
    normal_idxs = numpy.logical_not(outliers_idxs)
    outliers_idxs = outliers_idxs & numpy.logical_not(outliers_4q_idxs)
    # hidden_outliers_count = numpy.count_nonzero(outliers_4q_idxs)

    data = ts.data[normal_idxs]
    data_times = time_points[normal_idxs]
    outliers = ts.data[outliers_idxs]
    outliers_times = time_points[outliers_idxs]

    ax = fig.add_subplot(111)

    if plot_points:
        alpha = colors.noise_alpha if plot_avg_dev else 1.0
        ax.plot(data_times, data, style.point_shape,
                color=colors.primary_color, alpha=alpha, label="Data")
        ax.plot(outliers_times, outliers, style.err_point_shape,
                color=colors.err_color, label="Outliers")

    has_negative_dev = False
    plus_minus = "\xb1"

    if plot_avg_dev and len(data) < style.avg_range * 2:
        logger.warning("Array %r to small to plot average over %s points", title, style.avg_range)
    elif plot_avg_dev:
        avg_vals = moving_average(data, style.avg_range)
        dev_vals = moving_dev(data, style.avg_range)
        avg_times = moving_average(data_times, style.avg_range)

        if style.approx_average:
            avg_vals = approximate_curve(avg_times, avg_vals, avg_times, style.curve_approx_level)
            dev_vals = approximate_curve(avg_times, dev_vals, avg_times, style.curve_approx_level)

        ax.plot(avg_times, avg_vals, c=colors.suppl_color1, label="Average")

        low_vals_dev = avg_vals - dev_vals * style.dev_range_x
        hight_vals_dev = avg_vals + dev_vals * style.dev_range_x
        if style.dev_range_x - int(style.dev_range_x) < 0.01:
            ax.plot(avg_times, low_vals_dev, c=colors.suppl_color2,
                    label="{}{}*stdev".format(plus_minus, int(style.dev_range_x)))
        else:
            ax.plot(avg_times, low_vals_dev, c=colors.suppl_color2,
                    label="{}{}*stdev".format(plus_minus, style.dev_range_x))
        ax.plot(avg_times, hight_vals_dev, c=colors.suppl_color2)
        has_negative_dev = low_vals_dev.min() < 0

    ax.set_xlim(-5, max(time_points) + 5)
    ax.set_xlabel("Time, seconds from test begin")

    if plot_avg_dev:
        ax.set_ylabel("{}. Average and {}stddev over {} points".format(units, plus_minus, style.avg_range))
    else:
        ax.set_ylabel(units)

    if has_negative_dev:
        ax.set_ylim(bottom=0)

    apply_style(fig, title, style, eng=True)


@provide_plot
def plot_lat_over_time(fig: Figure,
                       title: str,
                       ts: TimeSeries,
                       ylabel: str,
                       samples: int = 5,
                       colors: ColorProfile = DefColorProfile, style: StyleProfile = DefStyleProfile) -> None:

    min_time = min(ts.times)
    times = [int(tm - min_time + 500) // 1000 for tm in ts.times]
    ts_len = len(times)
    step = ts_len / samples
    points = [times[int(i * step + 0.5)] for i in range(samples)]
    points.append(times[-1])
    bounds = list(zip(points[:-1], points[1:]))
    agg_data = []
    positions = []
    labels = []

    for begin, end in bounds:
        agg_hist = ts.data[begin:end].sum(axis=0)

        if style.violin_instead_of_box:
            # cut outliers
            idx1, idx2 = hist_outliers_perc(agg_hist, style.outliers_lat)
            agg_hist = agg_hist[idx1:idx2]
            curr_bins_vals = ts.histo_bins[idx1:idx2]

            correct_coef = style.violin_point_count / sum(agg_hist)
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

    ax = fig.add_subplot(111)
    if style.violin_instead_of_box:
        patches = ax.violinplot(agg_data,
                                positions=positions,
                                showmeans=True,
                                showmedians=True,
                                widths=step / 2)

        patches['cmeans'].set_color("blue")
        patches['cmedians'].set_color("green")
        if style.legend_for_eng:
            legend_location = "center left"
            legend_bbox_to_anchor = (1.03, 0.81)
            ax.legend([patches['cmeans'], patches['cmedians']], ["mean", "median"],
                      loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)
    else:
        ax.boxplot(agg_data, 0, '', positions=positions, labels=labels, widths=step / 4)

    ax.set_xlim(min(times), max(times))
    ax.set(ylabel=ylabel, xlabel="Time, seconds from test begin, sampled for ~{} seconds".format(int(step)))
    apply_style(fig, title, style, eng=True, no_legend=True)
    fig.subplots_adjust(right=style.subplot_adjust_r)


@provide_plot
def plot_histo_heatmap(fig: Figure,
                       title: str,
                       ts: TimeSeries,
                       ylabel: str,
                       xlabel: str = "time, s",
                       colors: ColorProfile = DefColorProfile, style: StyleProfile = DefStyleProfile) -> None:

    fig.set_size_inches(*style.figsize_long)

    # only histogram-based ts can be plotted
    assert len(ts.data.shape) == 2

    # Find global outliers. As load is expected to be stable during one job
    # outliers range can be detected globally
    total_hist = ts.data.sum(axis=0)
    idx1, idx2 = hist_outliers_perc(total_hist,
                                    bounds_perc=style.outliers_lat,
                                    min_bins_left=style.hm_hist_bins_count)

    # merge outliers with most close non-outliers cell
    orig_data = ts.data[:, idx1:idx2].copy()
    if idx1 > 0:
        orig_data[:, 0] += ts.data[:, :idx1].sum(axis=1)

    if idx2 < ts.data.shape[1]:
        orig_data[:, -1] += ts.data[:, idx2:].sum(axis=1)

    bins_vals = ts.histo_bins[idx1:idx2]

    # rebin over X axis
    # aggregate some lines in ts.data to plot not more than style.hm_x_slots x bins
    agg_idx = float(len(orig_data)) / style.hm_x_slots
    if agg_idx >= 2:
        data = numpy.zeros([style.hm_x_slots, orig_data.shape[1]], dtype=numpy.float32)  # type: List[numpy.ndarray]
        next = agg_idx
        count = 0
        data_idx = 0
        for idx, arr in enumerate(orig_data):
            if idx >= next:
                data[data_idx] /= count
                data_idx += 1
                next += agg_idx
                count = 0
            data[data_idx] += arr
            count += 1

        if count > 1:
            data[-1] /= count
    else:
        data = orig_data

    # rebin over Y axis
    # =================

    # don't using rebin_histogram here, as we need apply same bins for many arrays
    step = (bins_vals[-1] - bins_vals[0]) / style.hm_hist_bins_count
    new_bins_edges = numpy.arange(style.hm_hist_bins_count) * step + bins_vals[0]
    bin_mapping = numpy.clip(numpy.searchsorted(new_bins_edges, bins_vals) - 1, 0, len(new_bins_edges) - 1)

    # map origin bins ranges to heatmap bins, iterate over rows
    cmap = []
    for line in data:
        curr_bins = [0] * style.hm_hist_bins_count
        for idx, count in zip(bin_mapping, line):
            curr_bins[idx] += count
        cmap.append(curr_bins)
    ncmap = numpy.array(cmap)

    # plot data
    # =========

    boxes = 3
    gs = gridspec.GridSpec(1, boxes)
    ax = fig.add_subplot(gs[0, :boxes - 1])

    labels = list(map(float2str, (new_bins_edges[:-1] + new_bins_edges[1:]) / 2)) + \
        [float2str(new_bins_edges[-1]) + "+"]
    seaborn.heatmap(ncmap[:,::-1].T, xticklabels=False, cmap="Blues", ax=ax)
    ax.set_yticklabels(labels, rotation='horizontal')
    ax.set_xticklabels([])

    # plot overall histogram
    # =======================

    ax2 = fig.add_subplot(gs[0, boxes - 1])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    histo = ncmap.sum(axis=0).reshape((-1,))
    ax2.set_ylim(top=histo.size, bottom=0)
    ax2.barh(numpy.arange(histo.size) + 0.5, width=histo)

    ax.set(ylabel=ylabel, xlabel=xlabel)

    apply_style(fig, title, style, eng=True, no_legend=True)


@provide_plot
def io_chart(fig: Figure,
             title: str,
             legend: str,
             iosums: List[IOSummary],
             iops_log_spine: bool = False,
             lat_log_spine: bool = False,
             colors: ColorProfile = DefColorProfile, style: StyleProfile = DefStyleProfile) -> None:

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

    matplotlib.style.use(style.io_chart_style)

    block_size = iosums[0].block_size
    xpos = numpy.arange(1, len(iosums) + 1, dtype='uint')

    ax = fig.add_subplot(111)

    coef_mb = float(unit_conversion_coef(iosums[0].bw.units, "MiBps"))
    coef_iops = float(unit_conversion_coef(iosums[0].bw.units, "KiBps")) / block_size

    iops_primary = block_size < style.large_blocks

    coef = coef_iops if iops_primary else coef_mb
    ax.set_ylabel("IOPS" if iops_primary else "BW (MiBps)")

    vals = [iosum.bw.average * coef for iosum in iosums]

    # set correct x limits for primary IO spine
    min_io = min(iosum.bw.average - iosum.bw.deviation * style.dev_range_x for iosum in iosums)
    max_io = max(iosum.bw.average + iosum.bw.deviation * style.dev_range_x for iosum in iosums)
    border = (max_io - min_io) * extra_y_space
    io_lims = (min_io - border, max_io + border)

    ax.set_ylim(io_lims[0] * coef, io_lims[-1] * coef)
    ax.bar(xpos - width / 2, vals, width=width, color=colors.box_color, label=legend)

    # plot deviation and confidence error ranges
    err1_legend = err2_legend = None
    for pos, iosum in zip(xpos, iosums):
        dev_bar_pos = pos - err_x_offset
        err1_legend = ax.errorbar(dev_bar_pos,
                                  iosum.bw.average * coef,
                                  iosum.bw.deviation * style.dev_range_x * coef,
                                  alpha=colors.subinfo_alpha,
                                  color=colors.suppl_color1)  # 'magenta'

        conf_bar_pos = pos + err_x_offset
        err2_legend = ax.errorbar(conf_bar_pos,
                                  iosum.bw.average * coef,
                                  iosum.bw.confidence * coef,
                                  alpha=colors.subinfo_alpha,
                                  color=colors.suppl_color2)  # 'teal'

    if style.grid:
        ax.grid(True)

    handles1, labels1 = ax.get_legend_handles_labels()

    handles1 += [err1_legend, err2_legend]
    labels1 += ["{}% dev".format(style.dev_perc),
                "{}% conf".format(int(100 * iosums[0].bw.confidence_level))]

    # extra y spine for latency on right side
    ax2 = ax.twinx()

    # plot median and 95 perc latency
    lat_coef_ms = float(unit_conversion_coef(iosums[0].lat.units, "ms"))
    ax2.plot(xpos, [iosum.lat.perc_50 * lat_coef_ms for iosum in iosums], label="lat med")
    ax2.plot(xpos, [iosum.lat.perc_95 * lat_coef_ms for iosum in iosums], label="lat 95%")

    for grid_line in ax2.get_ygridlines():
        grid_line.set_linestyle(":")

    # extra y spine for BW/IOPS on left side
    if style.extra_io_spine:
        ax3 = ax.twinx()
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
    ax.legend(handles1 + handles2, labels1 + labels2,
              loc=legend_location,
              bbox_to_anchor=legend_bbox_to_anchor)

    # limit and label x spine
    ax.set_xlim(extra_x_space, len(iosums) + extra_x_space)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["{0} * {1} = {2}".format(iosum.qd, iosum.nodes_count, iosum.qd * iosum.nodes_count)
                       for iosum in iosums])
    ax.set_xlabel("IO queue depth * test node count = total parallel requests")

    # apply log scales for X spines, if set
    if iops_log_spine:
        ax.set_yscale('log')

    if lat_log_spine:
        ax2.set_yscale('log')

    # adjust central box size to fit legend
    apply_style(fig, title, style, eng=False, no_legend=True)

    # override some styles
    fig.set_size_inches(*style.qd_chart_inches)
    fig.subplots_adjust(right=StyleProfile.subplot_adjust_r)

    if style.extra_io_spine:
        ax3.grid(False)


#  --------------------  REPORT HELPERS --------------------------------------------------------------------------------


class HTMLBlock:
    data = None  # type: str
    js_links = []  # type: List[str]
    css_links = []  # type: List[str]
    order_attr = None  # type: Any

    def __init__(self, data: str, order_attr: Any = None) -> None:
        self.data = data
        self.order_attr = order_attr

    def __eq__(self, o: Any) -> bool:
        return o.order_attr == self.order_attr  # type: ignore

    def __lt__(self, o: Any) -> bool:
        return o.order_attr > self.order_attr  # type: ignore


class Table:
    def __init__(self, header: List[str]) -> None:
        self.header = header
        self.data = []

    def add_line(self, values: List[str]) -> None:
        self.data.append(values)

    def html(self):
        return html.table("", self.header, self.data)


class Menu1st:
    engineering = "Engineering"
    summary = "Summary"
    per_job = "Per Job"


class Menu2ndEng:
    iops_time = "IOPS(time)"
    hist = "IOPS/lat overall histogram"
    lat_time = "Lat(time)"


class Menu2ndSumm:
    io_lat_qd = "IO & Lat vs QD"


menu_1st_order = [Menu1st.summary, Menu1st.engineering, Menu1st.per_job]


#  --------------------  REPORTS  --------------------------------------------------------------------------------------


class Reporter(metaclass=abc.ABCMeta):
    suite_types = set() # type: Set[str]

    @abc.abstractmethod
    def get_divs(self, suite: SuiteConfig, storage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        pass


class JobReporter(metaclass=abc.ABCMeta):
    suite_type = set()  # type: Set[str]

    @abc.abstractmethod
    def get_divs(self,
                 suite: SuiteConfig,
                 job: JobConfig,
                 storage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        pass


# Main performance report
class PerformanceSummary(Reporter):
    """Aggregated summary fro storage"""


# Main performance report
class IO_QD(Reporter):
    """Creates graph, which show how IOPS and Latency depend on QD"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        ts_map = defaultdict(list)  # type: Dict[FioJobParams, List[Tuple[SuiteConfig, FioJobConfig]]]
        str_summary = {}  # type: Dict[FioJobParams, List[IOSummary]]
        for job in rstorage.iter_job(suite):
            fjob = cast(FioJobConfig, job)
            fjob_no_qd = cast(FioJobParams, fjob.params.copy(qd=None))
            str_summary[fjob_no_qd] = (fjob_no_qd.summary, fjob_no_qd.long_summary)
            ts_map[fjob_no_qd].append((suite, fjob))

        for tpl, suites_jobs in ts_map.items():
            if len(suites_jobs) >= StyleProfile.min_iops_vs_qd_jobs:

                iosums = [make_iosum(rstorage, suite, job) for suite, job in suites_jobs]
                iosums.sort(key=lambda x: x.qd)
                summary, summary_long = str_summary[tpl]

                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, \
                    HTMLBlock(html.H2(html.center("IOPS, BW, Lat = func(QD). " + summary_long)))

                ds = DataSource(suite_id=suite.storage_id,
                                job_id=summary,
                                node_id=AGG_TAG,
                                sensor="fio",
                                dev=AGG_TAG,
                                metric="io_over_qd",
                                tag=io_chart_format)

                fpath = io_chart(rstorage, ds, title="", legend="IOPS/BW", iosums=iosums)  # type: str
                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, HTMLBlock(html.center(html.img(fpath)))


# Linearization report
class IOPS_Bsize(Reporter):
    """Creates graphs, which show how IOPS and Latency depend on block size"""


class StatInfo(JobReporter):
    """Statistic info for job results"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:

        fjob = cast(FioJobConfig, job)
        io_sum = make_iosum(rstorage, suite, fjob)

        res = html.H2(html.center("Test summary - " + job.params.long_summary))
        stat_data_headers = ["Name", "Average ~ Dev", "Conf interval", "Mediana", "Mode", "Kurt / Skew", "95%", "99%",
                             "ADF test"]

        bw_target_units = 'Bps'
        bw_coef = float(unit_conversion_coef(io_sum.bw.units, bw_target_units))

        adf_v, *_1, stats, _2 = adfuller(io_sum.bw.data)

        for v in ("1%", "5%", "10%"):
            if adf_v <= stats[v]:
                ad_test = v
                break
        else:
            ad_test = "Failed"

        bw_data = ["Bandwidth",
                   "{}{} ~ {}{}".format(b2ssize(io_sum.bw.average * bw_coef), bw_target_units,
                                        b2ssize(io_sum.bw.deviation * bw_coef), bw_target_units),
                   b2ssize(io_sum.bw.confidence * bw_coef) + bw_target_units,
                   b2ssize(io_sum.bw.perc_50 * bw_coef) + bw_target_units,
                   "-",
                   "{:.2f} / {:.2f}".format(io_sum.bw.kurt, io_sum.bw.skew),
                   b2ssize(io_sum.bw.perc_5 * bw_coef) + bw_target_units,
                   b2ssize(io_sum.bw.perc_1 * bw_coef) + bw_target_units,
                   ad_test]

        iops_coef = float(unit_conversion_coef(io_sum.bw.units, 'KiBps')) / fjob.bsize
        iops_data = ["IOPS",
                     "{}IOPS ~ {}IOPS".format(b2ssize_10(io_sum.bw.average * iops_coef),
                                              b2ssize_10(io_sum.bw.deviation * iops_coef)),
                     b2ssize_10(io_sum.bw.confidence * iops_coef) + "IOPS",
                     b2ssize_10(io_sum.bw.perc_50 * iops_coef) + "IOPS",
                     "-",
                     "{:.2f} / {:.2f}".format(io_sum.bw.kurt, io_sum.bw.skew),
                     b2ssize_10(io_sum.bw.perc_5 * iops_coef) + "IOPS",
                     b2ssize_10(io_sum.bw.perc_1 * iops_coef) + "IOPS",
                     ad_test]

        lat_target_unit = 's'
        lat_coef = unit_conversion_coef(io_sum.lat.units, lat_target_unit)
        # latency
        lat_data = ["Latency",
                    "-",
                    "-",
                    b2ssize_10(io_sum.lat.perc_50 * lat_coef) + lat_target_unit,
                    "-",
                    "-",
                    b2ssize_10(io_sum.lat.perc_95 * lat_coef) + lat_target_unit,
                    b2ssize_10(io_sum.lat.perc_99 * lat_coef) + lat_target_unit,
                    '-']

        # sensor usage
        stat_data = [iops_data, bw_data, lat_data]
        res += html.center(html.table("Load stats info", stat_data_headers, stat_data))
        yield Menu1st.per_job, job.summary, HTMLBlock(res)


def avg_dev_div(vec: numpy.ndarray, denom: numpy.ndarray, avg_ranges: int = 10) -> Tuple[float, float]:
    step = min(vec.size, denom.size) // avg_ranges
    assert step >= 1
    vals = []

    whole_sum = denom.sum() / denom.size * step * 0.5
    for i in range(0, avg_ranges):
        s1 = denom[i * step: (i + 1) * step].sum()
        if s1 >= whole_sum:
            vals.append(vec[i * step: (i + 1) * step].sum() / s1)

    assert len(vals) > 1
    return vec.sum() / denom.sum(), numpy.std(vals, ddof=1)


class Resources(JobReporter):
    """Statistic info for job results"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:

        fjob = cast(FioJobConfig, job)
        io_sum = make_iosum(rstorage, suite, fjob)

        tot_io_coef = float(unit_conversion_coef(io_sum.bw.units, "Bps"))
        io_transfered = io_sum.bw.data * tot_io_coef
        ops_done = io_transfered / (fjob.bsize * float(unit_conversion_coef("KiBps", "Bps")))

        io_made = "Client IOP made"
        data_tr = "Client data transfered"

        records = {
            io_made: (b2ssize_10(ops_done.sum()) + "OP", None, None),
            data_tr: (b2ssize(io_transfered.sum()) + "B", None, None)
        }  # type: Dict[str, Tuple[str, float, float]]

        test_send = "Test nodes net send"
        test_recv = "Test nodes net recv"
        test_net = "Test nodes net total"
        test_send_pkt = "Test nodes send pkt"
        test_recv_pkt = "Test nodes recv pkt"
        test_net_pkt = "Test nodes total pkt"

        test_write = "Test nodes disk write"
        test_read = "Test nodes disk read"
        test_write_iop = "Test nodes write IOP"
        test_read_iop = "Test nodes read IOP"
        test_iop = "Test nodes IOP"
        test_rw = "Test nodes disk IO"

        storage_send = "Storage nodes net send"
        storage_recv = "Storage nodes net recv"
        storage_send_pkt = "Storage nodes send pkt"
        storage_recv_pkt = "Storage nodes recv pkt"
        storage_net = "Storage nodes net total"
        storage_net_pkt = "Storage nodes total pkt"

        storage_write = "Storage nodes disk write"
        storage_read = "Storage nodes disk read"
        storage_write_iop = "Storage nodes write IOP"
        storage_read_iop = "Storage nodes read IOP"
        storage_iop = "Storage nodes IOP"
        storage_rw = "Storage nodes disk IO"

        storage_cpu = "Storage nodes CPU"
        storage_cpu_s = "Storage nodes CPU s/IOP"
        storage_cpu_s_b = "Storage nodes CPU s/B"

        all_metrics = [
            (test_send, 'net-io', 'send_bytes', b2ssize, ['testnode'], "B", io_transfered),
            (test_recv, 'net-io', 'recv_bytes', b2ssize, ['testnode'], "B", io_transfered),
            (test_send_pkt, 'net-io', 'send_packets', b2ssize_10, ['testnode'], "pkt", ops_done),
            (test_recv_pkt, 'net-io', 'recv_packets', b2ssize_10, ['testnode'], "pkt", ops_done),

            (test_write, 'block-io', 'sectors_written', b2ssize, ['testnode'], "B", io_transfered),
            (test_read, 'block-io', 'sectors_read', b2ssize, ['testnode'], "B", io_transfered),
            (test_write_iop, 'block-io', 'writes_completed', b2ssize_10, ['testnode'], "OP", ops_done),
            (test_read_iop, 'block-io', 'reads_completed', b2ssize_10, ['testnode'], "OP", ops_done),

            (storage_send, 'net-io', 'send_bytes', b2ssize, STORAGE_ROLES, "B", io_transfered),
            (storage_recv, 'net-io', 'recv_bytes', b2ssize, STORAGE_ROLES, "B", io_transfered),
            (storage_send_pkt, 'net-io', 'send_packets', b2ssize_10, STORAGE_ROLES, "OP", ops_done),
            (storage_recv_pkt, 'net-io', 'recv_packets', b2ssize_10, STORAGE_ROLES, "OP", ops_done),

            (storage_write, 'block-io', 'sectors_written', b2ssize, STORAGE_ROLES, "B", io_transfered),
            (storage_read, 'block-io', 'sectors_read', b2ssize, STORAGE_ROLES, "B", io_transfered),
            (storage_write_iop, 'block-io', 'writes_completed', b2ssize_10, STORAGE_ROLES, "OP", ops_done),
            (storage_read_iop, 'block-io', 'reads_completed', b2ssize_10, STORAGE_ROLES, "OP", ops_done),
        ]

        all_agg = {}

        for vname, sensor, metric, ffunc, roles, units, service_provided_count in all_metrics:
            res_ts = summ_sensors(rstorage, roles, sensor=sensor, metric=metric, time_range=job.reliable_info_range_s)
            if res_ts is None:
                continue

            data = res_ts.data
            if units == "B":
                data = data * float(unit_conversion_coef(res_ts.units, "B"))

            records[vname] = (ffunc(data.sum()) + units, *avg_dev_div(data, service_provided_count))
            all_agg[vname] = data

        # cpu usage
        nodes_count = len(list(find_nodes_by_roles(rstorage, STORAGE_ROLES)))
        cpu_ts = get_cluster_cpu_load(rstorage, STORAGE_ROLES, job.reliable_info_range_s)

        cpus_used_sec = (1.0 - cpu_ts['idle'].data / cpu_ts['total'].data) * nodes_count
        used_s = b2ssize_10(cpus_used_sec.sum()) + 's'

        all_agg[storage_cpu] = cpus_used_sec
        records[storage_cpu_s] = (used_s, *avg_dev_div(cpus_used_sec, ops_done))
        records[storage_cpu_s_b] = (used_s, *avg_dev_div(cpus_used_sec, io_transfered))

        cums = [
            (test_iop, test_read_iop, test_write_iop, b2ssize_10, "OP", ops_done),
            (test_rw, test_read, test_write, b2ssize, "B", io_transfered),
            (test_net, test_send, test_recv, b2ssize, "B", io_transfered),
            (test_net_pkt, test_send_pkt, test_recv_pkt, b2ssize_10, "pkt", ops_done),

            (storage_iop, storage_read_iop, storage_write_iop, b2ssize_10, "OP", ops_done),
            (storage_rw, storage_read, storage_write, b2ssize, "B", io_transfered),
            (storage_net, storage_send, storage_recv, b2ssize, "B", io_transfered),
            (storage_net_pkt, storage_send_pkt, storage_recv_pkt, b2ssize_10, "pkt", ops_done),
        ]

        for vname, name1, name2, ffunc, units, service_provided_masked in cums:
            if name1 in all_agg and name2 in all_agg:
                agg = all_agg[name1] + all_agg[name2]
                records[vname] = (ffunc(agg.sum()) + units, *avg_dev_div(agg, service_provided_masked))

        table_structure = [
            "Service provided",
            (io_made, data_tr),
            "Test nodes total load",
            (test_send_pkt, test_send),
            (test_recv_pkt, test_recv),
            (test_net_pkt, test_net),
            (test_write_iop, test_write),
            (test_read_iop, test_read),
            (test_iop, test_rw),
            (test_iop, test_rw),
            "Storage nodes resource consumed",
            (storage_send_pkt, storage_send),
            (storage_recv_pkt, storage_recv),
            (storage_net_pkt, storage_net),
            (storage_write_iop, storage_write),
            (storage_read_iop, storage_read),
            (storage_iop, storage_rw),
            (storage_cpu_s, storage_cpu_s_b),
        ]  # type: List[Union[str, Tuple[Optional[str], Optional[str]]]

        yield Menu1st.per_job, job.summary, HTMLBlock(html.H2(html.center("Resources usage")))

        doc = xmlbuilder3.XMLBuilder("table",
                                     **{"class": "table table-bordered table-striped table-condensed table-hover",
                                        "style": "width: auto;"})

        with doc.thead:
            with doc.tr:
                [doc.th(header) for header in ["Resource", "Usage count", "To service"] * 2]

        cols = 6

        short_name = {
            name: (name if name in {io_made, data_tr} else " ".join(name.split()[2:]).capitalize())
            for name in records.keys()
        }

        short_name[storage_cpu_s] = "CPU (s/IOP)"
        short_name[storage_cpu_s_b] = "CPU (s/B)"

        with doc.tbody:
            with doc.tr:
                doc.td(colspan=str(cols // 2)).center.b("Operations")
                doc.td(colspan=str(cols // 2)).center.b("Bytes")

            for line in table_structure:
                with doc.tr:
                    if isinstance(line, str):
                        with doc.td(colspan=str(cols)):
                            doc.center.b(line)
                    else:
                        for name in line:
                            if name is None:
                                doc.td("-", colspan=str(cols // 2))
                                continue

                            amount_s, avg, dev = records[name]

                            if name in (storage_cpu_s, storage_cpu_s_b) and avg is not None:
                                dev_s = str(int(dev * 100 / avg)) + "%" if avg > 1E-9 else b2ssize_10(dev) + 's'
                                rel_val_s = "{}s ~ {}".format(b2ssize_10(avg), dev_s)
                            else:
                                if avg is None:
                                    rel_val_s = '-'
                                else:
                                    avg_s = int(avg) if avg > 10 else '{:.1f}'.format(avg)
                                    if avg > 1E-5:
                                        dev_s = str(int(dev * 100 / avg)) + "%"
                                    else:
                                        dev_s = int(dev) if dev > 10 else '{:.1f}'.format(dev)
                                    rel_val_s = "{} ~ {}".format(avg_s, dev_s)

                            doc.td(short_name[name], align="left")
                            doc.td(amount_s, align="right")

                            if avg is None or avg < 0.9:
                                doc.td(rel_val_s, align="right")
                            elif avg < 2.0:
                                doc.td(align="right").font(rel_val_s, color='green')
                            elif avg < 5.0:
                                doc.td(align="right").font(rel_val_s, color='orange')
                            else:
                                doc.td(align="right").font(rel_val_s, color='red')

        res = xmlbuilder3.tostr(doc).split("\n", 1)[1]
        yield Menu1st.per_job, job.summary, HTMLBlock(html.center(res))

        iop_names = [test_write_iop, test_read_iop, test_iop,
                     storage_write_iop, storage_read_iop, storage_iop]

        bytes_names = [test_write, test_read, test_rw,
                       test_send, test_recv, test_net,
                       storage_write, storage_read, storage_rw,
                       storage_send, storage_recv, storage_net]

        net_pkt_names = [test_send_pkt, test_recv_pkt, test_net_pkt,
                         storage_send_pkt, storage_recv_pkt, storage_net_pkt]

        for tp, names in [('iop', iop_names), ("bytes", bytes_names), ('Net packets per IOP', net_pkt_names)]:
            vals = []
            devs = []
            avail_names = []
            for name in names:
                if name in records:
                    avail_names.append(name)
                    _, avg, dev = records[name]
                    vals.append(avg)
                    devs.append(dev)

            # synchronously sort values and names, values is a key
            vals, names, devs = map(list, zip(*sorted(zip(vals, names, devs))))

            ds = DataSource(suite_id=suite.storage_id,
                            job_id=job.storage_id,
                            node_id=AGG_TAG,
                            sensor='resources',
                            dev=AGG_TAG,
                            metric=tp.replace(' ', "_") + '2service_bar',
                            tag=default_format)

            fname = plot_simple_bars(rstorage, ds,
                                     "Resource consuption / service provided, " + tp,
                                     [name.replace(" nodes", "") for name in names],
                                     vals, devs)

            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fname))


class BottleNeck(JobReporter):
    """Statistic info for job results"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, job: JobConfig, rstorage: ResultStorage) -> \
            Iterator[Tuple[str, str, HTMLBlock]]:

        nodes = list(find_nodes_by_roles(rstorage, STORAGE_ROLES))

        sensor = 'block-io'
        metric = 'io_queue'
        bn_val = 16

        for node in nodes:
            bn = 0
            tot = 0
            for _, ds in rstorage.iter_sensors(node_id=node.node_id, sensor=sensor, metric=metric):
                if ds.dev in ('sdb', 'sdc', 'sdd', 'sde'):
                    data = rstorage.load_sensor(ds)
                    p1 = job.reliable_info_range_s[0] * unit_conversion_coef('s', data.time_units)
                    p2 = job.reliable_info_range_s[1] * unit_conversion_coef('s', data.time_units)
                    idx1, idx2 = numpy.searchsorted(data.times, (p1, p2))
                    bn += (data.data[idx1: idx2] > bn_val).sum()
                    tot += idx2 - idx1
            print(node, bn, tot)

        yield Menu1st.per_job, job.summary, HTMLBlock("")


# CPU load
class CPULoadPlot(JobReporter):
    def get_divs(self,
                 suite: SuiteConfig,
                 job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:

        # plot CPU time
        for rt, roles in [('storage', STORAGE_ROLES), ('test', ['testnode'])]:
            cpu_ts = get_cluster_cpu_load(rstorage, roles, job.reliable_info_range_s)
            tss = [(name, ts.data * 100 / cpu_ts['total'].data)
                   for name, ts in cpu_ts.items()
                   if name in {'user', 'sys', 'irq', 'idle'}]
            fname = plot_simple_over_time(rstorage,
                                          cpu_ts['idle'].source(job_id=job.storage_id,
                                                                suite_id=suite.storage_id,
                                                                metric='allcpu', tag=rt + '.plt.' + default_format),
                                          tss=tss,
                                          average=True,
                                          ylabel="CPU time %",
                                          title="{} nodes CPU usage".format(rt.capitalize()))

            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fname))


# IO time and QD
class QDIOTimeHeatmap(JobReporter):
    def get_divs(self,
                 suite: SuiteConfig,
                 job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:

        # TODO: fix this hardcode, need to track what devices are actually used on test and storage nodes
        # use saved storage info in nodes

        journal_devs = None
        storage_devs = None
        test_nodes_devs = ['rbd0']

        for node in find_nodes_by_roles(rstorage, STORAGE_ROLES):
            cjd = set(node.params['ceph_journal_devs'])
            if journal_devs is None:
                journal_devs = cjd
            else:
                assert journal_devs == cjd, "{!r} != {!r}".format(journal_devs, cjd)

            csd = set(node.params['ceph_storage_devs'])
            if storage_devs is None:
                storage_devs = csd
            else:
                assert storage_devs == csd, "{!r} != {!r}".format(storage_devs, csd)

        trange = (job.reliable_info_range[0] // 1000, job.reliable_info_range[1] // 1000)

        for name, devs, roles in [('storage', storage_devs, STORAGE_ROLES),
                                  ('journal', journal_devs, STORAGE_ROLES),
                                  ('test', test_nodes_devs, ['testnode'])]:

            yield Menu1st.per_job, job.summary, \
                HTMLBlock(html.H2(html.center("{} IO heatmaps".format(name.capitalize()))))

            # QD heatmap
            ioq2d = find_sensors_to_2d(rstorage, roles, sensor='block-io', devs=devs,
                                       metric='io_queue', time_range=trange)

            ds = DataSource(suite.storage_id, job.storage_id, AGG_TAG, 'block-io', name, tag="hmap." + default_format)

            fname = plot_hmap_from_2d(rstorage,
                                      ds(metric='io_queue'),
                                      ioq2d,
                                      ylabel="IO QD",
                                      title=name.capitalize() + " devs QD",
                                      xlabel='Time',
                                      bins=StyleProfile.qd_bins)  # type: str
            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fname))

            # Block size heatmap
            wc2d = find_sensors_to_2d(rstorage, roles, sensor='block-io', devs=devs,
                                      metric='writes_completed', time_range=trange)
            wc2d[wc2d < 1E-3] = 1
            sw2d = find_sensors_to_2d(rstorage, roles, sensor='block-io', devs=devs,
                                      metric='sectors_written', time_range=trange)
            data2d = sw2d / wc2d / 1024
            fname = plot_hmap_from_2d(rstorage,
                                      ds(metric='wr_block_size'),
                                      data2d,
                                      ylabel="IO bsize, KiB",
                                      title=name.capitalize() + " write block size",
                                      xlabel='Time',
                                      bins=StyleProfile.block_size_bins)  # type: str
            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fname))

            # iotime heatmap
            wtime2d = find_sensors_to_2d(rstorage, roles, sensor='block-io', devs=devs,
                                         metric='io_time', time_range=trange)
            fname = plot_hmap_from_2d(rstorage,
                                      ds(metric='io_time'),
                                      wtime2d,
                                      ylabel="IO time (ms) per second",
                                      title=name.capitalize() + " iotime",
                                      xlabel='Time',
                                      bins=StyleProfile.iotime_bins)  # type: str
            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fname))


# IOPS/latency over test time for each job
class LoadToolResults(JobReporter):
    """IOPS/latency during test"""
    suite_types = {'fio'}

    def get_divs(self,
                 suite: SuiteConfig,
                 job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:

        fjob = cast(FioJobConfig, job)

        yield Menu1st.per_job, job.summary, HTMLBlock(html.H2(html.center("Load tool results")))

        agg_io = get_aggregated(rstorage, suite, fjob, "bw")
        if fjob.bsize >= DefStyleProfile.large_blocks:
            title = "Fio measured Bandwidth over time"
            units = "MiBps"
            agg_io.data //= int(unit_conversion_coef(units, agg_io.units))
        else:
            title = "Fio measured IOPS over time"
            agg_io.data //= (int(unit_conversion_coef("KiBps", agg_io.units)) * fjob.bsize)
            units = "IOPS"

        fpath = plot_v_over_time(rstorage, agg_io.source(tag='ts.' + default_format), title, units, agg_io)  # type: str
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        agg_lat = get_aggregated(rstorage, suite, fjob, "lat").copy()
        TARGET_UNITS = 'ms'
        coef = unit_conversion_coef(agg_lat.units, TARGET_UNITS)
        agg_lat.histo_bins = agg_lat.histo_bins.copy() * float(coef)
        agg_lat.units = TARGET_UNITS

        fpath = plot_lat_over_time(rstorage, agg_lat.source(tag='ts.' + default_format), "Latency",
                                   agg_lat, ylabel="Latency, " + agg_lat.units)  # type: str
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        fpath = plot_histo_heatmap(rstorage,
                                   agg_lat.source(tag='hmap.' + default_format),
                                   "Latency heatmap",
                                   agg_lat,
                                   ylabel="Latency, " + agg_lat.units,
                                   xlabel='Test time')  # type: str

        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        fjob = cast(FioJobConfig, job)

        # agg_lat = get_aggregated(rstorage, suite, fjob, "lat")
        # # bins_edges = numpy.array(get_lat_vals(agg_lat.data.shape[1]), dtype='float32') / 1000  # convert us to ms
        # lat_stat_prop = calc_histo_stat_props(agg_lat, bins_edges=None, rebins_count=StyleProfile.hist_lat_boxes)
        #
        # long_summary = cast(FioJobParams, fjob.params).long_summary
        #
        # title = "Latency distribution"
        # units = "ms"
        #
        # fpath = plot_hist(rstorage, agg_lat.source(tag='hist.svg'), title, units, lat_stat_prop)  # type: str
        # yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        agg_io = get_aggregated(rstorage, suite, fjob, "bw")

        if fjob.bsize >= DefStyleProfile.large_blocks:
            title = "BW distribution"
            units = "MiBps"
            agg_io.data //= int(unit_conversion_coef(units, agg_io.units))
        else:
            title = "IOPS distribution"
            agg_io.data //= (int(unit_conversion_coef("KiBps", agg_io.units)) * fjob.bsize)
            units = "IOPS"

        io_stat_prop = calc_norm_stat_props(agg_io, bins_count=StyleProfile.hist_boxes)
        fpath = plot_hist(rstorage, agg_io.source(tag='hist.' + default_format),
                          title, units, io_stat_prop)  # type: str
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))


# Cluster load over test time
class ClusterLoad(JobReporter):
    """IOPS/latency during test"""

    # TODO: units should came from sensor
    storage_sensors = [
        ('block-io', 'reads_completed', "Read", 'iop'),
        ('block-io', 'writes_completed', "Write", 'iop'),
        ('block-io', 'sectors_read', "Read", 'MiB'),
        ('block-io', 'sectors_written', "Write", 'MiB'),
    ]

    def get_divs(self,
                 suite: SuiteConfig,
                 job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        yield Menu1st.per_job, job.summary, HTMLBlock(html.H2(html.center("Cluster load")))

        for sensor, metric, op, units in self.storage_sensors:
            ts = summ_sensors(rstorage, STORAGE_ROLES, sensor, metric, job.reliable_info_range_s)
            ds = DataSource(suite_id=suite.storage_id,
                            job_id=job.storage_id,
                            node_id="storage",
                            sensor=sensor,
                            dev=AGG_TAG,
                            metric=metric,
                            tag="ts." + default_format)

            data = ts.data if units != 'MiB' else ts.data * float(unit_conversion_coef(ts.units, 'MiB'))
            ts = TimeSeries(name="",
                            times=numpy.arange(*job.reliable_info_range_s),
                            data=data,
                            raw=None,
                            units=units if ts.units is None else ts.units,
                            time_units=ts.time_units,
                            source=ds,
                            histo_bins=ts.histo_bins)

            sensor_title = "{} {}".format(op, units)
            fpath = plot_v_over_time(rstorage, ds, sensor_title, units, ts=ts)  # type: str
            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fpath))



# Node load over test time
class NodeLoad(Reporter):
    """IOPS/latency during test"""


# Ceph cluster summary
class CephClusterSummary(Reporter):
    """IOPS/latency during test"""


# TODO: Ceph operation breakout report
# TODO: Resource consumption for different type of test


# ------------------------------------------  REPORT STAGES  -----------------------------------------------------------


class HtmlReportStage(Stage):
    priority = StepOrder.REPORT

    def run(self, ctx: TestRun) -> None:
        rstorage = ResultStorage(ctx.storage)

        job_reporters = [StatInfo(), Resources(), LoadToolResults(), ClusterLoad(), CPULoadPlot(),
                         QDIOTimeHeatmap()] # type: List[JobReporter]
        # job_reporters = [QDIOTimeHeatmap()] # type: List[JobReporter]
        # job_reporters = []
        reporters = [IO_QD()]  # type: List[Reporter]
        # reporters = []  # type: List[Reporter]

        root_dir = os.path.dirname(os.path.dirname(wally.__file__))
        doc_templ_path = os.path.join(root_dir, "report_templates/index.html")
        report_template = open(doc_templ_path, "rt").read()
        css_file_src = os.path.join(root_dir, "report_templates/main.css")
        css_file = open(css_file_src, "rt").read()

        menu_block = []
        content_block = []
        link_idx = 0

        # matplotlib.rcParams.update(ctx.config.reporting.matplotlib_params.raw())
        # ColorProfile.__dict__.update(ctx.config.reporting.colors.raw())
        # StyleProfile.__dict__.update(ctx.config.reporting.style.raw())

        items = defaultdict(lambda: defaultdict(list))  # type: Dict[str, Dict[str, List[HTMLBlock]]]
        DEBUG = False
        # TODO: filter reporters
        for suite in rstorage.iter_suite(FioTest.name):
            all_jobs = list(rstorage.iter_job(suite))
            all_jobs.sort(key=lambda job: job.params)
            for job in all_jobs:
                if 'rwd16384_qd1' == job.summary:
                    try:
                        for reporter in job_reporters:
                            logger.debug("Start reporter %s on job %s suite %s",
                                         reporter.__class__.__name__, job.summary, suite.test_type)
                            for block, item, html in reporter.get_divs(suite, job, rstorage):
                                items[block][item].append(html)
                        if DEBUG:
                            break
                    except Exception:
                        logger.exception("Failed to generate report for %s", job)

            for reporter in reporters:
                try:
                    logger.debug("Start reporter %s on suite %s", reporter.__class__.__name__, suite.test_type)
                    for block, item, html in reporter.get_divs(suite, rstorage):
                        items[block][item].append(html)
                except Exception as exc:
                    logger.exception("Failed to generate report")

            if DEBUG:
                break

        logger.debug("Generating result html")

        for idx_1st, menu_1st in enumerate(sorted(items, key=lambda x: menu_1st_order.index(x))):
            menu_block.append(
                '<a href="#item{}" class="nav-group" data-toggle="collapse" data-parent="#MainMenu">{}</a>'
                .format(idx_1st, menu_1st)
            )
            menu_block.append('<div class="collapse" id="item{}">'.format(idx_1st))
            for menu_2nd in sorted(items[menu_1st]):
                menu_block.append('    <a href="#content{}" class="nav-group-item">{}</a>'
                                  .format(link_idx, menu_2nd))
                content_block.append('<div id="content{}">'.format(link_idx))
                content_block.extend("    " + x.data for x in items[menu_1st][menu_2nd])
                content_block.append('</div>')
                link_idx += 1
            menu_block.append('</div>')

        report = report_template.replace("{{{menu}}}", ("\n" + " " * 16).join(menu_block))
        report = report.replace("{{{content}}}", ("\n" + " " * 16).join(content_block))
        report_path = rstorage.put_report(report, "index.html")
        rstorage.put_report(css_file, "main.css")
        logger.info("Report is stored into %r", report_path)
