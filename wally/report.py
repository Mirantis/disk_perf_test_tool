import os
import abc
import logging
from io import BytesIO
from functools import wraps
from typing import Dict, Any, Iterator, Tuple, cast, List, Callable, Set, Optional
from collections import defaultdict

import numpy
import scipy.stats
import matplotlib.pyplot as plt

import wally

from . import html
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .hlstorage import ResultStorage
from .node_interfaces import NodeInfo
from .utils import b2ssize, b2ssize_10, STORAGE_ROLES
from .statistic import (calc_norm_stat_props, calc_histo_stat_props, moving_average, moving_dev,
                        hist_outliers_perc, ts_hist_outliers_perc, find_ouliers_ts, approximate_curve,
                        rebin_histogram)
from .result_classes import (StatProps, DataSource, TimeSeries, NormStatProps, HistoStatProps, SuiteConfig,
                             IResultStorage)
from .suits.io.fio_hist import get_lat_vals, expected_lat_bins
from .suits.io.fio import FioTest, FioJobConfig
from .suits.io.fio_job import FioJobParams
from .suits.job import JobConfig


logger = logging.getLogger("wally")


# ----------------  CONSTS ---------------------------------------------------------------------------------------------


DEBUG = False
LARGE_BLOCKS = 256
MiB2KiB = 1024
MS2S = 1000


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


class StyleProfile:
    grid = True
    tide_layout = True
    hist_boxes = 10
    hist_lat_boxes = 25
    hm_hist_bins_count = 25
    min_points_for_dev = 5

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
    figsize = (10, 6)

    extra_io_spine = True

    legend_for_eng = True
    heatmap_interpolation = '1d'
    heatmap_interpolation_points = 300
    outliers_q_nd = 3.0
    outliers_hide_q_nd = 4.0
    outliers_lat = (0.01, 0.995)

    violin_instead_of_box = True
    violin_point_count = 30000

    heatmap_colorbar = False

    min_iops_vs_qd_jobs = 3

    units = {
        'bw': ("MiBps", MiB2KiB, "bandwith"),
        'iops': ("IOPS", 1, "iops"),
        'lat': ("ms", 1, "latency")
    }


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

def make_iosum(rstorage: ResultStorage, suite: SuiteConfig, job: FioJobConfig) -> IOSummary:
    lat = get_aggregated(rstorage, suite, job, "lat")
    bins_edges = numpy.array(get_lat_vals(lat.data.shape[1]), dtype='float32') / 1000
    io = get_aggregated(rstorage, suite, job, "bw")

    return IOSummary(job.qd,
                     nodes_count=len(suite.nodes_ids),
                     block_size=job.bsize,
                     lat=calc_histo_stat_props(lat, bins_edges, StyleProfile.hist_boxes),
                     bw=calc_norm_stat_props(io, StyleProfile.hist_boxes))

#
# def iter_io_results(rstorage: ResultStorage,
#                     qds: List[int] = None,
#                     op_types: List[str] = None,
#                     sync_types: List[str] = None,
#                     block_sizes: List[int] = None) -> Iterator[Tuple[TestSuiteConfig, FioJobConfig]]:
#
#     for suite in rstorage.iter_suite(FioTest.name):
#         for job in rstorage.iter_job(suite):
#             fjob = cast(FioJobConfig, job)
#             assert int(fjob.vals['numjobs']) == 1
#
#             if sync_types is not None and fjob.sync_mode in sync_types:
#                 continue
#
#             if block_sizes is not None and fjob.bsize not in block_sizes:
#                 continue
#
#             if op_types is not None and fjob.op_type not in op_types:
#                 continue
#
#             if qds is not None and fjob.qd not in qds:
#                 continue
#
#             yield suite, fjob


AGG_TAG = 'ALL'


def get_aggregated(rstorage: ResultStorage, suite: SuiteConfig, job: FioJobConfig, metric: str) -> TimeSeries:
    tss = list(rstorage.iter_ts(suite, job, sensor=metric))
    ds = DataSource(suite_id=suite.storage_id,
                    job_id=job.storage_id,
                    node_id=AGG_TAG,
                    sensor='fio',
                    dev=AGG_TAG,
                    metric=metric,
                    tag='csv')

    agg_ts = TimeSeries(metric,
                        raw=None,
                        source=ds,
                        data=numpy.zeros(tss[0].data.shape, dtype=tss[0].data.dtype),
                        times=tss[0].times.copy(),
                        units=tss[0].units)

    for ts in tss:
        if metric == 'lat' and (len(ts.data.shape) != 2 or ts.data.shape[1] != expected_lat_bins):
            logger.error("Sensor %s.%s on node %s has" +
                         "shape=%s. Can only process sensors with shape=[X, %s].",
                         ts.source.dev, ts.source.sensor, ts.source.node_id,
                         ts.data.shape, expected_lat_bins)
            continue

        if metric != 'lat' and len(ts.data.shape) != 1:
            logger.error("Sensor %s.%s on node %s has" +
                         "shape=%s. Can only process 1D sensors.",
                         ts.source.dev, ts.source.sensor, ts.source.node_id, ts.data.shape)
            continue

        # TODO: match times on different ts
        agg_ts.data += ts.data

    return agg_ts


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


def get_sensor_for_time_range(storage: IResultStorage,
                              node_id: str,
                              sensor: str,
                              dev: str,
                              metric: str,
                              time_range: Tuple[int, int]) -> numpy.array:
    """Return sensor values for given node for given period. Return per second estimated values array

    Raise an error if required range is not full covered by data in storage.
    First it finds range of results from sensor, which fully covers requested range.
    ...."""

    ds = DataSource(node_id=node_id, sensor=sensor, dev=dev, metric=metric)
    sensor_data = storage.load_sensor(ds)
    assert sensor_data.time_units == 'us'

    # collected_at is array of pairs (collection_started_at, collection_finished_at)
    # extract start time from each pair
    collection_start_at = sensor_data.times[::2]  # type: numpy.array

    MICRO = 1000000

    # convert seconds to us
    begin = time_range[0] * MICRO
    end = time_range[1] * MICRO

    if begin < collection_start_at[0] or end > collection_start_at[-1] or end <= begin:
        raise AssertionError(("Incorrect data for get_sensor - time_range={!r}, collected_at=[{}, ..., {}]," +
                              "sensor = {}_{}.{}.{}").format(time_range,
                                                             sensor_data.times[0] // MICRO,
                                                             sensor_data.times[-1] // MICRO,
                                                             node_id, sensor, dev, metric))

    pos1, pos2 = numpy.searchsorted(collection_start_at, (begin, end))

    # current real data time chunk begin time
    edge_it = iter(collection_start_at[pos1 - 1: pos2 + 1])

    # current real data value
    val_it = iter(sensor_data.data[pos1 - 1: pos2 + 1])

    # result array, cumulative value per second
    result = numpy.zeros((end - begin) // MICRO)
    idx = 0
    curr_summ = 0

    # end of current time slot
    results_cell_ends = begin + MICRO

    # hack to unify looping
    real_data_end = next(edge_it)
    while results_cell_ends <= end:
        real_data_start = real_data_end
        real_data_end = next(edge_it)
        real_val_left = next(val_it)

        # real data "speed" for interval [real_data_start, real_data_end]
        real_val_ps = float(real_val_left) / (real_data_end - real_data_start)

        while real_data_end >= results_cell_ends and results_cell_ends <= end:
            # part of current real value, which is fit into current result cell
            curr_real_chunk = int((results_cell_ends - real_data_start) * real_val_ps)

            # calculate rest of real data for next result cell
            real_val_left -= curr_real_chunk
            result[idx] = curr_summ + curr_real_chunk
            idx += 1
            curr_summ = 0

            # adjust real data start time
            real_data_start = results_cell_ends
            results_cell_ends += MICRO

        # don't lost any real data
        curr_summ += real_val_left

    return result


# --------------  PLOT HELPERS FUNCTIONS  ------------------------------------------------------------------------------

def get_emb_data_svg(plt: Any, format: str = 'svg') -> bytes:
    bio = BytesIO()
    if format in ('png', 'jpg'):
        plt.savefig(bio, format=format)
        return bio.getvalue()
    elif format == 'svg':
        plt.savefig(bio, format='svg')
        img_start = "<!-- Created with matplotlib (http://matplotlib.org/) -->"
        return bio.getvalue().decode("utf8").split(img_start, 1)[1].encode("utf8")


def provide_plot(func: Callable[..., None]) -> Callable[..., str]:
    @wraps(func)
    def closure1(storage: ResultStorage,
                 path: DataSource,
                 *args, **kwargs) -> str:
        fpath = storage.check_plot_file(path)
        if not fpath:
            format = path.tag.split(".")[-1]

            plt.figure(figsize=StyleProfile.figsize)
            plt.subplots_adjust(right=0.66)

            func(*args, **kwargs)
            fpath = storage.put_plot_file(get_emb_data_svg(plt, format=format), path)
            logger.debug("Plot %s saved to %r", path, fpath)
            plt.clf()
            plt.close('all')
        return fpath
    return closure1


def apply_style(style: StyleProfile, eng: bool = True, no_legend: bool = False) -> None:
    if style.grid:
        plt.grid(True)

    if (style.legend_for_eng or not eng) and not no_legend:
        legend_location = "center left"
        legend_bbox_to_anchor = (1.03, 0.81)
        plt.legend(loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)


# --------------  PLOT FUNCTIONS  --------------------------------------------------------------------------------------


@provide_plot
def plot_hist(title: str, units: str,
              prop: StatProps,
              colors: Any = ColorProfile,
              style: Any = StyleProfile) -> None:

    # TODO: unit should came from ts
    normed_bins = prop.bins_populations / prop.bins_populations.sum()
    bar_width = prop.bins_edges[1] - prop.bins_edges[0]
    plt.bar(prop.bins_edges, normed_bins, color=colors.box_color, width=bar_width, label="Real data")

    plt.xlabel(units)
    plt.ylabel("Value probability")
    plt.title(title)

    dist_plotted = False
    if isinstance(prop, NormStatProps):
        nprop = cast(NormStatProps, prop)
        stats = scipy.stats.norm(nprop.average, nprop.deviation)

        new_edges, step = numpy.linspace(prop.bins_edges[0], prop.bins_edges[-1],
                                         len(prop.bins_edges) * 10, retstep=True)

        ypoints = stats.cdf(new_edges) * 11
        ypoints = [next - prev for (next, prev) in zip(ypoints[1:], ypoints[:-1])]
        xpoints = (new_edges[1:] + new_edges[:-1]) / 2

        plt.plot(xpoints, ypoints, color=colors.primary_color, label="Expected from\nnormal\ndistribution")
        dist_plotted = True

    plt.gca().set_xlim(left=prop.bins_edges[0])
    if prop.log_bins:
        plt.xscale('log')

    apply_style(style, eng=True, no_legend=not dist_plotted)


@provide_plot
def plot_v_over_time(title: str, units: str,
                     ts: TimeSeries,
                     plot_avg_dev: bool = True,
                     colors: Any = ColorProfile, style: Any = StyleProfile) -> None:

    min_time = min(ts.times)

    # /1000 is us to ms conversion
    time_points = numpy.array([(val_time - min_time) / 1000 for val_time in ts.times])

    outliers_idxs = find_ouliers_ts(ts.data, cut_range=style.outliers_q_nd)
    outliers_4q_idxs = find_ouliers_ts(ts.data, cut_range=style.outliers_hide_q_nd)
    normal_idxs = numpy.logical_not(outliers_idxs)
    outliers_idxs = outliers_idxs & numpy.logical_not(outliers_4q_idxs)
    hidden_outliers_count = numpy.count_nonzero(outliers_4q_idxs)

    data = ts.data[normal_idxs]
    data_times = time_points[normal_idxs]
    outliers = ts.data[outliers_idxs]
    outliers_times = time_points[outliers_idxs]

    alpha = colors.noise_alpha if plot_avg_dev else 1.0
    plt.plot(data_times, data, style.point_shape,
             color=colors.primary_color, alpha=alpha, label="Data")
    plt.plot(outliers_times, outliers, style.err_point_shape,
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

        plt.plot(avg_times, avg_vals, c=colors.suppl_color1, label="Average")

        low_vals_dev = avg_vals - dev_vals * style.dev_range_x
        hight_vals_dev = avg_vals + dev_vals * style.dev_range_x
        if style.dev_range_x - int(style.dev_range_x) < 0.01:
            plt.plot(avg_times, low_vals_dev, c=colors.suppl_color2,
                     label="{}{}*stdev".format(plus_minus, int(style.dev_range_x)))
        else:
            plt.plot(avg_times, low_vals_dev, c=colors.suppl_color2,
                     label="{}{}*stdev".format(plus_minus, style.dev_range_x))
        plt.plot(avg_times, hight_vals_dev, c=colors.suppl_color2)
        has_negative_dev = low_vals_dev.min() < 0

    plt.xlim(-5, max(time_points) + 5)
    plt.xlabel("Time, seconds from test begin")
    plt.ylabel("{}. Average and {}stddev over {} points".format(units, plus_minus, style.avg_range))
    plt.title(title)

    if has_negative_dev:
        plt.gca().set_ylim(bottom=0)

    apply_style(style, eng=True)


@provide_plot
def plot_lat_over_time(title: str, ts: TimeSeries, bins_vals: List[int], samples: int = 5,
                       colors: Any = ColorProfile,
                       style: Any = StyleProfile) -> None:

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
            curr_bins_vals = bins_vals[idx1:idx2]

            correct_coef = style.violin_point_count / sum(agg_hist)
            if correct_coef > 1:
                correct_coef = 1
        else:
            curr_bins_vals = bins_vals
            correct_coef = 1

        vals = numpy.empty(shape=(numpy.sum(agg_hist),), dtype='float32')
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

    if style.violin_instead_of_box:
        patches = plt.violinplot(agg_data,
                                 positions=positions,
                                 showmeans=True,
                                 showmedians=True,
                                 widths=step / 2)

        patches['cmeans'].set_color("blue")
        patches['cmedians'].set_color("green")
        if style.legend_for_eng:
            legend_location = "center left"
            legend_bbox_to_anchor = (1.03, 0.81)
            plt.legend([patches['cmeans'], patches['cmedians']], ["mean", "median"],
                       loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)
    else:
        plt.boxplot(agg_data, 0, '', positions=positions, labels=labels, widths=step / 4)

    plt.xlim(min(times), max(times))
    plt.xlabel("Time, seconds from test begin, sampled for ~{} seconds".format(int(step)))
    plt.ylabel("Latency, ms")
    plt.title(title)
    apply_style(style, eng=True, no_legend=True)


@provide_plot
def plot_heatmap(title: str,
                 ts: TimeSeries,
                 bins_vals: List[int],
                 colors: Any = ColorProfile,
                 style: Any = StyleProfile) -> None:

    assert len(ts.data.shape) == 2
    assert ts.data.shape[1] == len(bins_vals)

    total_hist = ts.data.sum(axis=0)

    # idx1, idx2 = hist_outliers_perc(total_hist, style.outliers_lat)
    idx1, idx2 = ts_hist_outliers_perc(ts.data, bounds_perc=style.outliers_lat)

    # don't cut too many bins
    min_bins_left = style.hm_hist_bins_count
    if idx2 - idx1 < min_bins_left:
        missed = min_bins_left - (idx2 - idx1) // 2
        idx2 = min(len(total_hist), idx2 + missed)
        idx1 = max(0, idx1 - missed)

    data = ts.data[:, idx1:idx2]
    bins_vals = bins_vals[idx1:idx2]

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

    xmin = 0
    xmax = (ts.times[-1] - ts.times[0]) / 1000 + 1
    ymin = new_bins_edges[0]
    ymax = new_bins_edges[-1]

    fig, ax = plt.subplots(figsize=style.figsize)

    if style.heatmap_interpolation == '1d':
        interpolation = 'none'
        res = []
        for column in ncmap:
            new_x = numpy.linspace(0, len(column), style.heatmap_interpolation_points)
            old_x = numpy.arange(len(column)) + 0.5
            new_vals = numpy.interp(new_x, old_x, column)
            res.append(new_vals)
        ncmap = numpy.array(res)
    else:
        interpolation = style.heatmap_interpolation

    ax.imshow(ncmap[:,::-1].T,
              interpolation=interpolation,
              extent=(xmin, xmax, ymin, ymax),
              cmap=colors.imshow_colormap)

    ax.set_aspect((xmax - xmin) / (ymax - ymin) * (6 / 9))
    ax.set_ylabel("Latency, ms")
    ax.set_xlabel("Test time, s")

    plt.title(title)


@provide_plot
def io_chart(title: str,
             legend: str,
             iosums: List[IOSummary],
             iops_log_spine: bool = False,
             lat_log_spine: bool = False,
             colors: Any = ColorProfile,
             style: Any = StyleProfile) -> None:

    # --------------  MAGIC VALUES  ---------------------
    # IOPS bar width
    width = 0.35

    # offset from center of bar to deviation/confidence range indicator
    err_x_offset = 0.05

    # extra space on top and bottom, comparing to maximal tight layout
    extra_y_space = 0.05

    # additional spine for BW/IOPS on left side of plot
    extra_io_spine_x_offset = -0.1

    # extra space on left and right sides
    extra_x_space = 0.5

    # legend location settings
    legend_location = "center left"
    legend_bbox_to_anchor = (1.1, 0.81)

    # plot box size adjust (only plot, not spines and legend)
    plot_box_adjust = {'right': 0.66}
    # --------------  END OF MAGIC VALUES  ---------------------

    block_size = iosums[0].block_size
    lc = len(iosums)
    xt = list(range(1, lc + 1))

    # x coordinate of middle of the bars
    xpos = [i - width / 2 for i in xt]

    # import matplotlib.gridspec as gridspec
    # gs = gridspec.GridSpec(1, 3, width_ratios=[1, 4, 1])
    # p1 = plt.subplot(gs[1])

    fig, p1 = plt.subplots(figsize=StyleProfile.figsize)

    # plot IOPS/BW bars
    if block_size >= LARGE_BLOCKS:
        iops_primary = False
        coef = MiB2KiB
        p1.set_ylabel("BW (MiBps)")
    else:
        iops_primary = True
        coef = block_size
        p1.set_ylabel("IOPS")

    p1.bar(xpos, [iosum.bw.average / coef for iosum in iosums], width=width, color=colors.box_color, label=legend)

    # set correct x limits for primary IO spine
    min_io = min(iosum.bw.average - iosum.bw.deviation * style.dev_range_x for iosum in iosums)
    max_io = max(iosum.bw.average + iosum.bw.deviation * style.dev_range_x for iosum in iosums)
    border = (max_io - min_io) * extra_y_space
    io_lims = (min_io - border, max_io + border)

    p1.set_ylim(io_lims[0] / coef, io_lims[-1] / coef)

    # plot deviation and confidence error ranges
    err1_legend = err2_legend = None
    for pos, iosum in zip(xpos, iosums):
        err1_legend = p1.errorbar(pos + width / 2 - err_x_offset,
                                  iosum.bw.average / coef,
                                  iosum.bw.deviation * style.dev_range_x / coef,
                                  alpha=colors.subinfo_alpha,
                                  color=colors.suppl_color1)  # 'magenta'
        err2_legend = p1.errorbar(pos + width / 2 + err_x_offset,
                                  iosum.bw.average / coef,
                                  iosum.bw.confidence / coef,
                                  alpha=colors.subinfo_alpha,
                                  color=colors.suppl_color2)  # 'teal'

    if style.grid:
        p1.grid(True)

    handles1, labels1 = p1.get_legend_handles_labels()

    handles1 += [err1_legend, err2_legend]
    labels1 += ["{}% dev".format(style.dev_perc),
                "{}% conf".format(int(100 * iosums[0].bw.confidence_level))]

    # extra y spine for latency on right side
    p2 = p1.twinx()

    # plot median and 95 perc latency
    p2.plot(xt, [iosum.lat.perc_50 for iosum in iosums], label="lat med")
    p2.plot(xt, [iosum.lat.perc_95 for iosum in iosums], label="lat 95%")

    # limit and label x spine
    plt.xlim(extra_x_space, lc + extra_x_space)
    plt.xticks(xt, ["{0} * {1}".format(iosum.qd, iosum.nodes_count) for iosum in iosums])
    p1.set_xlabel("QD * Test node count")

    # apply log scales for X spines, if set
    if iops_log_spine:
        p1.set_yscale('log')

    if lat_log_spine:
        p2.set_yscale('log')

    # extra y spine for BW/IOPS on left side
    if style.extra_io_spine:
        p3 = p1.twinx()
        if iops_log_spine:
            p3.set_yscale('log')

        if iops_primary:
            p3.set_ylabel("BW (MiBps)")
            p3.set_ylim(io_lims[0] / MiB2KiB, io_lims[1] / MiB2KiB)
        else:
            p3.set_ylabel("IOPS")
            p3.set_ylim(io_lims[0] / block_size, io_lims[1] / block_size)

        p3.spines["left"].set_position(("axes", extra_io_spine_x_offset))
        p3.spines["left"].set_visible(True)
        p3.yaxis.set_label_position('left')
        p3.yaxis.set_ticks_position('left')

    p2.set_ylabel("Latency (ms)")

    plt.title(title)

    # legend box
    handles2, labels2 = p2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2,
               loc=legend_location,
               bbox_to_anchor=legend_bbox_to_anchor)

    # adjust central box size to fit legend
    plt.subplots_adjust(**plot_box_adjust)
    apply_style(style, eng=False, no_legend=True)


#  --------------------  REPORT HELPERS --------------------------------------------------------------------------------


class HTMLBlock:
    data = None  # type: str
    js_links = []  # type: List[str]
    css_links = []  # type: List[str]
    order_attr = None  # type: Any

    def __init__(self, data: str, order_attr: Any = None) -> None:
        self.data = data
        self.order_attr = order_attr

    def __eq__(self, o: object) -> bool:
        return o.order_attr == self.order_attr  # type: ignore

    def __lt__(self, o: object) -> bool:
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
            if len(suites_jobs) > StyleProfile.min_iops_vs_qd_jobs:
                iosums = [make_iosum(rstorage, suite, job) for suite, job in suites_jobs]
                iosums.sort(key=lambda x: x.qd)
                summary, summary_long = str_summary[tpl]
                ds = DataSource(suite_id=suite.storage_id,
                                job_id=summary,
                                node_id=AGG_TAG,
                                sensor="fio",
                                dev=AGG_TAG,
                                metric="io_over_qd",
                                tag="svg")

                title = "IOPS, BW, Lat vs. QD.\n" + summary_long
                fpath = io_chart(rstorage, ds, title=title, legend="IOPS/BW", iosums=iosums)  # type: str
                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, HTMLBlock(html.img(fpath))


# Linearization report
class IOPS_Bsize(Reporter):
    """Creates graphs, which show how IOPS and Latency depend on block size"""


def summ_sensors(rstorage: ResultStorage,
                 nodes: List[str],
                 sensor: str,
                 metric: str,
                 time_range: Tuple[int, int]) -> Optional[numpy.array]:

    res = None  # type: Optional[numpy.array]
    for node_id in nodes:
        for _, groups in rstorage.iter_sensors(node_id=node_id, sensor=sensor, metric=metric):
            data = get_sensor_for_time_range(rstorage,
                                             node_id=node_id,
                                             sensor=sensor,
                                             dev=groups['dev'],
                                             metric=metric,
                                             time_range=time_range)
            if res is None:
                res = data
            else:
                res += data
    return res


# IOPS/latency distribution
class StatInfo(JobReporter):
    """Statistic info for job results"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:

        fjob = cast(FioJobConfig, job)
        io_sum = make_iosum(rstorage, suite, fjob)

        summary_data = [
            ["Summary", job.params.long_summary],
        ]

        res = html.H2(html.center("Test summary"))
        res += html.table("Test info", None, summary_data)
        stat_data_headers = ["Name", "Average ~ Dev", "Conf interval", "Mediana", "Mode", "Kurt / Skew", "95%", "99%"]

        KB = 1024
        bw_data = ["Bandwidth",
                   "{}Bps ~ {}Bps".format(b2ssize(io_sum.bw.average * KB), b2ssize(io_sum.bw.deviation * KB)),
                   b2ssize(io_sum.bw.confidence * KB) + "Bps",
                   b2ssize(io_sum.bw.perc_50 * KB) + "Bps",
                   "-",
                   "{:.2f} / {:.2f}".format(io_sum.bw.kurt, io_sum.bw.skew),
                   b2ssize(io_sum.bw.perc_5 * KB) + "Bps",
                   b2ssize(io_sum.bw.perc_1 * KB) + "Bps"]

        iops_data = ["IOPS",
                     "{}IOPS ~ {}IOPS".format(b2ssize_10(io_sum.bw.average / fjob.bsize),
                                              b2ssize_10(io_sum.bw.deviation / fjob.bsize)),
                     b2ssize_10(io_sum.bw.confidence / fjob.bsize) + "IOPS",
                     b2ssize_10(io_sum.bw.perc_50 / fjob.bsize) + "IOPS",
                     "-",
                     "{:.2f} / {:.2f}".format(io_sum.bw.kurt, io_sum.bw.skew),
                     b2ssize_10(io_sum.bw.perc_5 / fjob.bsize) + "IOPS",
                     b2ssize_10(io_sum.bw.perc_1 / fjob.bsize) + "IOPS"]

        MICRO = 1000000
        # latency
        lat_data = ["Latency",
                    "-",
                    "-",
                    b2ssize_10(io_sum.bw.perc_50 / MICRO) + "s",
                    "-",
                    "-",
                    b2ssize_10(io_sum.bw.perc_95 / MICRO) + "s",
                    b2ssize_10(io_sum.bw.perc_99 / MICRO) + "s"]

        # sensor usage
        stat_data = [iops_data, bw_data, lat_data]
        res += html.table("Load stats info", stat_data_headers, stat_data)

        resource_headers = ["Resource", "Usage count", "Proportional to work done"]

        io_transfered = io_sum.bw.data.sum() * KB
        resource_data = [
            ["IO made", b2ssize_10(io_transfered / KB / fjob.bsize) + "OP", "-"],
            ["Data transfered", b2ssize(io_transfered) + "B", "-"]
        ]


        storage = rstorage.storage
        nodes = storage.load_list(NodeInfo, 'all_nodes')  # type: List[NodeInfo]

        storage_nodes = [node.node_id for node in nodes if node.roles.intersection(STORAGE_ROLES)]
        test_nodes = [node.node_id for node in nodes if "testnode" in node.roles]

        trange = [job.reliable_info_range[0] / 1000, job.reliable_info_range[1] / 1000]
        ops_done = io_transfered / fjob.bsize / KB

        all_metrics = [
            ("Test nodes net send", 'net-io', 'send_bytes', b2ssize, test_nodes, "B", io_transfered),
            ("Test nodes net recv", 'net-io', 'recv_bytes', b2ssize, test_nodes, "B", io_transfered),

            ("Test nodes disk write", 'block-io', 'sectors_written', b2ssize, test_nodes, "B", io_transfered),
            ("Test nodes disk read", 'block-io', 'sectors_read', b2ssize, test_nodes, "B", io_transfered),
            ("Test nodes writes", 'block-io', 'writes_completed', b2ssize_10, test_nodes, "OP", ops_done),
            ("Test nodes reads", 'block-io', 'reads_completed', b2ssize_10, test_nodes, "OP", ops_done),

            ("Storage nodes net send", 'net-io', 'send_bytes', b2ssize, storage_nodes, "B", io_transfered),
            ("Storage nodes net recv", 'net-io', 'recv_bytes', b2ssize, storage_nodes, "B", io_transfered),

            ("Storage nodes disk write", 'block-io', 'sectors_written', b2ssize, storage_nodes, "B", io_transfered),
            ("Storage nodes disk read", 'block-io', 'sectors_read', b2ssize, storage_nodes, "B", io_transfered),
            ("Storage nodes writes", 'block-io', 'writes_completed', b2ssize_10, storage_nodes, "OP", ops_done),
            ("Storage nodes reads", 'block-io', 'reads_completed', b2ssize_10, storage_nodes, "OP", ops_done),
        ]

        all_agg = {}

        for descr, sensor, metric, ffunc, nodes, units, denom in all_metrics:
            if not nodes:
                continue

            res_arr = summ_sensors(rstorage, nodes=nodes, sensor=sensor, metric=metric, time_range=trange)
            if res_arr is None:
                continue

            agg = res_arr.sum()
            resource_data.append([descr, ffunc(agg) + units, "{:.1f}".format(agg / denom)])
            all_agg[descr] = agg


        cums = [
            ("Test nodes writes", "Test nodes reads", "Total test ops", b2ssize_10, "OP", ops_done),
            ("Storage nodes writes", "Storage nodes reads", "Total storage ops", b2ssize_10, "OP", ops_done),
            ("Storage nodes disk write", "Storage nodes disk read", "Total storage IO size", b2ssize,
             "B", io_transfered),
            ("Test nodes disk write", "Test nodes disk read", "Total test nodes IO size", b2ssize, "B", io_transfered),
        ]

        for name1, name2, descr, ffunc, units, denom in cums:
            if name1 in all_agg and name2 in all_agg:
                agg = all_agg[name1] + all_agg[name2]
                resource_data.append([descr, ffunc(agg) + units, "{:.1f}".format(agg / denom)])

        res += html.table("Resources usage", resource_headers, resource_data)

        yield Menu1st.per_job, job.summary, HTMLBlock(res)


# IOPS/latency distribution
class IOHist(JobReporter):
    """IOPS.latency distribution histogram"""
    suite_types = {'fio'}

    def get_divs(self,
                 suite: SuiteConfig,
                 job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:

        fjob = cast(FioJobConfig, job)

        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.H2(html.center("Load histograms")))

        agg_lat = get_aggregated(rstorage, suite, fjob, "lat")
        bins_edges = numpy.array(get_lat_vals(agg_lat.data.shape[1]), dtype='float32') / 1000  # convert us to ms
        lat_stat_prop = calc_histo_stat_props(agg_lat, bins_edges, bins_count=StyleProfile.hist_lat_boxes)

        # import IPython
        # IPython.embed()

        long_summary = cast(FioJobParams, fjob.params).long_summary

        title = "Latency distribution"
        units = "ms"

        fpath = plot_hist(rstorage, agg_lat.source(tag='hist.svg'), title, units, lat_stat_prop)  # type: str
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        agg_io = get_aggregated(rstorage, suite, fjob, "bw")

        if fjob.bsize >= LARGE_BLOCKS:
            title = "BW distribution"
            units = "MiBps"
            agg_io.data //= MiB2KiB
        else:
            title = "IOPS distribution"
            agg_io.data //= fjob.bsize
            units = "IOPS"

        io_stat_prop = calc_norm_stat_props(agg_io, bins_count=StyleProfile.hist_boxes)
        fpath = plot_hist(rstorage, agg_io.source(tag='hist.svg'), title, units, io_stat_prop)  # type: str
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))


# IOPS/latency over test time for each job
class IOTime(JobReporter):
    """IOPS/latency during test"""
    suite_types = {'fio'}

    def get_divs(self,
                 suite: SuiteConfig,
                 job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:

        fjob = cast(FioJobConfig, job)

        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.H2(html.center("Load over time")))

        agg_io = get_aggregated(rstorage, suite, fjob, "bw")
        if fjob.bsize >= LARGE_BLOCKS:
            title = "Bandwidth"
            units = "MiBps"
            agg_io.data //= MiB2KiB
        else:
            title = "IOPS"
            agg_io.data //= fjob.bsize
            units = "IOPS"

        fpath = plot_v_over_time(rstorage, agg_io.source(tag='ts.svg'), title, units, agg_io)  # type: str
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        agg_lat = get_aggregated(rstorage, suite, fjob, "lat")
        bins_edges = numpy.array(get_lat_vals(agg_lat.data.shape[1]), dtype='float32') / 1000
        title = "Latency"

        fpath = plot_lat_over_time(rstorage, agg_lat.source(tag='ts.svg'), title, agg_lat, bins_edges)  # type: str
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        title = "Latency heatmap"
        fpath = plot_heatmap(rstorage, agg_lat.source(tag='hmap.png'), title, agg_lat, bins_edges)  # type: str

        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))


class ResourceUsage:
    def __init__(self, io_r_ops: int, io_w_ops: int, io_r_kb: int, io_w_kb: int) -> None:
        self.io_w_ops = io_w_ops
        self.io_r_ops = io_r_ops
        self.io_w_kb = io_w_kb
        self.io_r_kb = io_r_kb

        self.cpu_used_user = None  # type: int
        self.cpu_used_sys = None  # type: int
        self.cpu_wait_io = None  # type: int

        self.net_send_packets = None  # type: int
        self.net_recv_packets = None  # type: int
        self.net_send_kb = None  # type: int
        self.net_recv_kb = None  # type: int


# Cluster load over test time
class ClusterLoad(JobReporter):
    """IOPS/latency during test"""

    # TODO: units should came from sensor
    storage_sensors = [
        ('block-io', 'reads_completed', "Read ops", 'iops'),
        ('block-io', 'writes_completed', "Write ops", 'iops'),
        ('block-io', 'sectors_read', "Read kb", 'kb'),
        ('block-io', 'sectors_written', "Write kb", 'kb'),
    ]

    def get_divs(self,
                 suite: SuiteConfig,
                 job: JobConfig,
                 rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        # split nodes on test and other
        storage = rstorage.storage
        nodes = storage.load_list(NodeInfo, "all_nodes")  # type: List[NodeInfo]

        yield Menu1st.per_job, job.summary, HTMLBlock(html.H2(html.center("Cluster load")))
        test_nodes = {node.node_id for node in nodes if 'testnode' in node.roles}
        cluster_nodes = {node.node_id for node in nodes if 'testnode' not in node.roles}

        # convert ms to s
        time_range = (job.reliable_info_range[0] // MS2S, job.reliable_info_range[1] // MS2S)
        len = time_range[1] - time_range[0]
        for sensor, metric, sensor_title, units in self.storage_sensors:
            sum_testnode = numpy.zeros((len,))
            sum_other = numpy.zeros((len,))
            for path, groups in rstorage.iter_sensors(sensor=sensor, metric=metric):
                # todo: should return sensor units
                data = get_sensor_for_time_range(rstorage,
                                                 groups['node_id'],
                                                 sensor,
                                                 groups['dev'],
                                                 metric, time_range)
                if groups['node_id'] in test_nodes:
                    sum_testnode += data
                else:
                    sum_other += data

            ds = DataSource(suite_id=suite.storage_id,
                            job_id=job.storage_id,
                            node_id="test_nodes",
                            sensor=sensor,
                            dev=AGG_TAG,
                            metric=metric,
                            tag="ts.svg")

            # s to ms
            ts = TimeSeries(name="",
                            times=numpy.arange(*time_range) * MS2S,
                            data=sum_testnode,
                            raw=None,
                            units=units,
                            time_units="us",
                            source=ds)
            fpath = plot_v_over_time(rstorage, ds, sensor_title, sensor_title, ts=ts)  # type: str
            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fpath))


# Ceph cluster summary
class ResourceConsumption(Reporter):
    """Resources consumption report, only text"""


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

        job_reporters = [StatInfo(), IOTime(), IOHist(), ClusterLoad()] # type: List[JobReporter]
        reporters = [IO_QD()]  # type: List[Reporter]

        # job_reporters = [ClusterLoad()]
        # reporters = []

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

        # TODO: filter reporters
        for suite in rstorage.iter_suite(FioTest.name):
            all_jobs = list(rstorage.iter_job(suite))
            all_jobs.sort(key=lambda job: job.params)
            for job in all_jobs:
                for reporter in job_reporters:
                    for block, item, html in reporter.get_divs(suite, job, rstorage):
                        items[block][item].append(html)
                if DEBUG:
                    break

            for reporter in reporters:
                for block, item, html in reporter.get_divs(suite, rstorage):
                    items[block][item].append(html)

            if DEBUG:
                break

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


class ConsoleReportStage(Stage):

    priority = StepOrder.REPORT

    def run(self, ctx: TestRun) -> None:
        # TODO(koder): load data from storage
        raise NotImplementedError("...")
