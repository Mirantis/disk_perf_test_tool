import os
import re
import abc
import bisect
import logging
from io import BytesIO
from functools import wraps
from typing import Dict, Any, Iterator, Tuple, cast, List, Callable
from collections import defaultdict

import numpy
import matplotlib
# have to be before pyplot import to avoid tkinter(default graph frontend) import error
matplotlib.use('svg')
import matplotlib.pyplot as plt
import scipy.stats

import wally

from . import html
from .utils import b2ssize
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .hlstorage import ResultStorage
from .node_interfaces import NodeInfo
from .storage import Storage
from .statistic import calc_norm_stat_props, calc_histo_stat_props
from .result_classes import (StatProps, DataSource, TimeSeries, TestSuiteConfig,
                             NormStatProps, HistoStatProps, TestJobConfig)
from .suits.io.fio_hist import get_lat_vals, expected_lat_bins
from .suits.io.fio import FioTest, FioJobConfig
from .suits.io.fio_task_parser import FioTestSumm
from .statistic import approximate_curve, average, dev


logger = logging.getLogger("wally")


# ----------------  CONSTS ---------------------------------------------------------------------------------------------


DEBUG = False
LARGE_BLOCKS = 256
MiB2KiB = 1024
MS2S = 1000


# ----------------  PROFILES  ------------------------------------------------------------------------------------------


class ColorProfile:
    primary_color = 'b'
    suppl_color1 = 'teal'
    suppl_color2 = 'magenta'
    box_color = 'y'

    noise_alpha = 0.3
    subinfo_alpha = 0.7


class StyleProfile:
    grid = True
    tide_layout = True
    hist_boxes = 10
    min_points_for_dev = 5

    dev_range_x = 2.0
    dev_perc = 95

    avg_range = 20

    curve_approx_level = 5
    curve_approx_points = 100
    assert avg_range >= min_points_for_dev

    extra_io_spine = True

    legend_for_eng = True

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
rexpr = {
    'sensor': r'(?P<sensor>[-a-z]+)',
    'dev': r'(?P<dev>[^.]+)',
    'metric': r'(?P<metric>[a-z_]+)',
    'node': r'(?P<node>\d+\.\d+\.\d+\.\d+:\d+)',
}

def iter_sensors(storage: Storage, node: str = None, sensor: str = None, dev: str = None, metric: str = None):
    if node is None:
        node = rexpr['node']
    if sensor is None:
        sensor = rexpr['sensor']
    if dev is None:
        dev = rexpr['dev']
    if metric is None:
        metric = rexpr['metric']

    rr = r"{}_{}\.{}\.{}$".format(node, sensor, dev, metric)
    sensor_name_re = re.compile(rr)

    for is_file, sensor_data_name in storage.list("sensors"):
        if is_file:
            rr = sensor_name_re.match(sensor_data_name)
            if rr:
                yield 'sensors/' + sensor_data_name, rr.groupdict()


def make_iosum(rstorage: ResultStorage, suite: TestSuiteConfig, job: FioJobConfig) -> IOSummary:
    lat = get_aggregated(rstorage, suite, job, "lat")
    bins_edges = numpy.array(get_lat_vals(lat.second_axis_size), dtype='float32') / 1000
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


def get_aggregated(rstorage: ResultStorage, suite: TestSuiteConfig, job: FioJobConfig, sensor: str) -> TimeSeries:
    tss = list(rstorage.iter_ts(suite, job, sensor=sensor))
    ds = DataSource(suite_id=suite.storage_id,
                    job_id=job.storage_id,
                    node_id="__all__",
                    dev='fio',
                    sensor=sensor,
                    tag=None)

    agg_ts = TimeSeries(sensor,
                        raw=None,
                        source=ds,
                        data=numpy.zeros(tss[0].data.shape, dtype=tss[0].data.dtype),
                        times=tss[0].times.copy(),
                        second_axis_size=tss[0].second_axis_size)

    for ts in tss:
        if sensor == 'lat' and ts.second_axis_size != expected_lat_bins:
            logger.error("Sensor %s.%s on node %s has" +
                         "second_axis_size=%s. Can only process sensors with second_axis_size=%s.",
                         ts.source.dev, ts.source.sensor, ts.source.node_id,
                         ts.second_axis_size, expected_lat_bins)
            continue

        if sensor != 'lat' and ts.second_axis_size != 1:
            logger.error("Sensor %s.%s on node %s has" +
                         "second_axis_size=%s. Can only process sensors with second_axis_size=1.",
                         ts.source.dev, ts.source.sensor, ts.source.node_id, ts.second_axis_size)
            continue

        # TODO: match times on different ts
        agg_ts.data += ts.data

    return agg_ts


# --------------  PLOT HELPERS FUNCTIONS  ------------------------------------------------------------------------------

def get_emb_data_svg(plt: Any) -> bytes:
    bio = BytesIO()
    plt.savefig(bio, format='svg')
    img_start = "<!-- Created with matplotlib (http://matplotlib.org/) -->"
    return bio.getvalue().decode("utf8").split(img_start, 1)[1].encode("utf8")


def provide_plot(func: Callable[..., None]) -> Callable[..., str]:
    @wraps(func)
    def closure1(storage: ResultStorage, path: DataSource, *args, **kwargs) -> str:
        fpath = storage.check_plot_file(path)
        if not fpath:
            func(*args, **kwargs)
            fpath = storage.put_plot_file(get_emb_data_svg(plt), path)
            plt.clf()
            logger.debug("Save plot for %s to %r", path, fpath)
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
    total = sum(prop.bins_populations)
    mids = prop.bins_mids
    normed_bins = [population / total for population in prop.bins_populations]
    bar_width = mids[1] - mids[0]
    plt.bar(mids - bar_width / 2, normed_bins, color=colors.box_color, width=bar_width, label="Real data")

    plt.xlabel(units)
    plt.ylabel("Value probability")
    plt.title(title)

    dist_plotted = False
    if isinstance(prop, NormStatProps):
        nprop = cast(NormStatProps, prop)
        stats = scipy.stats.norm(nprop.average, nprop.deviation)

        # xpoints = numpy.linspace(mids[0], mids[-1], style.curve_approx_points)
        # ypoints = stats.pdf(xpoints) / style.curve_approx_points

        edges, step = numpy.linspace(mids[0], mids[-1], len(mids) * 10, retstep=True)

        ypoints = stats.cdf(edges) * 11
        ypoints = [next - prev for (next, prev) in zip(ypoints[1:], ypoints[:-1])]
        xpoints = (edges[1:] + edges[:-1]) / 2

        plt.plot(xpoints, ypoints, color=colors.primary_color, label="Expected from\nnormal distribution")
        dist_plotted = True

    apply_style(style, eng=True, no_legend=not dist_plotted)


@provide_plot
def plot_v_over_time(title: str, units: str,
                     ts: TimeSeries,
                     plot_avg_dev: bool = True,
                     colors: Any = ColorProfile, style: Any = StyleProfile) -> None:

    min_time = min(ts.times)

    # /1000 is us to ms conversion
    time_points = [(val_time - min_time) / 1000 for val_time in ts.times]

    alpha = colors.noise_alpha if plot_avg_dev else 1.0
    plt.plot(time_points, ts.data, "o", color=colors.primary_color, alpha=alpha, label="Data")

    if plot_avg_dev:
        avg_vals = []
        low_vals_dev = []
        hight_vals_dev = []
        avg_times = []
        dev_times = []

        start = (len(ts.data) % style.avg_range) // 2
        points = list(range(start, len(ts.data) + 1, style.avg_range))

        for begin, end in zip(points[:-1], points[1:]):
            vals = ts.data[begin: end]

            cavg = average(vals)
            cdev = dev(vals)
            tavg = average(time_points[begin: end])

            avg_vals.append(cavg)
            avg_times.append(tavg)

            low_vals_dev.append(cavg - style.dev_range_x * cdev)
            hight_vals_dev.append(cavg + style.dev_range_x * cdev)
            dev_times.append(tavg)

        avg_timepoints = cast(List[float], numpy.linspace(avg_times[0], avg_times[-1], style.curve_approx_points))

        low_vals_dev = approximate_curve(dev_times, low_vals_dev, avg_timepoints, style.curve_approx_level)
        hight_vals_dev = approximate_curve(dev_times, hight_vals_dev, avg_timepoints, style.curve_approx_level)
        new_vals_avg = approximate_curve(avg_times, avg_vals, avg_timepoints, style.curve_approx_level)

        plt.plot(avg_timepoints, new_vals_avg, c=colors.suppl_color1,
                 label="Average\nover {}s".format(style.avg_range))
        plt.plot(avg_timepoints, low_vals_dev, c=colors.suppl_color2,
                 label="Avg \xB1 {} * stdev\nover {}s".format(style.dev_range_x, style.avg_range))
        plt.plot(avg_timepoints, hight_vals_dev, c=colors.suppl_color2)

    plt.xlim(-5, max(time_points) + 5)

    plt.xlabel("Time, seconds from test begin")
    plt.ylabel("{}. Average and \xB1stddev over {} points".format(units, style.avg_range))
    plt.title(title)
    apply_style(style, eng=True)


@provide_plot
def plot_lat_over_time(title: str, ts: TimeSeries, bins_vals: List[int], samples: int = 5,
                       colors: Any = ColorProfile, style: Any = StyleProfile) -> None:

    min_time = min(ts.times)
    times = [int(tm - min_time + 500) // 1000 for tm in ts.times]
    ts_len = len(times)
    step = ts_len / samples
    points = [times[int(i * step + 0.5)] for i in range(samples)]
    points.append(times[-1])
    bounds = list(zip(points[:-1], points[1:]))
    data = numpy.array(ts.data, dtype='int32')
    data.shape = [len(ts.data) // ts.second_axis_size, ts.second_axis_size]  # type: ignore
    agg_data = []
    positions = []
    labels = []

    min_idxs = []
    max_idxs = []

    for begin, end in bounds:
        agg_hist = numpy.sum(data[begin:end], axis=0)

        vals = numpy.empty(shape=(numpy.sum(agg_hist),), dtype='float32')
        cidx = 0
        non_zero = agg_hist.nonzero()[0]
        min_idxs.append(non_zero[0])
        max_idxs.append(non_zero[-1])

        for pos in non_zero:
            vals[cidx:cidx + agg_hist[pos]] = bins_vals[pos]
            cidx += agg_hist[pos]

        agg_data.append(vals)
        positions.append((end + begin) / 2)
        labels.append(str((end + begin) // 2))

    min_y = bins_vals[min(min_idxs)]
    max_y = bins_vals[max(max_idxs)]

    min_y -= (max_y - min_y) * 0.05
    max_y += (max_y - min_y) * 0.05

    # plot box size adjust (only plot, not spines and legend)
    plt.boxplot(agg_data, 0, '', positions=positions, labels=labels, widths=step / 4)
    plt.xlim(min(times), max(times))
    plt.ylim(min_y, max_y)
    plt.xlabel("Time, seconds from test begin, sampled for ~{} seconds".format(int(step)))
    plt.ylabel("Latency, ms")
    plt.title(title)
    apply_style(style, eng=True, no_legend=True)


@provide_plot
def plot_heatmap(title: str, ts: TimeSeries, bins_vals: List[int], samples: int = 5,
                 colors: Any = ColorProfile, style: Any = StyleProfile) -> None:
    hist_bins_count = 20
    bin_top = [100 * 2 ** i for i in range(20)]
    bin_ranges = [[0, 0]]
    cborder_it = iter(bin_top)
    cborder = next(cborder_it)
    for bin_val in bins_vals:
        if bin_val < cborder:
            bin_ranges

    # bins: [100us, 200us, ...., 104s]
    # msp origin bins ranges to heatmap bins

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

    # figure size in inches
    figsize = (12, 6)

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

    fig, p1 = plt.subplots(figsize=figsize)

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
    plt.legend(handles1 + handles2, labels1 + labels2, loc=legend_location, bbox_to_anchor=legend_bbox_to_anchor)

    # adjust central box size to fit legend
    plt.subplots_adjust(**plot_box_adjust)
    apply_style(style, eng=False, no_legend=True)


#  --------------------  REPORT HELPERS --------------------------------------------------------------------------------


class HTMLBlock:
    data = None  # type: str
    js_links = []  # type: List[str]
    css_links = []  # type: List[str]


class Menu1st:
    engineering = "Engineering"
    summary = "Summary"


class Menu2ndEng:
    iops_time = "IOPS(time)"
    hist = "IOPS/lat overall histogram"
    lat_time = "Lat(time)"


class Menu2ndSumm:
    io_lat_qd = "IO & Lat vs QD"


menu_1st_order = [Menu1st.summary, Menu1st.engineering]


#  --------------------  REPORTS  --------------------------------------------------------------------------------------


class Reporter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_divs(self, suite: TestSuiteConfig, storage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        pass


# Main performance report
class PerformanceSummary(Reporter):
    """Creates graph, which show how IOPS and Latency depend on QD"""


# Main performance report
class IO_QD(Reporter):
    """Creates graph, which show how IOPS and Latency depend on QD"""
    def get_divs(self, suite: TestSuiteConfig, rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        ts_map = {}  # type: Dict[FioTestSumm, List[IOSummary]]
        str_summary = {}  # type: Dict[FioTestSumm, List[IOSummary]]
        for job in rstorage.iter_job(suite):
            fjob = cast(FioJobConfig, job)
            tpl_no_qd = fjob.characterized_tuple_no_qd()
            io_summ = make_iosum(rstorage, suite, job)

            if tpl_no_qd not in ts_map:
                ts_map[tpl_no_qd] = [io_summ]
                str_summary[tpl_no_qd] = (fjob.summary_no_qd(), fjob.long_summary_no_qd())
            else:
                ts_map[tpl_no_qd].append(io_summ)

        for tpl, iosums in ts_map.items():
            iosums.sort(key=lambda x: x.qd)
            summary, summary_long = str_summary[tlp]

            ds = DataSource(suite_id=suite.storage_id,
                            job_id="io_over_qd_".format(summary),
                            node_id="__all__",
                            dev='fio',
                            sensor="io_over_qd",
                            tag="svg")

            title = "IOPS, BW, Lat vs. QD.\n" + summary_long
            fpath = io_chart(rstorage, ds, title=title, legend="IOPS/BW", iosums=iosums)
            yield Menu1st.summary, Menu2ndSumm.io_lat_qd, html.img(fpath)
            if DEBUG:
                return


# Linearization report
class IOPS_Bsize(Reporter):
    """Creates graphs, which show how IOPS and Latency depend on block size"""


# IOPS/latency distribution
class IOHist(Reporter):
    """IOPS.latency distribution histogram"""
    def get_divs(self, suite: TestSuiteConfig, rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        for job in rstorage.iter_job(suite):
            fjob = cast(FioJobConfig, job)
            agg_lat = get_aggregated(rstorage, suite, fjob, "lat")
            bins_edges = numpy.array(get_lat_vals(agg_lat.second_axis_size), dtype='float32') / 1000  # convert us to ms
            lat_stat_prop = calc_histo_stat_props(agg_lat, bins_edges, bins_count=StyleProfile.hist_boxes)

            title = "Latency distribution. " + fjob.long_summary
            units = "ms"

            fpath = plot_hist(rstorage, agg_lat.source(tag='hist.svg'), title, units, lat_stat_prop)
            if DEBUG:
                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, html.img(fpath)
            else:
                yield Menu1st.engineering, Menu2ndEng.hist, html.img(fpath)

            agg_io = get_aggregated(rstorage, suite, fjob, "bw")

            if fjob.bsize >= LARGE_BLOCKS:
                title = "BW distribution. " + fjob.long_summary
                units = "MiBps"
                agg_io.data /= MiB2KiB
            else:
                title = "IOPS distribution. " + fjob.long_summary
                agg_io.data /= fjob.bsize
                units = "IOPS"

            io_stat_prop = calc_norm_stat_props(agg_io, bins_count=StyleProfile.hist_boxes)
            fpath = plot_hist(rstorage, agg_io.source(tag='hist.svg'), title, units, io_stat_prop)
            if DEBUG:
                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, html.img(fpath)
                return
            else:
                yield Menu1st.engineering, Menu2ndEng.hist, html.img(fpath)


# IOPS/latency over test time for each job
class IOTime(Reporter):
    """IOPS/latency during test"""
    def get_divs(self, suite: TestSuiteConfig, rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        for job in rstorage.iter_job(suite):
            fjob = cast(FioJobConfig, job)
            agg_lat = get_aggregated(rstorage, suite, fjob, "lat")
            bins_edges = numpy.array(get_lat_vals(agg_lat.second_axis_size), dtype='float32') / 1000
            title = "Latency during test. " + fjob.long_summary

            fpath = plot_lat_over_time(rstorage, agg_lat.source(tag='ts.svg'), title, agg_lat, bins_edges)
            if DEBUG:
                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, html.img(fpath)
            else:
                yield Menu1st.engineering, Menu2ndEng.lat_time, html.img(fpath)

            fpath = plot_heatmap(rstorage, agg_lat.source(tag='hmap.svg'), title, agg_lat, bins_edges)
            if DEBUG:
                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, html.img(fpath)
            else:
                yield Menu1st.engineering, Menu2ndEng.lat_time, html.img(fpath)

            agg_io = get_aggregated(rstorage, suite, fjob, "bw")
            if fjob.bsize >= LARGE_BLOCKS:
                title = "BW during test. " + fjob.long_summary
                units = "MiBps"
                agg_io.data /= MiB2KiB
            else:
                title = "IOPS during test. " + fjob.long_summary
                agg_io.data /= fjob.bsize
                units = "IOPS"

            fpath = plot_v_over_time(rstorage, agg_io.source(tag='ts.svg'), title, units, agg_io)

            if DEBUG:
                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, html.img(fpath)
                return
            else:
                yield Menu1st.engineering, Menu2ndEng.iops_time, html.img(fpath)


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



def get_sensor(storage: Storage, node: str, sensor: str, dev: str, metric: str,
               time_range: Tuple[int, int]) -> numpy.array:
    """Return sensor values for given node for given period. Return per second estimated values array

    Raise an error if required range is not full covered by data in storage.
    First it finds range of results from sensor, which fully covers requested range.
    ...."""

    collected_at = numpy.array(storage.get_array("sensors/{}_collected_at".format(node)), dtype="int")
    data = numpy.array(storage.get_array("sensors/{}_{}.{}.{}".format(node, sensor, dev, metric)))

    # collected_at is array of pairs (collection_started_at, collection_finished_at)
    collection_start_at = collected_at[::2]

    MICRO = 1000000

    # convert secods to us
    begin = time_range[0] * MICRO
    end = time_range[1] * MICRO

    if begin < collection_start_at[0] or end > collection_start_at[-1] or end <= begin:
        raise AssertionError(("Incorrect data for get_sensor - time_range={!r}, collected_at=[{}, ..., {}]," +
                              "sensor = {}_{}.{}.{}").format(time_range,
                                                             collected_at[0] // MICRO,
                                                             collected_at[-1] // MICRO,
                                                             node, sensor, dev, metric))

    pos1, pos2 = numpy.searchsorted(collection_start_at, (begin, end))
    assert pos1 >= 1

    time_bounds = collection_start_at[pos1 - 1: pos2]
    edge_it = iter(time_bounds)
    val_it = iter(data[pos1 - 1: pos2])

    result = []
    curr_summ = 0

    results_cell_ends = begin + MICRO
    curr_end = next(edge_it)

    while results_cell_ends <= end:
        curr_start = curr_end
        curr_end = next(edge_it)
        curr_val = next(val_it)
        while curr_end >= results_cell_ends and results_cell_ends <= end:
            current_part = (results_cell_ends - curr_start) / (curr_end - curr_start) * curr_val
            result.append(curr_summ + current_part)
            curr_summ = 0
            curr_val -= current_part
            curr_start = results_cell_ends
            results_cell_ends += MICRO
        curr_summ += curr_val

    assert len(result) == (end - begin) // MICRO
    return result


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
class ClusterLoad(Reporter):
    """IOPS/latency during test"""

    storage_sensors = [
        ('block-io', 'reads_completed', "Read ops"),
        ('block-io', 'writes_completed', "Write ops"),
        ('block-io', 'sectors_read', "Read kb"),
        ('block-io', 'sectors_written', "Write kb"),
    ]

    def get_divs(self, suite: TestSuiteConfig, rstorage: ResultStorage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        # split nodes on test and other
        storage = rstorage.storage
        nodes = storage.load_list(NodeInfo, "all_nodes")  # type: List[NodeInfo]

        test_nodes = {node.node_id for node in nodes if 'testnode' in node.roles}
        cluster_nodes = {node.node_id for node in nodes if 'testnode' not in node.roles}

        for job in rstorage.iter_job(suite):
            # convert ms to s
            time_range = (job.reliable_info_starts_at // MS2S, job.reliable_info_stops_at // MS2S)
            len = time_range[1] - time_range[0]

            for sensor, metric, sensor_title in self.storage_sensors:
                sum_testnode = numpy.zeros((len,))
                sum_other = numpy.zeros((len,))

                for path, groups in iter_sensors(rstorage.storage, sensor=sensor, metric=metric):
                    data = get_sensor(rstorage.storage, groups['node'], sensor, groups['dev'], metric, time_range)
                    if groups['node'] in test_nodes:
                        sum_testnode += data
                    else:
                        sum_other += data

                ds = DataSource(suite_id=suite.storage_id, job_id=job.summary, node_id="cluster",
                                dev=sensor, sensor=metric, tag="ts.svg")

                # s to ms
                ts = TimeSeries(name="", times=numpy.arange(*time_range) * MS2S, data=sum_testnode, raw=None)
                fpath = plot_v_over_time(rstorage, ds, "{}.{}".format(sensor, metric), sensor_title, ts=ts)
                yield Menu1st.engineering, Menu2ndEng.iops_time, html.img(fpath)

            if DEBUG:
                return


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
        reporters = [ClusterLoad()] # IO_QD(), IOTime(), IOHist()] # type: List[Reporter]

        root_dir = os.path.dirname(os.path.dirname(wally.__file__))
        doc_templ_path = os.path.join(root_dir, "report_templates/index.html")
        report_template = open(doc_templ_path, "rt").read()
        css_file_src = os.path.join(root_dir, "report_templates/main.css")
        css_file = open(css_file_src, "rt").read()

        menu_block = []
        content_block = []
        link_idx = 0

        matplotlib.rcParams.update({'font.size': 10})

        items = defaultdict(lambda: defaultdict(list))  # type: Dict[str, Dict[str, list]]
        for suite in rstorage.iter_suite(FioTest.name):
            for reporter in reporters:
                for block, item, html in reporter.get_divs(suite, rstorage):
                    items[block][item].append(html)

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
                content_block.extend("    " + x for x in items[menu_1st][menu_2nd])
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


#  ---------------------------   LEGASY --------------------------------------------------------------------------------


# # disk_info = None
# # base = None
# # linearity = None
#
#
# def group_by_name(test_data):
#     name_map = collections.defaultdict(lambda: [])
#
#     for data in test_data:
#         name_map[(data.name, data.summary())].append(data)
#
#     return name_map
#
#
# def report(name, required_fields):
#     def closure(func):
#         report_funcs.append((required_fields.split(","), name, func))
#         return func
#     return closure
#
#
# def get_test_lcheck_params(pinfo):
#     res = [{
#         's': 'sync',
#         'd': 'direct',
#         'a': 'async',
#         'x': 'sync direct'
#     }[pinfo.sync_mode]]
#
#     res.append(pinfo.p.rw)
#
#     return " ".join(res)
#
#
# def get_emb_data_svg(plt):
#     sio = StringIO()
#     plt.savefig(sio, format='svg')
#     img_start = "<!-- Created with matplotlib (http://matplotlib.org/) -->"
#     return sio.getvalue().split(img_start, 1)[1]
#
#
# def get_template(templ_name):
#     very_root_dir = os.path.dirname(os.path.dirname(wally.__file__))
#     templ_dir = os.path.join(very_root_dir, 'report_templates')
#     templ_file = os.path.join(templ_dir, templ_name)
#     return open(templ_file, 'r').read()
#
#
# def group_by(data, func):
#     if len(data) < 2:
#         yield data
#         return
#
#     ndata = [(func(dt), dt) for dt in data]
#     ndata.sort(key=func)
#     pkey, dt = ndata[0]
#     curr_list = [dt]
#
#     for key, val in ndata[1:]:
#         if pkey != key:
#             yield curr_list
#             curr_list = [val]
#         else:
#             curr_list.append(val)
#         pkey = key
#
#     yield curr_list
#
#
# @report('linearity', 'linearity_test')
# def linearity_report(processed_results, lab_info, comment):
#     labels_and_data_mp = collections.defaultdict(lambda: [])
#     vls = {}
#
#     # plot io_time = func(bsize)
#     for res in processed_results.values():
#         if res.name.startswith('linearity_test'):
#             iotimes = [1000. / val for val in res.iops.raw]
#
#             op_summ = get_test_summary(res.params)[:3]
#
#             labels_and_data_mp[op_summ].append(
#                 [res.p.blocksize, res.iops.raw, iotimes])
#
#             cvls = res.params.vals.copy()
#             del cvls['blocksize']
#             del cvls['rw']
#
#             cvls.pop('sync', None)
#             cvls.pop('direct', None)
#             cvls.pop('buffered', None)
#
#             if op_summ not in vls:
#                 vls[op_summ] = cvls
#             else:
#                 assert cvls == vls[op_summ]
#
#     all_labels = None
#     _, ax1 = plt.subplots()
#     for name, labels_and_data in labels_and_data_mp.items():
#         labels_and_data.sort(key=lambda x: ssize2b(x[0]))
#
#         labels, _, iotimes = zip(*labels_and_data)
#
#         if all_labels is None:
#             all_labels = labels
#         else:
#             assert all_labels == labels
#
#         plt.boxplot(iotimes)
#         if len(labels_and_data) > 2 and \
#            ssize2b(labels_and_data[-2][0]) >= 4096:
#
#             xt = range(1, len(labels) + 1)
#
#             def io_time(sz, bw, initial_lat):
#                 return sz / bw + initial_lat
#
#             x = numpy.array(map(ssize2b, labels))
#             y = numpy.array([sum(dt) / len(dt) for dt in iotimes])
#             popt, _ = scipy.optimize.curve_fit(io_time, x, y, p0=(100., 1.))
#
#             y1 = io_time(x, *popt)
#             plt.plot(xt, y1, linestyle='--',
#                      label=name + ' LS linear approx')
#
#             for idx, (sz, _, _) in enumerate(labels_and_data):
#                 if ssize2b(sz) >= 4096:
#                     break
#
#             bw = (x[-1] - x[idx]) / (y[-1] - y[idx])
#             lat = y[-1] - x[-1] / bw
#             y2 = io_time(x, bw, lat)
#             plt.plot(xt, y2, linestyle='--',
#                      label=abbv_name_to_full(name) +
#                      ' (4k & max) linear approx')
#
#     plt.setp(ax1, xticklabels=labels)
#
#     plt.xlabel("Block size")
#     plt.ylabel("IO time, ms")
#
#     plt.subplots_adjust(top=0.85)
#     plt.legend(bbox_to_anchor=(0.5, 1.15),
#                loc='upper center',
#                prop={'size': 10}, ncol=2)
#     plt.grid()
#     iotime_plot = get_emb_data_svg(plt)
#     plt.clf()
#
#     # plot IOPS = func(bsize)
#     _, ax1 = plt.subplots()
#
#     for name, labels_and_data in labels_and_data_mp.items():
#         labels_and_data.sort(key=lambda x: ssize2b(x[0]))
#         _, data, _ = zip(*labels_and_data)
#         plt.boxplot(data)
#         avg = [float(sum(arr)) / len(arr) for arr in data]
#         xt = range(1, len(data) + 1)
#         plt.plot(xt, avg, linestyle='--',
#                  label=abbv_name_to_full(name) + " avg")
#
#     plt.setp(ax1, xticklabels=labels)
#     plt.xlabel("Block size")
#     plt.ylabel("IOPS")
#     plt.legend(bbox_to_anchor=(0.5, 1.15),
#                loc='upper center',
#                prop={'size': 10}, ncol=2)
#     plt.grid()
#     plt.subplots_adjust(top=0.85)
#
#     iops_plot = get_emb_data_svg(plt)
#
#     res = set(get_test_lcheck_params(res) for res in processed_results.values())
#     ncount = list(set(res.testnodes_count for res in processed_results.values()))
#     conc = list(set(res.concurence for res in processed_results.values()))
#
#     assert len(conc) == 1
#     assert len(ncount) == 1
#
#     descr = {
#         'vm_count': ncount[0],
#         'concurence': conc[0],
#         'oper_descr': ", ".join(res).capitalize()
#     }
#
#     params_map = {'iotime_vs_size': iotime_plot,
#                   'iops_vs_size': iops_plot,
#                   'descr': descr}
#
#     return get_template('report_linearity.html').format(**params_map)
#
#
# @report('lat_vs_iops', 'lat_vs_iops')
# def lat_vs_iops(processed_results, lab_info, comment):
#     lat_iops = collections.defaultdict(lambda: [])
#     requsted_vs_real = collections.defaultdict(lambda: {})
#
#     for res in processed_results.values():
#         if res.name.startswith('lat_vs_iops'):
#             lat_iops[res.concurence].append((res.lat,
#                                              0,
#                                              res.iops.average,
#                                              res.iops.deviation))
#             # lat_iops[res.concurence].append((res.lat.average / 1000.0,
#             #                                  res.lat.deviation / 1000.0,
#             #                                  res.iops.average,
#             #                                  res.iops.deviation))
#             requested_iops = res.p.rate_iops * res.concurence
#             requsted_vs_real[res.concurence][requested_iops] = \
#                 (res.iops.average, res.iops.deviation)
#
#     colors = ['red', 'green', 'blue', 'orange', 'magenta', "teal"]
#     colors_it = iter(colors)
#     for conc, lat_iops in sorted(lat_iops.items()):
#         lat, dev, iops, iops_dev = zip(*lat_iops)
#         plt.errorbar(iops, lat, xerr=iops_dev, yerr=dev, fmt='ro',
#                      label=str(conc) + " threads",
#                      color=next(colors_it))
#
#     plt.xlabel("IOPS")
#     plt.ylabel("Latency, ms")
#     plt.grid()
#     plt.legend(loc=0)
#     plt_iops_vs_lat = get_emb_data_svg(plt)
#     plt.clf()
#
#     colors_it = iter(colors)
#     for conc, req_vs_real in sorted(requsted_vs_real.items()):
#         req, real = zip(*sorted(req_vs_real.items()))
#         iops, dev = zip(*real)
#         plt.errorbar(req, iops, yerr=dev, fmt='ro',
#                      label=str(conc) + " threads",
#                      color=next(colors_it))
#     plt.xlabel("Requested IOPS")
#     plt.ylabel("Get IOPS")
#     plt.grid()
#     plt.legend(loc=0)
#     plt_iops_vs_requested = get_emb_data_svg(plt)
#
#     res1 = processed_results.values()[0]
#     params_map = {'iops_vs_lat': plt_iops_vs_lat,
#                   'iops_vs_requested': plt_iops_vs_requested,
#                   'oper_descr': get_test_lcheck_params(res1).capitalize()}
#
#     return get_template('report_iops_vs_lat.html').format(**params_map)
#
#
# def render_all_html(comment, info, lab_description, images, templ_name):
#     data = info.__dict__.copy()
#     for name, val in data.items():
#         if not name.startswith('__'):
#             if val is None:
#                 if name in ('direct_iops_w64_max', 'direct_iops_w_max'):
#                     data[name] = ('-', '-', '-')
#                 else:
#                     data[name] = '-'
#             elif isinstance(val, (int, float, long)):
#                 data[name] = round_3_digit(val)
#
#     data['bw_read_max'] = (data['bw_read_max'][0] // 1024,
#                            data['bw_read_max'][1],
#                            data['bw_read_max'][2])
#
#     data['bw_write_max'] = (data['bw_write_max'][0] // 1024,
#                             data['bw_write_max'][1],
#                             data['bw_write_max'][2])
#
#     images.update(data)
#     templ = get_template(templ_name)
#     return templ.format(lab_info=lab_description,
#                         comment=comment,
#                         **images)
#
#
# def io_chart(title, concurence,
#              latv, latv_min, latv_max,
#              iops_or_bw, iops_or_bw_err,
#              legend,
#              log_iops=False,
#              log_lat=False,
#              boxplots=False,
#              latv_50=None,
#              latv_95=None,
#              error2=None):
#
#     matplotlib.rcParams.update({'font.size': 10})
#     points = " MiBps" if legend == 'BW' else ""
#     lc = len(concurence)
#     width = 0.35
#     xt = range(1, lc + 1)
#
#     op_per_vm = [v / (vm * th) for v, (vm, th) in zip(iops_or_bw, concurence)]
#     fig, p1 = plt.subplots()
#     xpos = [i - width / 2 for i in xt]
#
#     p1.bar(xpos, iops_or_bw,
#            width=width,
#            color='y',
#            label=legend)
#
#     err1_leg = None
#     for pos, y, err in zip(xpos, iops_or_bw, iops_or_bw_err):
#         err1_leg = p1.errorbar(pos + width / 2,
#                                y,
#                                err,
#                                color='magenta')
#
#     err2_leg = None
#     if error2 is not None:
#         for pos, y, err in zip(xpos, iops_or_bw, error2):
#             err2_leg = p1.errorbar(pos + width / 2 + 0.08,
#                                    y,
#                                    err,
#                                    lw=2,
#                                    alpha=0.5,
#                                    color='teal')
#
#     p1.grid(True)
#     p1.plot(xt, op_per_vm, '--', label=legend + "/thread", color='black')
#     handles1, labels1 = p1.get_legend_handles_labels()
#
#     handles1 += [err1_leg]
#     labels1 += ["95% conf"]
#
#     if err2_leg is not None:
#         handles1 += [err2_leg]
#         labels1 += ["95% dev"]
#
#     p2 = p1.twinx()
#
#     if latv_50 is None:
#         p2.plot(xt, latv_max, label="lat max")
#         p2.plot(xt, latv, label="lat avg")
#         p2.plot(xt, latv_min, label="lat min")
#     else:
#         p2.plot(xt, latv_50, label="lat med")
#         p2.plot(xt, latv_95, label="lat 95%")
#
#     plt.xlim(0.5, lc + 0.5)
#     plt.xticks(xt, ["{0} * {1}".format(vm, th) for (vm, th) in concurence])
#     p1.set_xlabel("VM Count * Thread per VM")
#     p1.set_ylabel(legend + points)
#     p2.set_ylabel("Latency ms")
#     plt.title(title)
#     handles2, labels2 = p2.get_legend_handles_labels()
#
#     plt.legend(handles1 + handles2, labels1 + labels2,
#                loc='center left', bbox_to_anchor=(1.1, 0.81))
#
#     if log_iops:
#         p1.set_yscale('log')
#
#     if log_lat:
#         p2.set_yscale('log')
#
#     plt.subplots_adjust(right=0.68)
#
#     return get_emb_data_svg(plt)
#
#
# def make_plots(processed_results, plots):
#     """
#     processed_results: [PerfInfo]
#     plots = [(test_name_prefix:str, fname:str, description:str)]
#     """
#     files = {}
#     for name_pref, fname, desc in plots:
#         chart_data = []
#
#         for res in processed_results:
#             summ = res.name + "_" + res.summary
#             if summ.startswith(name_pref):
#                 chart_data.append(res)
#
#         if len(chart_data) == 0:
#             raise ValueError("Can't found any date for " + name_pref)
#
#         use_bw = ssize2b(chart_data[0].p.blocksize) > 16 * 1024
#
#         chart_data.sort(key=lambda x: x.params['vals']['numjobs'])
#
#         lat = None
#         lat_min = None
#         lat_max = None
#
#         lat_50 = [x.lat_50 for x in chart_data]
#         lat_95 = [x.lat_95 for x in chart_data]
#
#         lat_diff_max = max(x.lat_95 / x.lat_50 for x in chart_data)
#         lat_log_scale = (lat_diff_max > 10)
#
#         testnodes_count = x.testnodes_count
#         concurence = [(testnodes_count, x.concurence)
#                       for x in chart_data]
#
#         if use_bw:
#             data = [x.bw.average / 1000 for x in chart_data]
#             data_conf = [x.bw.confidence / 1000 for x in chart_data]
#             data_dev = [x.bw.deviation * 2.5 / 1000 for x in chart_data]
#             name = "BW"
#         else:
#             data = [x.iops.average for x in chart_data]
#             data_conf = [x.iops.confidence for x in chart_data]
#             data_dev = [x.iops.deviation * 2 for x in chart_data]
#             name = "IOPS"
#
#         fc = io_chart(title=desc,
#                       concurence=concurence,
#
#                       latv=lat,
#                       latv_min=lat_min,
#                       latv_max=lat_max,
#
#                       iops_or_bw=data,
#                       iops_or_bw_err=data_conf,
#
#                       legend=name,
#                       log_lat=lat_log_scale,
#
#                       latv_50=lat_50,
#                       latv_95=lat_95,
#
#                       error2=data_dev)
#         files[fname] = fc
#
#     return files
#
#
# def find_max_where(processed_results, sync_mode, blocksize, rw, iops=True):
#     result = None
#     attr = 'iops' if iops else 'bw'
#     for measurement in processed_results:
#         ok = measurement.sync_mode == sync_mode
#         ok = ok and (measurement.p.blocksize == blocksize)
#         ok = ok and (measurement.p.rw == rw)
#
#         if ok:
#             field = getattr(measurement, attr)
#
#             if result is None:
#                 result = field
#             elif field.average > result.average:
#                 result = field
#
#     return result
#
#
# def get_disk_info(processed_results):
#     di = DiskInfo()
#     di.direct_iops_w_max = find_max_where(processed_results,
#                                           'd', '4k', 'randwrite')
#     di.direct_iops_r_max = find_max_where(processed_results,
#                                           'd', '4k', 'randread')
#
#     di.direct_iops_w64_max = find_max_where(processed_results,
#                                             'd', '64k', 'randwrite')
#
#     for sz in ('16m', '64m'):
#         di.bw_write_max = find_max_where(processed_results,
#                                          'd', sz, 'randwrite', False)
#         if di.bw_write_max is not None:
#             break
#
#     if di.bw_write_max is None:
#         for sz in ('1m', '2m', '4m', '8m'):
#             di.bw_write_max = find_max_where(processed_results,
#                                              'd', sz, 'write', False)
#             if di.bw_write_max is not None:
#                 break
#
#     for sz in ('16m', '64m'):
#         di.bw_read_max = find_max_where(processed_results,
#                                         'd', sz, 'randread', False)
#         if di.bw_read_max is not None:
#             break
#
#     if di.bw_read_max is None:
#         di.bw_read_max = find_max_where(processed_results,
#                                         'd', '1m', 'read', False)
#
#     rws4k_iops_lat_th = []
#     for res in processed_results:
#         if res.sync_mode in 'xs' and res.p.blocksize == '4k':
#             if res.p.rw != 'randwrite':
#                 continue
#             rws4k_iops_lat_th.append((res.iops.average,
#                                       res.lat,
#                                       # res.lat.average,
#                                       res.concurence))
#
#     rws4k_iops_lat_th.sort(key=lambda x: x[2])
#
#     latv = [lat for _, lat, _ in rws4k_iops_lat_th]
#
#     for tlat in [10, 30, 100]:
#         pos = bisect.bisect_left(latv, tlat)
#         if 0 == pos:
#             setattr(di, 'rws4k_{}ms'.format(tlat), 0)
#         elif pos == len(latv):
#             iops3, _, _ = rws4k_iops_lat_th[-1]
#             iops3 = int(round_3_digit(iops3))
#             setattr(di, 'rws4k_{}ms'.format(tlat), ">=" + str(iops3))
#         else:
#             lat1 = latv[pos - 1]
#             lat2 = latv[pos]
#
#             iops1, _, th1 = rws4k_iops_lat_th[pos - 1]
#             iops2, _, th2 = rws4k_iops_lat_th[pos]
#
#             th_lat_coef = (th2 - th1) / (lat2 - lat1)
#             th3 = th_lat_coef * (tlat - lat1) + th1
#
#             th_iops_coef = (iops2 - iops1) / (th2 - th1)
#             iops3 = th_iops_coef * (th3 - th1) + iops1
#             iops3 = int(round_3_digit(iops3))
#             setattr(di, 'rws4k_{}ms'.format(tlat), iops3)
#
#     hdi = DiskInfo()
#
#     def pp(x):
#         med, conf = x.rounded_average_conf()
#         conf_perc = int(float(conf) / med * 100)
#         dev_perc = int(float(x.deviation) / med * 100)
#         return (round_3_digit(med), conf_perc, dev_perc)
#
#     hdi.direct_iops_r_max = pp(di.direct_iops_r_max)
#
#     if di.direct_iops_w_max is not None:
#         hdi.direct_iops_w_max = pp(di.direct_iops_w_max)
#     else:
#         hdi.direct_iops_w_max = None
#
#     if di.direct_iops_w64_max is not None:
#         hdi.direct_iops_w64_max = pp(di.direct_iops_w64_max)
#     else:
#         hdi.direct_iops_w64_max = None
#
#     hdi.bw_write_max = pp(di.bw_write_max)
#     hdi.bw_read_max = pp(di.bw_read_max)
#
#     hdi.rws4k_10ms = di.rws4k_10ms if 0 != di.rws4k_10ms else None
#     hdi.rws4k_30ms = di.rws4k_30ms if 0 != di.rws4k_30ms else None
#     hdi.rws4k_100ms = di.rws4k_100ms if 0 != di.rws4k_100ms else None
#     return hdi
#
#
# @report('hdd', 'hdd')
# def make_hdd_report(processed_results, lab_info, comment):
#     plots = [
#         ('hdd_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
#         ('hdd_rwx4k', 'rand_write_4k', 'Random write 4k sync IOPS')
#     ]
#     perf_infos = [res.disk_perf_info() for res in processed_results]
#     images = make_plots(perf_infos, plots)
#     di = get_disk_info(perf_infos)
#     return render_all_html(comment, di, lab_info, images, "report_hdd.html")
#
#
# @report('cinder_iscsi', 'cinder_iscsi')
# def make_cinder_iscsi_report(processed_results, lab_info, comment):
#     plots = [
#         ('cinder_iscsi_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
#         ('cinder_iscsi_rwx4k', 'rand_write_4k', 'Random write 4k sync IOPS')
#     ]
#     perf_infos = [res.disk_perf_info() for res in processed_results]
#     try:
#         images = make_plots(perf_infos, plots)
#     except ValueError:
#         plots = [
#             ('cinder_iscsi_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
#             ('cinder_iscsi_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS')
#         ]
#         images = make_plots(perf_infos, plots)
#     di = get_disk_info(perf_infos)
#
#     return render_all_html(comment, di, lab_info, images, "report_cinder_iscsi.html")
#
#
# @report('ceph', 'ceph')
# def make_ceph_report(processed_results, lab_info, comment):
#     plots = [
#         ('ceph_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
#         ('ceph_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS'),
#         ('ceph_rrd16m', 'rand_read_16m', 'Random read 16m direct MiBps'),
#         ('ceph_rwd16m', 'rand_write_16m',
#          'Random write 16m direct MiBps'),
#     ]
#
#     perf_infos = [res.disk_perf_info() for res in processed_results]
#     images = make_plots(perf_infos, plots)
#     di = get_disk_info(perf_infos)
#     return render_all_html(comment, di, lab_info, images, "report_ceph.html")
#
#
# @report('mixed', 'mixed')
# def make_mixed_report(processed_results, lab_info, comment):
#     #
#     # IOPS(X% read) = 100 / ( X / IOPS_W + (100 - X) / IOPS_R )
#     #
#
#     perf_infos = [res.disk_perf_info() for res in processed_results]
#     mixed = collections.defaultdict(lambda: [])
#
#     is_ssd = False
#     for res in perf_infos:
#         if res.name.startswith('mixed'):
#             if res.name.startswith('mixed-ssd'):
#                 is_ssd = True
#             mixed[res.concurence].append((res.p.rwmixread,
#                                           res.lat,
#                                           0,
#                                           # res.lat.average / 1000.0,
#                                           # res.lat.deviation / 1000.0,
#                                           res.iops.average,
#                                           res.iops.deviation))
#
#     if len(mixed) == 0:
#         raise ValueError("No mixed load found")
#
#     fig, p1 = plt.subplots()
#     p2 = p1.twinx()
#
#     colors = ['red', 'green', 'blue', 'orange', 'magenta', "teal"]
#     colors_it = iter(colors)
#     for conc, mix_lat_iops in sorted(mixed.items()):
#         mix_lat_iops = sorted(mix_lat_iops)
#         read_perc, lat, dev, iops, iops_dev = zip(*mix_lat_iops)
#         p1.errorbar(read_perc, iops, color=next(colors_it),
#                     yerr=iops_dev, label=str(conc) + " th")
#
#         p2.errorbar(read_perc, lat, color=next(colors_it),
#                     ls='--', yerr=dev, label=str(conc) + " th lat")
#
#     if is_ssd:
#         p1.set_yscale('log')
#         p2.set_yscale('log')
#
#     p1.set_xlim(-5, 105)
#
#     read_perc = set(read_perc)
#     read_perc.add(0)
#     read_perc.add(100)
#     read_perc = sorted(read_perc)
#
#     plt.xticks(read_perc, map(str, read_perc))
#
#     p1.grid(True)
#     p1.set_xlabel("% of reads")
#     p1.set_ylabel("Mixed IOPS")
#     p2.set_ylabel("Latency, ms")
#
#     handles1, labels1 = p1.get_legend_handles_labels()
#     handles2, labels2 = p2.get_legend_handles_labels()
#     plt.subplots_adjust(top=0.85)
#     plt.legend(handles1 + handles2, labels1 + labels2,
#                bbox_to_anchor=(0.5, 1.15),
#                loc='upper center',
#                prop={'size': 12}, ncol=3)
#     plt.show()
#
#
# def make_load_report(idx, results_dir, fname):
#     dpath = os.path.join(results_dir, "io_" + str(idx))
#     files = sorted(os.listdir(dpath))
#     gf = lambda x: "_".join(x.rsplit(".", 1)[0].split('_')[:3])
#
#     for key, group in itertools.groupby(files, gf):
#         fname = os.path.join(dpath, key + ".fio")
#
#         cfgs = list(parse_all_in_1(open(fname).read(), fname))
#
#         fname = os.path.join(dpath, key + "_lat.log")
#
#         curr = []
#         arrays = []
#
#         with open(fname) as fd:
#             for offset, lat, _, _ in csv.reader(fd):
#                 offset = int(offset)
#                 lat = int(lat)
#                 if len(curr) > 0 and curr[-1][0] > offset:
#                     arrays.append(curr)
#                     curr = []
#                 curr.append((offset, lat))
#             arrays.append(curr)
#         conc = int(cfgs[0].vals.get('numjobs', 1))
#
#         if conc != 5:
#             continue
#
#         assert len(arrays) == len(cfgs) * conc
#
#         garrays = [[(0, 0)] for _ in range(conc)]
#
#         for offset in range(len(cfgs)):
#             for acc, new_arr in zip(garrays, arrays[offset * conc:(offset + 1) * conc]):
#                 last = acc[-1][0]
#                 for off, lat in new_arr:
#                     acc.append((off / 1000. + last, lat / 1000.))
#
#         for cfg, arr in zip(cfgs, garrays):
#             plt.plot(*zip(*arr[1:]))
#         plt.show()
#         exit(1)
#
#
# def make_io_report(dinfo, comment, path, lab_info=None):
#     lab_info = {
#         "total_disk": "None",
#         "total_memory": "None",
#         "nodes_count": "None",
#         "processor_count": "None"
#     }
#
#     try:
#         res_fields = sorted(v.name for v in dinfo)
#
#         found = False
#         for fields, name, func in report_funcs:
#             for field in fields:
#                 pos = bisect.bisect_left(res_fields, field)
#
#                 if pos == len(res_fields):
#                     break
#
#                 if not res_fields[pos].startswith(field):
#                     break
#             else:
#                 found = True
#                 hpath = path.format(name)
#
#                 try:
#                     report = func(dinfo, lab_info, comment)
#                 except:
#                     logger.exception("Diring {0} report generation".format(name))
#                     continue
#
#                 if report is not None:
#                     try:
#                         with open(hpath, "w") as fd:
#                             fd.write(report)
#                     except:
#                         logger.exception("Diring saving {0} report".format(name))
#                         continue
#                     logger.info("Report {0} saved into {1}".format(name, hpath))
#                 else:
#                     logger.warning("No report produced by {0!r}".format(name))
#
#         if not found:
#             logger.warning("No report generator found for this load")
#
#     except Exception as exc:
#         import traceback
#         traceback.print_exc()
#         logger.error("Failed to generate html report:" + str(exc))
#
#
#     # @classmethod
#     # def prepare_data(cls, results) -> List[Dict[str, Any]]:
#     #     """create a table with io performance report for console"""
#     #
#     #     def key_func(data: FioRunResult) -> Tuple[str, str, str, str, int]:
#     #         tpl = data.summary_tpl()
#     #         return (data.name,
#     #                 tpl.oper,
#     #                 tpl.mode,
#     #                 ssize2b(tpl.bsize),
#     #                 int(tpl.th_count) * int(tpl.vm_count))
#     #     res = []
#     #
#     #     for item in sorted(results, key=key_func):
#     #         test_dinfo = item.disk_perf_info()
#     #         testnodes_count = len(item.config.nodes)
#     #
#     #         iops, _ = test_dinfo.iops.rounded_average_conf()
#     #
#     #         if test_dinfo.iops_sys is not None:
#     #             iops_sys, iops_sys_conf = test_dinfo.iops_sys.rounded_average_conf()
#     #             _, iops_sys_dev = test_dinfo.iops_sys.rounded_average_dev()
#     #             iops_sys_per_vm = round_3_digit(iops_sys / testnodes_count)
#     #             iops_sys = round_3_digit(iops_sys)
#     #         else:
#     #             iops_sys = None
#     #             iops_sys_per_vm = None
#     #             iops_sys_dev = None
#     #             iops_sys_conf = None
#     #
#     #         bw, bw_conf = test_dinfo.bw.rounded_average_conf()
#     #         _, bw_dev = test_dinfo.bw.rounded_average_dev()
#     #         conf_perc = int(round(bw_conf * 100 / bw))
#     #         dev_perc = int(round(bw_dev * 100 / bw))
#     #
#     #         lat_50 = round_3_digit(int(test_dinfo.lat_50))
#     #         lat_95 = round_3_digit(int(test_dinfo.lat_95))
#     #         lat_avg = round_3_digit(int(test_dinfo.lat_avg))
#     #
#     #         iops_per_vm = round_3_digit(iops / testnodes_count)
#     #         bw_per_vm = round_3_digit(bw / testnodes_count)
#     #
#     #         iops = round_3_digit(iops)
#     #         bw = round_3_digit(bw)
#     #
#     #         summ = "{0.oper}{0.mode} {0.bsize:>4} {0.th_count:>3}th {0.vm_count:>2}vm".format(item.summary_tpl())
#     #
#     #         res.append({"name": key_func(item)[0],
#     #                     "key": key_func(item)[:4],
#     #                     "summ": summ,
#     #                     "iops": int(iops),
#     #                     "bw": int(bw),
#     #                     "conf": str(conf_perc),
#     #                     "dev": str(dev_perc),
#     #                     "iops_per_vm": int(iops_per_vm),
#     #                     "bw_per_vm": int(bw_per_vm),
#     #                     "lat_50": lat_50,
#     #                     "lat_95": lat_95,
#     #                     "lat_avg": lat_avg,
#     #
#     #                     "iops_sys": iops_sys,
#     #                     "iops_sys_per_vm": iops_sys_per_vm,
#     #                     "sys_conf": iops_sys_conf,
#     #                     "sys_dev": iops_sys_dev})
#     #
#     #     return res
#     #
#     # Field = collections.namedtuple("Field", ("header", "attr", "allign", "size"))
#     # fiels_and_header = [
#     #     Field("Name",           "name",        "l",  7),
#     #     Field("Description",    "summ",        "l", 19),
#     #     Field("IOPS\ncum",      "iops",        "r",  3),
#     #     # Field("IOPS_sys\ncum",  "iops_sys",    "r",  3),
#     #     Field("KiBps\ncum",     "bw",          "r",  6),
#     #     Field("Cnf %\n95%",     "conf",        "r",  3),
#     #     Field("Dev%",           "dev",         "r",  3),
#     #     Field("iops\n/vm",      "iops_per_vm", "r",  3),
#     #     Field("KiBps\n/vm",     "bw_per_vm",   "r",  6),
#     #     Field("lat ms\nmedian", "lat_50",      "r",  3),
#     #     Field("lat ms\n95%",    "lat_95",      "r",  3),
#     #     Field("lat\navg",       "lat_avg",     "r",  3),
#     # ]
#     #
#     # fiels_and_header_dct = dict((item.attr, item) for item in fiels_and_header)
#     #
#     # @classmethod
#     # def format_for_console(cls, results) -> str:
#     #     """create a table with io performance report for console"""
#     #
#     #     tab = texttable.Texttable(max_width=120)
#     #     tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
#     #     tab.set_cols_align([f.allign for f in cls.fiels_and_header])
#     #     sep = ["-" * f.size for f in cls.fiels_and_header]
#     #     tab.header([f.header for f in cls.fiels_and_header])
#     #     prev_k = None
#     #     for item in cls.prepare_data(results):
#     #         if prev_k is not None:
#     #             if prev_k != item["key"]:
#     #                 tab.add_row(sep)
#     #
#     #         prev_k = item["key"]
#     #         tab.add_row([item[f.attr] for f in cls.fiels_and_header])
#     #
#     #     return tab.draw()
#     #
#     # @classmethod
#     # def format_diff_for_console(cls, list_of_results: List[Any]) -> str:
#     #     """create a table with io performance report for console"""
#     #
#     #     tab = texttable.Texttable(max_width=200)
#     #     tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
#     #
#     #     header = [
#     #         cls.fiels_and_header_dct["name"].header,
#     #         cls.fiels_and_header_dct["summ"].header,
#     #     ]
#     #     allign = ["l", "l"]
#     #
#     #     header.append("IOPS ~ Cnf% ~ Dev%")
#     #     allign.extend(["r"] * len(list_of_results))
#     #     header.extend(
#     #         "IOPS_{0} %".format(i + 2) for i in range(len(list_of_results[1:]))
#     #     )
#     #
#     #     header.append("BW")
#     #     allign.extend(["r"] * len(list_of_results))
#     #     header.extend(
#     #         "BW_{0} %".format(i + 2) for i in range(len(list_of_results[1:]))
#     #     )
#     #
#     #     header.append("LAT")
#     #     allign.extend(["r"] * len(list_of_results))
#     #     header.extend(
#     #         "LAT_{0}".format(i + 2) for i in range(len(list_of_results[1:]))
#     #     )
#     #
#     #     tab.header(header)
#     #     sep = ["-" * 3] * len(header)
#     #     processed_results = map(cls.prepare_data, list_of_results)
#     #
#     #     key2results = []
#     #     for res in processed_results:
#     #         key2results.append(dict(
#     #             ((item["name"], item["summ"]), item) for item in res
#     #         ))
#     #
#     #     prev_k = None
#     #     iops_frmt = "{0[iops]} ~ {0[conf]:>2} ~ {0[dev]:>2}"
#     #     for item in processed_results[0]:
#     #         if prev_k is not None:
#     #             if prev_k != item["key"]:
#     #                 tab.add_row(sep)
#     #
#     #         prev_k = item["key"]
#     #
#     #         key = (item['name'], item['summ'])
#     #         line = list(key)
#     #         base = key2results[0][key]
#     #
#     #         line.append(iops_frmt.format(base))
#     #
#     #         for test_results in key2results[1:]:
#     #             val = test_results.get(key)
#     #             if val is None:
#     #                 line.append("-")
#     #             elif base['iops'] == 0:
#     #                 line.append("Nan")
#     #             else:
#     #                 prc_val = {'dev': val['dev'], 'conf': val['conf']}
#     #                 prc_val['iops'] = int(100 * val['iops'] / base['iops'])
#     #                 line.append(iops_frmt.format(prc_val))
#     #
#     #         line.append(base['bw'])
#     #
#     #         for test_results in key2results[1:]:
#     #             val = test_results.get(key)
#     #             if val is None:
#     #                 line.append("-")
#     #             elif base['bw'] == 0:
#     #                 line.append("Nan")
#     #             else:
#     #                 line.append(int(100 * val['bw'] / base['bw']))
#     #
#     #         for test_results in key2results:
#     #             val = test_results.get(key)
#     #             if val is None:
#     #                 line.append("-")
#     #             else:
#     #                 line.append("{0[lat_50]} - {0[lat_95]}".format(val))
#     #
#     #         tab.add_row(line)
#     #
#     #     tab.set_cols_align(allign)
#     #     return tab.draw()
#
#
# # READ_IOPS_DISCSTAT_POS = 3
# # WRITE_IOPS_DISCSTAT_POS = 7
# #
# #
# # def load_sys_log_file(ftype: str, fname: str) -> TimeSeriesValue:
# #     assert ftype == 'iops'
# #     pval = None
# #     with open(fname) as fd:
# #         iops = []
# #         for ln in fd:
# #             params = ln.split()
# #             cval = int(params[WRITE_IOPS_DISCSTAT_POS]) + \
# #                 int(params[READ_IOPS_DISCSTAT_POS])
# #             if pval is not None:
# #                 iops.append(cval - pval)
# #             pval = cval
# #
# #     vals = [(idx * 1000, val) for idx, val in enumerate(iops)]
# #     return TimeSeriesValue(vals)
# #
# #
# # def load_test_results(folder: str, run_num: int) -> 'FioRunResult':
# #     res = {}
# #     params = None
# #
# #     fn = os.path.join(folder, str(run_num) + '_params.yaml')
# #     params = yaml.load(open(fn).read())
# #
# #     conn_ids_set = set()
# #     rr = r"{}_(?P<conn_id>.*?)_(?P<type>[^_.]*)\.\d+\.log$".format(run_num)
# #     for fname in os.listdir(folder):
# #         rm = re.match(rr, fname)
# #         if rm is None:
# #             continue
# #
# #         conn_id_s = rm.group('conn_id')
# #         conn_id = conn_id_s.replace('_', ':')
# #         ftype = rm.group('type')
# #
# #         if ftype not in ('iops', 'bw', 'lat'):
# #             continue
# #
# #         ts = load_fio_log_file(os.path.join(folder, fname))
# #         res.setdefault(ftype, {}).setdefault(conn_id, []).append(ts)
# #
# #         conn_ids_set.add(conn_id)
# #
# #     rr = r"{}_(?P<conn_id>.*?)_(?P<type>[^_.]*)\.sys\.log$".format(run_num)
# #     for fname in os.listdir(folder):
# #         rm = re.match(rr, fname)
# #         if rm is None:
# #             continue
# #
# #         conn_id_s = rm.group('conn_id')
# #         conn_id = conn_id_s.replace('_', ':')
# #         ftype = rm.group('type')
# #
# #         if ftype not in ('iops', 'bw', 'lat'):
# #             continue
# #
# #         ts = load_sys_log_file(ftype, os.path.join(folder, fname))
# #         res.setdefault(ftype + ":sys", {}).setdefault(conn_id, []).append(ts)
# #
# #         conn_ids_set.add(conn_id)
# #
# #     mm_res = {}
# #
# #     if len(res) == 0:
# #         raise ValueError("No data was found")
# #
# #     for key, data in res.items():
# #         conn_ids = sorted(conn_ids_set)
# #         awail_ids = [conn_id for conn_id in conn_ids if conn_id in data]
# #         matr = [data[conn_id] for conn_id in awail_ids]
# #         mm_res[key] = MeasurementMatrix(matr, awail_ids)
# #
# #     raw_res = {}
# #     for conn_id in conn_ids:
# #         fn = os.path.join(folder, "{0}_{1}_rawres.json".format(run_num, conn_id_s))
# #
# #         # remove message hack
# #         fc = "{" + open(fn).read().split('{', 1)[1]
# #         raw_res[conn_id] = json.loads(fc)
# #
# #     fio_task = FioJobSection(params['name'])
# #     fio_task.vals.update(params['vals'])
# #
# #     config = TestConfig('io', params, None, params['nodes'], folder, None)
# #     return FioRunResult(config, fio_task, mm_res, raw_res, params['intervals'], run_num)
# #
#
# # class DiskPerfInfo:
# #     def __init__(self, name: str, summary: str, params: Dict[str, Any], testnodes_count: int) -> None:
# #         self.name = name
# #         self.bw = None
# #         self.iops = None
# #         self.lat = None
# #         self.lat_50 = None
# #         self.lat_95 = None
# #         self.lat_avg = None
# #
# #         self.raw_bw = []
# #         self.raw_iops = []
# #         self.raw_lat = []
# #
# #         self.params = params
# #         self.testnodes_count = testnodes_count
# #         self.summary = summary
# #
# #         self.sync_mode = get_test_sync_mode(self.params['vals'])
# #         self.concurence = self.params['vals'].get('numjobs', 1)
# #
# #
# # class IOTestResults:
# #     def __init__(self, suite_name: str, fio_results: 'FioRunResult', log_directory: str):
# #         self.suite_name = suite_name
# #         self.fio_results = fio_results
# #         self.log_directory = log_directory
# #
# #     def __iter__(self):
# #         return iter(self.fio_results)
# #
# #     def __len__(self):
# #         return len(self.fio_results)
# #
# #     def get_yamable(self) -> Dict[str, List[str]]:
# #         items = [(fio_res.summary(), fio_res.idx) for fio_res in self]
# #         return {self.suite_name: [self.log_directory] + items}
#
#
# # class FioRunResult(TestResults):
# #     """
# #     Fio run results
# #     config: TestConfig
# #     fio_task: FioJobSection
# #     ts_results: {str: MeasurementMatrix[TimeSeriesValue]}
# #     raw_result: ????
# #     run_interval:(float, float) - test tun time, used for sensors
# #     """
# #     def __init__(self, config, fio_task, ts_results, raw_result, run_interval, idx):
# #
# #         self.name = fio_task.name.rsplit("_", 1)[0]
# #         self.fio_task = fio_task
# #         self.idx = idx
# #
# #         self.bw = ts_results['bw']
# #         self.lat = ts_results['lat']
# #         self.iops = ts_results['iops']
# #
# #         if 'iops:sys' in ts_results:
# #             self.iops_sys = ts_results['iops:sys']
# #         else:
# #             self.iops_sys = None
# #
# #         res = {"bw": self.bw,
# #                "lat": self.lat,
# #                "iops": self.iops,
# #                "iops:sys": self.iops_sys}
# #
# #         self.sensors_data = None
# #         self._pinfo = None
# #         TestResults.__init__(self, config, res, raw_result, run_interval)
# #
# #     def get_params_from_fio_report(self):
# #         nodes = self.bw.connections_ids
# #
# #         iops = [self.raw_result[node]['jobs'][0]['mixed']['iops'] for node in nodes]
# #         total_ios = [self.raw_result[node]['jobs'][0]['mixed']['total_ios'] for node in nodes]
# #         runtime = [self.raw_result[node]['jobs'][0]['mixed']['runtime'] / 1000 for node in nodes]
# #         flt_iops = [float(ios) / rtime for ios, rtime in zip(total_ios, runtime)]
# #
# #         bw = [self.raw_result[node]['jobs'][0]['mixed']['bw'] for node in nodes]
# #         total_bytes = [self.raw_result[node]['jobs'][0]['mixed']['io_bytes'] for node in nodes]
# #         flt_bw = [float(tbytes) / rtime for tbytes, rtime in zip(total_bytes, runtime)]
# #
# #         return {'iops': iops,
# #                 'flt_iops': flt_iops,
# #                 'bw': bw,
# #                 'flt_bw': flt_bw}
# #
# #     def summary(self):
# #         return get_test_summary(self.fio_task, len(self.config.nodes))
# #
# #     def summary_tpl(self):
# #         return get_test_summary_tuple(self.fio_task, len(self.config.nodes))
# #
# #     def get_lat_perc_50_95_multy(self):
# #         lat_mks = collections.defaultdict(lambda: 0)
# #         num_res = 0
# #
# #         for result in self.raw_result.values():
# #             num_res += len(result['jobs'])
# #             for job_info in result['jobs']:
# #                 for k, v in job_info['latency_ms'].items():
# #                     if isinstance(k, basestring) and k.startswith('>='):
# #                         lat_mks[int(k[2:]) * 1000] += v
# #                     else:
# #                         lat_mks[int(k) * 1000] += v
# #
# #                 for k, v in job_info['latency_us'].items():
# #                     lat_mks[int(k)] += v
# #
# #         for k, v in lat_mks.items():
# #             lat_mks[k] = float(v) / num_res
# #         return get_lat_perc_50_95(lat_mks)
# #
# #     def disk_perf_info(self, avg_interval=2.0):
# #
# #         if self._pinfo is not None:
# #             return self._pinfo
# #
# #         testnodes_count = len(self.config.nodes)
# #
# #         pinfo = DiskPerfInfo(self.name,
# #                              self.summary(),
# #                              self.params,
# #                              testnodes_count)
# #
# #         def prepare(data, drop=1):
# #             if data is None:
# #                 return data
# #
# #             res = []
# #             for ts_data in data:
# #                 if ts_data.average_interval() < avg_interval:
# #                     ts_data = ts_data.derived(avg_interval)
# #
# #                 # drop last value on bounds
# #                 # as they may contains ranges without activities
# #                 assert len(ts_data.values) >= drop + 1, str(drop) + " " + str(ts_data.values)
# #
# #                 if drop > 0:
# #                     res.append(ts_data.values[:-drop])
# #                 else:
# #                     res.append(ts_data.values)
# #
# #             return res
# #
# #         def agg_data(matr):
# #             arr = sum(matr, [])
# #             min_len = min(map(len, arr))
# #             res = []
# #             for idx in range(min_len):
# #                 res.append(sum(dt[idx] for dt in arr))
# #             return res
# #
# #         pinfo.raw_lat = map(prepare, self.lat.per_vm())
# #         num_th = sum(map(len, pinfo.raw_lat))
# #         lat_avg = [val / num_th for val in agg_data(pinfo.raw_lat)]
# #         pinfo.lat_avg = data_property(lat_avg).average / 1000  # us to ms
# #
# #         pinfo.lat_50, pinfo.lat_95 = self.get_lat_perc_50_95_multy()
# #         pinfo.lat = pinfo.lat_50
# #
# #         pinfo.raw_bw = map(prepare, self.bw.per_vm())
# #         pinfo.raw_iops = map(prepare, self.iops.per_vm())
# #
# #         if self.iops_sys is not None:
# #             pinfo.raw_iops_sys = map(prepare, self.iops_sys.per_vm())
# #             pinfo.iops_sys = data_property(agg_data(pinfo.raw_iops_sys))
# #         else:
# #             pinfo.raw_iops_sys = None
# #             pinfo.iops_sys = None
# #
# #         fparams = self.get_params_from_fio_report()
# #         fio_report_bw = sum(fparams['flt_bw'])
# #         fio_report_iops = sum(fparams['flt_iops'])
# #
# #         agg_bw = agg_data(pinfo.raw_bw)
# #         agg_iops = agg_data(pinfo.raw_iops)
# #
# #         log_bw_avg = average(agg_bw)
# #         log_iops_avg = average(agg_iops)
# #
# #         # update values to match average from fio report
# #         coef_iops = fio_report_iops / float(log_iops_avg)
# #         coef_bw = fio_report_bw / float(log_bw_avg)
# #
# #         bw_log = data_property([val * coef_bw for val in agg_bw])
# #         iops_log = data_property([val * coef_iops for val in agg_iops])
# #
# #         bw_report = data_property([fio_report_bw])
# #         iops_report = data_property([fio_report_iops])
# #
# #         # When IOPS/BW per thread is too low
# #         # data from logs is rounded to match
# #         iops_per_th = sum(sum(pinfo.raw_iops, []), [])
# #         if average(iops_per_th) > 10:
# #             pinfo.iops = iops_log
# #             pinfo.iops2 = iops_report
# #         else:
# #             pinfo.iops = iops_report
# #             pinfo.iops2 = iops_log
# #
# #         bw_per_th = sum(sum(pinfo.raw_bw, []), [])
# #         if average(bw_per_th) > 10:
# #             pinfo.bw = bw_log
# #             pinfo.bw2 = bw_report
# #         else:
# #             pinfo.bw = bw_report
# #             pinfo.bw2 = bw_log
# #
# #         self._pinfo = pinfo
# #
# #         return pinfo
#
# # class TestResult:
# #     """Hold all information for a given test - test info,
# #     sensors data and performance results for test period from all nodes"""
# #     run_id = None  # type: int
# #     test_info = None  # type: Any
# #     begin_time = None  # type: int
# #     end_time = None  # type: int
# #     sensors = None  # Dict[Tuple[str, str, str], TimeSeries]
# #     performance = None  # Dict[Tuple[str, str], TimeSeries]
# #
# #     class TestResults:
# #         """
# #         this class describe test results
# #
# #         config:TestConfig - test config object
# #         params:dict - parameters from yaml file for this test
# #         results:{str:MeasurementMesh} - test results object
# #         raw_result:Any - opaque object to store raw results
# #         run_interval:(float, float) - test tun time, used for sensors
# #         """
# #
# #         def __init__(self,
# #                      config: TestConfig,
# #                      results: Dict[str, Any],
# #                      raw_result: Any,
# #                      run_interval: Tuple[float, float]) -> None:
# #             self.config = config
# #             self.params = config.params
# #             self.results = results
# #             self.raw_result = raw_result
# #             self.run_interval = run_interval
# #
# #         def __str__(self) -> str:
# #             res = "{0}({1}):\n    results:\n".format(
# #                 self.__class__.__name__,
# #                 self.summary())
# #
# #             for name, val in self.results.items():
# #                 res += "        {0}={1}\n".format(name, val)
# #
# #             res += "    params:\n"
# #
# #             for name, val in self.params.items():
# #                 res += "        {0}={1}\n".format(name, val)
# #
# #             return res
# #
# #         def summary(self) -> str:
# #             raise NotImplementedError()
# #             return ""
# #
# #         def get_yamable(self) -> Any:
# #             raise NotImplementedError()
# #             return None
#
#
#
#             # class MeasurementMatrix:
# #     """
# #     data:[[MeasurementResult]] - VM_COUNT x TH_COUNT matrix of MeasurementResult
# #     """
# #     def __init__(self, data, connections_ids):
# #         self.data = data
# #         self.connections_ids = connections_ids
# #
# #     def per_vm(self):
# #         return self.data
# #
# #     def per_th(self):
# #         return sum(self.data, [])
#
#
# # class MeasurementResults:
# #     data = None  # type: List[Any]
# #
# #     def stat(self) -> StatProps:
# #         return data_property(self.data)
# #
# #     def __str__(self) -> str:
# #         return 'TS([' + ", ".join(map(str, self.data)) + '])'
# #
# #
# # class SimpleVals(MeasurementResults):
# #     """
# #     data:[float] - list of values
# #     """
# #     def __init__(self, data: List[float]) -> None:
# #         self.data = data
# #
# #
# # class TimeSeriesValue(MeasurementResults):
# #     """
# #     data:[(float, float, float)] - list of (start_time, lenght, average_value_for_interval)
# #     odata: original values
# #     """
# #     def __init__(self, data: List[Tuple[float, float]]) -> None:
# #         assert len(data) > 0
# #         self.odata = data[:]
# #         self.data = []  # type: List[Tuple[float, float, float]]
# #
# #         cstart = 0.0
# #         for nstart, nval in data:
# #             self.data.append((cstart, nstart - cstart, nval))
# #             cstart = nstart
# #
# #     @property
# #     def values(self) -> List[float]:
# #         return [val[2] for val in self.data]
# #
# #     def average_interval(self) -> float:
# #         return float(sum([val[1] for val in self.data])) / len(self.data)
# #
# #     def skip(self, seconds) -> 'TimeSeriesValue':
# #         nres = []
# #         for start, ln, val in self.data:
# #             nstart = start + ln - seconds
# #             if nstart > 0:
# #                 nres.append([nstart, val])
# #         return self.__class__(nres)
# #
# #     def derived(self, tdelta) -> 'TimeSeriesValue':
# #         end = self.data[-1][0] + self.data[-1][1]
# #         tdelta = float(tdelta)
# #
# #         ln = end / tdelta
# #
# #         if ln - int(ln) > 0:
# #             ln += 1
# #
# #         res = [[tdelta * i, 0.0] for i in range(int(ln))]
# #
# #         for start, lenght, val in self.data:
# #             start_idx = int(start / tdelta)
# #             end_idx = int((start + lenght) / tdelta)
# #
# #             for idx in range(start_idx, end_idx + 1):
# #                 rstart = tdelta * idx
# #                 rend = tdelta * (idx + 1)
# #
# #                 intersection_ln = min(rend, start + lenght) - max(start, rstart)
# #                 if intersection_ln > 0:
# #                     try:
# #                         res[idx][1] += val * intersection_ln / tdelta
# #                     except IndexError:
# #                         raise
# #
# #         return self.__class__(res)
#
#
# def console_report_stage(ctx: TestRun) -> None:
#     # TODO(koder): load data from storage
#     raise NotImplementedError("...")
#     # first_report = True
#     # text_rep_fname = ctx.config.text_report_file
#     #
#     # with open(text_rep_fname, "w") as fd:
#     #     for tp, data in ctx.results.items():
#     #         if 'io' == tp and data is not None:
#     #             rep_lst = []
#     #             for result in data:
#     #                 rep_lst.append(
#     #                     IOPerfTest.format_for_console(list(result)))
#     #             rep = "\n\n".join(rep_lst)
#     #         elif tp in ['mysql', 'pgbench'] and data is not None:
#     #             rep = MysqlTest.format_for_console(data)
#     #         elif tp == 'omg':
#     #             rep = OmgTest.format_for_console(data)
#     #         else:
#     #             logger.warning("Can't generate text report for " + tp)
#     #             continue
#     #
#     #         fd.write(rep)
#     #         fd.write("\n")
#     #
#     #         if first_report:
#     #             logger.info("Text report were stored in " + text_rep_fname)
#     #             first_report = False
#     #
#     #         print("\n" + rep + "\n")
#
#
# # def test_load_report_stage(cfg: Config, ctx: TestRun) -> None:
# #     load_rep_fname = cfg.load_report_file
# #     found = False
# #     for idx, (tp, data) in enumerate(ctx.results.items()):
# #         if 'io' == tp and data is not None:
# #             if found:
# #                 logger.error("Making reports for more than one " +
# #                              "io block isn't supported! All " +
# #                              "report, except first are skipped")
# #                 continue
# #             found = True
# #             report.make_load_report(idx, cfg['results'], load_rep_fname)
# #
# #
#
# # def html_report_stage(ctx: TestRun) -> None:
#     # TODO(koder): load data from storage
#     # raise NotImplementedError("...")
#     # html_rep_fname = cfg.html_report_file
#     # found = False
#     # for tp, data in ctx.results.items():
#     #     if 'io' == tp and data is not None:
#     #         if found or len(data) > 1:
#     #             logger.error("Making reports for more than one " +
#     #                          "io block isn't supported! All " +
#     #                          "report, except first are skipped")
#     #             continue
#     #         found = True
#     #         report.make_io_report(list(data[0]),
#     #                               cfg.get('comment', ''),
#     #                               html_rep_fname,
#     #                               lab_info=ctx.nodes)
#
# #
# # def load_data_from_path(test_res_dir: str) -> Mapping[str, List[Any]]:
# #     files = get_test_files(test_res_dir)
# #     raw_res = yaml_load(open(files['raw_results']).read())
# #     res = collections.defaultdict(list)
# #
# #     for tp, test_lists in raw_res:
# #         for tests in test_lists:
# #             for suite_name, suite_data in tests.items():
# #                 result_folder = suite_data[0]
# #                 res[tp].append(TOOL_TYPE_MAPPER[tp].load(suite_name, result_folder))
# #
# #     return res
# #
# #
# # def load_data_from_path_stage(var_dir: str, _, ctx: TestRun) -> None:
# #     for tp, vals in load_data_from_path(var_dir).items():
# #         ctx.results.setdefault(tp, []).extend(vals)
# #
# #
# # def load_data_from(var_dir: str) -> Callable[[TestRun], None]:
# #     return functools.partial(load_data_from_path_stage, var_dir)
