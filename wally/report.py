import os
import abc
import logging
import collections
from collections import defaultdict
from typing import Dict, Any, Iterator, Tuple, cast, List, Set, Optional, Union, Type, Iterable

import numpy
import scipy.stats
from statsmodels.tsa.stattools import adfuller

import xmlbuilder3

import wally

from cephlib import html
from cephlib.units import b2ssize, b2ssize_10, unit_conversion_coef, unit_conversion_coef_f
from cephlib.statistic import calc_norm_stat_props
from cephlib.storage_selectors import sum_sensors, find_sensors_to_2d, update_storage_selector, DevRoles
from cephlib.wally_storage import find_nodes_by_roles
from cephlib.plot import (plot_simple_bars, plot_hmap_from_2d, plot_lat_over_time, plot_simple_over_time,
                          plot_histo_heatmap, plot_v_over_time, plot_hist, plot_dots_with_regression)
from cephlib.numeric_types import ndarray2d
from cephlib.node import NodeRole

from .utils import STORAGE_ROLES
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .result_classes import IWallyStorage
from .result_classes import DataSource, TimeSeries, SuiteConfig
from .suits.io.fio import FioTest, FioJobConfig
from .suits.io.fio_job import FioJobParams
from .suits.job import JobConfig
from .data_selectors import get_aggregated, AGG_TAG
from .report_profiles import (DefStyleProfile, DefColorProfile, StyleProfile, ColorProfile,
                              default_format, io_chart_format)
from .plot import io_chart
from .resources import ResourceNames, get_resources_usage, make_iosum, get_cluster_cpu_load
from .console_report import get_console_report_table, console_report_headers, console_report_align, Texttable


logger = logging.getLogger("wally")


# ----------------  CONSTS ---------------------------------------------------------------------------------------------


DEBUG = False


# --------------  AGGREGATION AND STAT FUNCTIONS  ----------------------------------------------------------------------

LEVEL_SENSORS = {("block-io", "io_queue"), ("system-cpu", "procs_blocked"), ("system-cpu", "procs_queue")}


def is_level_sensor(sensor: str, metric: str) -> bool:
    """Returns True if sensor measure level of any kind, E.g. queue depth."""
    return (sensor, metric) in LEVEL_SENSORS


def is_delta_sensor(sensor: str, metric: str) -> bool:
    """Returns True if sensor provides deltas for cumulative value. E.g. io completed in given period"""
    return not is_level_sensor(sensor, metric)


# def get_idle_load(rstorage: ResultStorage, *args, **kwargs) -> float:
#     if 'idle' not in rstorage.storage:
#         return 0.0
#     idle_time = rstorage.storage.get('idle')
#     ssum = summ_sensors(rstorage, time_range=idle_time, *args, **kwargs)
#     return numpy.average(ssum)


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
        self.data = []  # type: List[List[str]]

    def add_line(self, values: List[str]) -> None:
        self.data.append(values)

    def html(self) -> str:
        return html.table("", self.header, self.data)


class Menu1st:
    summary = "Summary"
    per_job = "Per Job"
    engineering = "Engineering"
    engineering_per_job = "Engineering per job"
    order = [summary, per_job, engineering, engineering_per_job]


class Menu2ndEng:
    summary = "Summary"
    iops_time = "IOPS(time)"
    hist = "IOPS/lat overall histogram"
    lat_time = "Lat(time)"
    resource_regression = "Resource usage LR"
    order = [summary, iops_time, hist, lat_time, resource_regression]


class Menu2ndSumm:
    summary = "Summary"
    io_lat_qd = "IO & Lat vs QD"
    resources_usage_qd = "Resource usage"
    order = [summary, io_lat_qd, resources_usage_qd]


menu_1st_order = [Menu1st.summary, Menu1st.engineering, Menu1st.per_job, Menu1st.engineering_per_job]


#  --------------------  REPORTS  --------------------------------------------------------------------------------------

class ReporterBase:
    def __init__(self, rstorage: IWallyStorage, style: StyleProfile, colors: ColorProfile) -> None:
        self.style = style
        self.colors = colors
        self.rstorage = rstorage

    def plt(self, func, ds: DataSource, *args, **kwargs) -> str:
        return func(self.rstorage, self.style, self.colors, ds, *args, **kwargs)


class SuiteReporter(ReporterBase, metaclass=abc.ABCMeta):
    suite_types = set()  # type: Set[str]

    @abc.abstractmethod
    def get_divs(self, suite: SuiteConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        pass


class JobReporter(ReporterBase, metaclass=abc.ABCMeta):
    suite_type = set()  # type: Set[str]

    @abc.abstractmethod
    def get_divs(self, suite: SuiteConfig, job: JobConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        pass


# # Linearization report
# class IOPSBsize(SuiteReporter):
#     """Creates graphs, which show how IOPS and Latency depend on block size"""
#
#


class StoragePerfSummary:
    iops_units = "KiBps"
    bw_units = "Bps"
    NO_VAL = -1

    def __init__(self) -> None:
        self.rw_iops_10ms = self.NO_VAL  # type: int
        self.rw_iops_30ms = self.NO_VAL  # type: int
        self.rw_iops_100ms = self.NO_VAL  # type: int

        self.rr_iops_10ms = self.NO_VAL  # type: int
        self.rr_iops_30ms = self.NO_VAL  # type: int
        self.rr_iops_100ms = self.NO_VAL  # type: int

        self.bw_write_max = self.NO_VAL  # type: int
        self.bw_read_max = self.NO_VAL  # type: int

        self.bw = None  # type: Optional[float]
        self.read_iops = None  # type: Optional[float]
        self.write_iops = None  # type: Optional[float]


def get_performance_summary(storage: IWallyStorage, suite: SuiteConfig,
                            hboxes: int, large_blocks: int) -> Tuple[StoragePerfSummary, StoragePerfSummary]:

    psum95 = StoragePerfSummary()
    psum50 = StoragePerfSummary()

    for job in storage.iter_job(suite):
        if isinstance(job, FioJobConfig):
            fjob = cast(FioJobConfig, job)
            io_sum = make_iosum(storage, suite, job, hboxes)

            bw_avg = io_sum.bw.average * unit_conversion_coef(io_sum.bw.units, StoragePerfSummary.bw_units)

            if fjob.bsize < large_blocks:
                lat_95_ms = io_sum.lat.perc_95 * unit_conversion_coef(io_sum.lat.units, 'ms')
                lat_50_ms = io_sum.lat.perc_50 * unit_conversion_coef(io_sum.lat.units, 'ms')

                iops_avg = io_sum.bw.average * unit_conversion_coef(io_sum.bw.units, StoragePerfSummary.iops_units)
                iops_avg /= fjob.bsize

                if fjob.oper == 'randwrite' and fjob.sync_mode == 'd':
                    for lat, field in [(10, 'rw_iops_10ms'), (30, 'rw_iops_30ms'), (100, 'rw_iops_100ms')]:
                        if lat_95_ms <= lat:
                            setattr(psum95, field, max(getattr(psum95, field), iops_avg))
                        if lat_50_ms <= lat:
                            setattr(psum50, field, max(getattr(psum50, field), iops_avg))

                if fjob.oper == 'randread' and fjob.sync_mode == 'd':
                    for lat, field in [(10, 'rr_iops_10ms'), (30, 'rr_iops_30ms'), (100, 'rr_iops_100ms')]:
                        if lat_95_ms <= lat:
                            setattr(psum95, field, max(getattr(psum95, field), iops_avg))
                        if lat_50_ms <= lat:
                            setattr(psum50, field, max(getattr(psum50, field), iops_avg))
            elif fjob.sync_mode == 'd':
                if fjob.oper in ('randwrite', 'write'):
                    psum50.bw_write_max = max(psum50.bw_write_max, bw_avg)
                elif fjob.oper in ('randread', 'read'):
                    psum50.bw_read_max = max(psum50.bw_read_max, bw_avg)

    return psum50, psum95


# Main performance report
class PerformanceSummary(SuiteReporter):
    """Aggregated summary for storage"""
    def get_divs(self, suite: SuiteConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        psum50, psum95 = get_performance_summary(self.rstorage, suite, self.style.hist_boxes, self.style.large_blocks)

        caption = "Storage summary report"
        res = html.H3(html.center(caption))

        headers = ["Mode", "Stats", "Explanation"]
        align = ['left', 'right', "left"]
        data = []

        if psum95.rr_iops_10ms != psum95.NO_VAL or psum95.rr_iops_30ms != psum95.NO_VAL or \
                psum95.rr_iops_100ms != psum95.NO_VAL:
            data.append("Average random read IOPS for small blocks")

        if psum95.rr_iops_10ms != psum95.NO_VAL:
            data.append(("Database", b2ssize_10(psum95.rr_iops_10ms), "Latency 95th percentile < 10ms"))
        if psum95.rr_iops_30ms != psum95.NO_VAL:
            data.append(("File system", b2ssize_10(psum95.rr_iops_30ms), "Latency 95th percentile < 30ms"))
        if psum95.rr_iops_100ms != psum95.NO_VAL:
            data.append(("File server", b2ssize_10(psum95.rr_iops_100ms), "Latency 95th percentile < 100ms"))

        if psum95.rw_iops_10ms != psum95.NO_VAL or psum95.rw_iops_30ms != psum95.NO_VAL or \
                psum95.rw_iops_100ms != psum95.NO_VAL:
            data.append("Average random write IOPS for small blocks")

        if psum95.rw_iops_10ms != psum95.NO_VAL:
            data.append(("Database", b2ssize_10(psum95.rw_iops_10ms), "Latency 95th percentile < 10ms"))
        if psum95.rw_iops_30ms != psum95.NO_VAL:
            data.append(("File system", b2ssize_10(psum95.rw_iops_30ms), "Latency 95th percentile < 30ms"))
        if psum95.rw_iops_100ms != psum95.NO_VAL:
            data.append(("File server", b2ssize_10(psum95.rw_iops_100ms), "Latency 95th percentile < 100ms"))

        if psum50.bw_write_max != psum50.NO_VAL or psum50.bw_read_max != psum50.NO_VAL:
            data.append("Average sequention IO")

        if psum50.bw_write_max != psum95.NO_VAL:
            data.append(("Write", b2ssize(psum50.bw_write_max) + psum50.bw_units,
                         "Large blocks (>={}KiB)".format(self.style.large_blocks)))
        if psum50.bw_read_max != psum95.NO_VAL:
            data.append(("Read", b2ssize(psum50.bw_read_max) + psum50.bw_units,
                         "Large blocks (>={}KiB)".format(self.style.large_blocks)))

        res += html.center(html.table("Performance", headers, data, align=align))
        yield Menu1st.summary, Menu2ndSumm.summary, HTMLBlock(res)


# # Node load over test time
# class NodeLoad(SuiteReporter):
#     """IOPS/latency during test"""

# # Ceph operation breakout report
# class CephClusterSummary(SuiteReporter):


# Main performance report
class IOQD(SuiteReporter):
    """Creates graph, which show how IOPS and Latency depend on QD"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        ts_map = defaultdict(list)  # type: Dict[FioJobParams, List[Tuple[SuiteConfig, FioJobConfig]]]
        str_summary = {}  # type: Dict[FioJobParams, Tuple[str, str]]

        for job in self.rstorage.iter_job(suite):
            fjob = cast(FioJobConfig, job)
            fjob_no_qd = cast(FioJobParams, fjob.params.copy(qd=None))
            str_summary[fjob_no_qd] = (fjob_no_qd.summary, fjob_no_qd.long_summary)
            ts_map[fjob_no_qd].append((suite, fjob))

        caption = "IOPS, bandwith, and latency as function of parallel IO request count (QD)"
        yield Menu1st.summary, Menu2ndSumm.io_lat_qd, HTMLBlock(html.H3(html.center(caption)))

        for tpl, suites_jobs in ts_map.items():
            if len(suites_jobs) >= self.style.min_iops_vs_qd_jobs:
                iosums = [make_iosum(self.rstorage, suite, job, self.style.hist_boxes) for suite, job in suites_jobs]
                iosums.sort(key=lambda x: x.qd)
                summary, summary_long = str_summary[tpl]

                ds = DataSource(suite_id=suite.storage_id,
                                job_id=summary,
                                node_id=AGG_TAG,
                                sensor="fio",
                                dev=AGG_TAG,
                                metric="io_over_qd",
                                tag=io_chart_format)

                fpath = self.plt(io_chart, ds, title=summary_long, legend="IOPS/BW", iosums=iosums)
                yield Menu1st.summary, Menu2ndSumm.io_lat_qd, HTMLBlock(html.img(fpath))


class ResourceQD(SuiteReporter):
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        qd_grouped_jobs = {}  # type: Dict[FioJobParams, List[FioJobConfig]]
        test_nc = len(list(find_nodes_by_roles(self.rstorage.storage, ['testnode'])))
        for job in self.rstorage.iter_job(suite):
            fjob = cast(FioJobConfig, job)
            if fjob.bsize != 4:
                continue

            fjob_no_qd = cast(FioJobParams, fjob.params.copy(qd=None))
            qd_grouped_jobs.setdefault(fjob_no_qd, []).append(fjob)

        yield Menu1st.summary, Menu2ndSumm.resources_usage_qd, HTMLBlock(html.center(html.H3("Resource usage summary")))

        for jc_no_qd, jobs in sorted(qd_grouped_jobs.items()):
            cpu_usage2qd = {}
            for job in jobs:
                usage, iops_ok = get_resources_usage(suite, job, self.rstorage, hist_boxes=self.style.hist_boxes,
                                                     large_block=self.style.large_blocks)

                if iops_ok:
                    cpu_usage2qd[job.qd] = usage[ResourceNames.storage_cpu_s]

            if len(cpu_usage2qd) < StyleProfile.min_iops_vs_qd_jobs:
                continue

            labels, vals, errs = zip(*((l, avg, dev) for l, (_, avg, dev) in sorted(cpu_usage2qd.items())))

            if test_nc == 1:
                labels = list(map(str, labels))
            else:
                labels = ["{} * {}".format(label, test_nc) for label in labels]

            ds = DataSource(suite_id=suite.storage_id,
                            job_id=jc_no_qd.summary,
                            node_id="cluster",
                            sensor=AGG_TAG,
                            dev='cpu',
                            metric="cpu_for_iop",
                            tag=io_chart_format)

            title = "CPU time per IOP, " + jc_no_qd.long_summary
            fpath = self.plt(plot_simple_bars, ds, title, labels, vals, errs,
                             xlabel="CPU core time per IOP",
                             ylabel="QD * Test nodes" if test_nc != 1 else "QD",
                             x_formatter=(lambda x, pos: b2ssize_10(x) + 's'),
                             one_point_zero_line=False)

            yield Menu1st.summary, Menu2ndSumm.resources_usage_qd, HTMLBlock(html.img(fpath))


def get_resources_usage2(suite: SuiteConfig, job: JobConfig, rstorage: IWallyStorage,
                         roles, sensor, metric, test_metric, agg_window: int = 5) -> ndarray2d:
    assert test_metric == 'iops'
    fjob = cast(FioJobConfig, job)
    bw = get_aggregated(rstorage, suite.storage_id, job.storage_id, "bw", job.reliable_info_range_s)
    io_transfered = bw.data * unit_conversion_coef_f(bw.units, "Bps")
    ops_done = io_transfered / (fjob.bsize * unit_conversion_coef_f("KiBps", "Bps"))
    nodes = [node for node in rstorage.load_nodes() if node.roles.intersection(STORAGE_ROLES)]

    if sensor == 'system-cpu':
        assert metric == 'used'
        core_count = None
        for node in nodes:
            if core_count is None:
                core_count = sum(cores for _, cores in node.hw_info.cpus)
            else:
                assert core_count == sum(cores for _, cores in node.hw_info.cpus)
        cpu_ts = get_cluster_cpu_load(rstorage, roles, job.reliable_info_range_s)
        metric_data = (1.0 - (cpu_ts['idle'].data + cpu_ts['iowait'].data) / cpu_ts['total'].data) * core_count
    else:
        metric_data = sum_sensors(rstorage, job.reliable_info_range_s,
                                  node_id=[node.node_id for node in nodes], sensor=sensor, metric=metric)

    res = []
    for pos in range(0, len(ops_done) - agg_window, agg_window):
        pe = pos + agg_window
        res.append((numpy.average(ops_done[pos: pe]), numpy.average(metric_data.data[pos: pe])))

    return res


class ResourceConsumptionSummary(SuiteReporter):
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        vs = 'iops'
        for job_tp in ('rwd4', 'rrd4'):
            for sensor_metric in ('net-io.send_packets', 'system-cpu.used'):
                sensor, metric = sensor_metric.split(".")
                usage = []
                for job in self.rstorage.iter_job(suite):
                    if job_tp in job.summary:
                        usage.extend(get_resources_usage2(suite, job, self.rstorage, STORAGE_ROLES,
                                                          sensor=sensor, metric=metric, test_metric=vs))

                if not usage:
                    continue

                iops, cpu = zip(*usage)
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(iops, cpu)
                x = numpy.array([0.0, max(iops) * 1.1])

                ds = DataSource(suite_id=suite.storage_id,
                                job_id=job_tp,
                                node_id="storage",
                                sensor='usage-regression',
                                dev=AGG_TAG,
                                metric=sensor_metric + '.VS.' + vs,
                                tag=default_format)

                fname = self.plt(plot_dots_with_regression, ds,
                                 "{}::{}.{}".format(job_tp, sensor_metric, vs),
                                 x=iops, y=cpu,
                                 xlabel=vs,
                                 ylabel=sensor_metric,
                                 x_approx=x, y_approx=intercept + slope * x)

                yield Menu1st.engineering, Menu2ndEng.resource_regression, HTMLBlock(html.img(fname))


class EngineeringSummary(SuiteReporter):
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        tbl = [line for line in get_console_report_table(suite, self.rstorage) if line is not Texttable.HLINE]
        align = [{'l': 'left', 'r': 'right'}[al] for al in console_report_align]
        res = html.center(html.table("Test results", console_report_headers, tbl, align=align))
        yield Menu1st.engineering, Menu2ndEng.summary, HTMLBlock(res)


class StatInfo(JobReporter):
    """Statistic info for job results"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, job: JobConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:

        fjob = cast(FioJobConfig, job)
        io_sum = make_iosum(self.rstorage, suite, fjob, self.style.hist_boxes)

        caption = "Test summary - " + job.params.long_summary
        test_nc = len(list(find_nodes_by_roles(self.rstorage.storage, ['testnode'])))
        if test_nc > 1:
            caption += " * {} nodes".format(test_nc)

        res = html.H3(html.center(caption))
        stat_data_headers = ["Name",
                             "Total done",
                             "Average ~ Dev",
                             "Conf interval",
                             "Mediana",
                             "Mode",
                             "Kurt / Skew",
                             "95%",
                             "99%",
                             "ADF test"]

        align = ['left'] + ['right'] * (len(stat_data_headers) - 1)

        bw_units = "B"
        bw_target_units = bw_units + 'ps'
        bw_coef = unit_conversion_coef_f(io_sum.bw.units, bw_target_units)

        adf_v, *_1, stats, _2 = adfuller(io_sum.bw.data)

        for v in ("1%", "5%", "10%"):
            if adf_v <= stats[v]:
                ad_test = v
                break
        else:
            ad_test = "Failed"

        bw_data = ["Bandwidth",
                   b2ssize(io_sum.bw.data.sum() * bw_coef) + bw_units,
                   "{}{} ~ {}{}".format(b2ssize(io_sum.bw.average * bw_coef), bw_target_units,
                                        b2ssize(io_sum.bw.deviation * bw_coef), bw_target_units),
                   b2ssize(io_sum.bw.confidence * bw_coef) + bw_target_units,
                   b2ssize(io_sum.bw.perc_50 * bw_coef) + bw_target_units,
                   "-",
                   "{:.2f} / {:.2f}".format(io_sum.bw.kurt, io_sum.bw.skew),
                   b2ssize(io_sum.bw.perc_5 * bw_coef) + bw_target_units,
                   b2ssize(io_sum.bw.perc_1 * bw_coef) + bw_target_units,
                   ad_test]

        stat_data = [bw_data]

        if fjob.bsize < StyleProfile.large_blocks:
            iops_coef = unit_conversion_coef_f(io_sum.bw.units, 'KiBps') / fjob.bsize
            iops_data = ["IOPS",
                         b2ssize_10(io_sum.bw.data.sum() * iops_coef),
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
            lat_coef = unit_conversion_coef_f(io_sum.lat.units, lat_target_unit)
            # latency
            lat_data = ["Latency",
                        "-",
                        "-",
                        "-",
                        b2ssize_10(io_sum.lat.perc_50 * lat_coef) + lat_target_unit,
                        "-",
                        "-",
                        b2ssize_10(io_sum.lat.perc_95 * lat_coef) + lat_target_unit,
                        b2ssize_10(io_sum.lat.perc_99 * lat_coef) + lat_target_unit,
                        '-']

            # sensor usage
            stat_data.extend([iops_data, lat_data])

        res += html.center(html.table("Test results", stat_data_headers, stat_data, align=align))
        yield Menu1st.per_job, job.summary, HTMLBlock(res)


class Resources(JobReporter):
    """Statistic info for job results"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, job: JobConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:

        records, iops_ok = get_resources_usage(suite, job, self.rstorage,
                                               large_block=self.style.large_blocks,
                                               hist_boxes=self.style.hist_boxes)

        table_structure = [
            "Service provided",
            (ResourceNames.io_made, ResourceNames.data_tr),
            "Test nodes total load",
            (ResourceNames.test_send_pkt, ResourceNames.test_send),
            (ResourceNames.test_recv_pkt, ResourceNames.test_recv),
            (ResourceNames.test_net_pkt, ResourceNames.test_net),
            (ResourceNames.test_write_iop, ResourceNames.test_write),
            (ResourceNames.test_read_iop, ResourceNames.test_read),
            (ResourceNames.test_iop, ResourceNames.test_rw),
            "Storage nodes resource consumed",
            (ResourceNames.storage_send_pkt, ResourceNames.storage_send),
            (ResourceNames.storage_recv_pkt, ResourceNames.storage_recv),
            (ResourceNames.storage_net_pkt, ResourceNames.storage_net),
            (ResourceNames.storage_write_iop, ResourceNames.storage_write),
            (ResourceNames.storage_read_iop, ResourceNames.storage_read),
            (ResourceNames.storage_iop, ResourceNames.storage_rw),
            (ResourceNames.storage_cpu_s, ResourceNames.storage_cpu_s_b),
        ]  # type: List[Union[str, Tuple[Optional[str], ...]]]

        if not iops_ok:
            table_structure2 = []  # type: List[Union[Tuple[str, ...], str]]
            for line in table_structure:
                if isinstance(line, str):
                    table_structure2.append(line)
                else:
                    assert len(line) == 2
                    table_structure2.append((line[1],))
            table_structure = table_structure2

        yield Menu1st.per_job, job.summary, HTMLBlock(html.H3(html.center("Resources usage")))

        doc = xmlbuilder3.XMLBuilder("table",
                                     **{"class": "table table-bordered table-striped table-condensed table-hover",
                                        "style": "width: auto;"})

        with doc.thead:
            with doc.tr:
                [doc.th(header) for header in ["Resource", "Usage count", "To service"] * (2 if iops_ok else 1)]

        cols = 6 if iops_ok else 3
        col_per_tp = 3

        short_name = {
            name: (name if name in {ResourceNames.io_made, ResourceNames.data_tr}
                   else " ".join(name.split()[2:]).capitalize())
            for name in records.keys()
        }

        short_name[ResourceNames.storage_cpu_s] = "CPU core (s/IOP)"
        short_name[ResourceNames.storage_cpu_s_b] = "CPU core (s/B)"

        with doc.tbody:
            with doc.tr:
                if iops_ok:
                    doc.td(colspan=str(col_per_tp)).center.b("Operations")
                doc.td(colspan=str(col_per_tp)).center.b("Bytes")

            for line in table_structure:
                with doc.tr:
                    if isinstance(line, str):
                        with doc.td(colspan=str(cols)):
                            doc.center.b(line)
                    else:
                        for name in line:
                            if name is None:
                                doc.td("-", colspan=str(col_per_tp))
                                continue

                            amount_s, avg, dev = records[name]

                            if name in (ResourceNames.storage_cpu_s, ResourceNames.storage_cpu_s_b) and avg is not None:
                                if dev is None:
                                    rel_val_s = b2ssize_10(avg) + 's'
                                else:
                                    dev_s = str(int(dev * 100 / avg)) + "%" if avg > 1E-9 else b2ssize_10(dev) + 's'
                                    rel_val_s = "{}s ~ {}".format(b2ssize_10(avg), dev_s)
                            else:
                                if avg is None:
                                    rel_val_s = '-'
                                else:
                                    avg_s = int(avg) if avg > 10 else '{:.1f}'.format(avg)
                                    if dev is None:
                                        rel_val_s = avg_s
                                    else:
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

        iop_names = [ResourceNames.test_write_iop, ResourceNames.test_read_iop, ResourceNames.test_iop,
                     ResourceNames.storage_write_iop, ResourceNames.storage_read_iop, ResourceNames.storage_iop]

        bytes_names = [ResourceNames.test_write, ResourceNames.test_read, ResourceNames.test_rw,
                       ResourceNames.test_send, ResourceNames.test_recv, ResourceNames.test_net,
                       ResourceNames.storage_write, ResourceNames.storage_read, ResourceNames.storage_rw,
                       ResourceNames.storage_send, ResourceNames.storage_recv,
                       ResourceNames.storage_net] # type: List[str]

        net_pkt_names = [ResourceNames.test_send_pkt, ResourceNames.test_recv_pkt, ResourceNames.test_net_pkt,
                         ResourceNames.storage_send_pkt, ResourceNames.storage_recv_pkt, ResourceNames.storage_net_pkt]

        pairs = [("bytes", bytes_names)]
        if iops_ok:
            pairs.insert(0, ('iop', iop_names))
            pairs.append(('Net packets per IOP', net_pkt_names))

        yield Menu1st.per_job, job.summary, \
            HTMLBlock(html.H3(html.center("Resource consumption per service provided")))

        for tp, names in pairs:
            vals = []  # type: List[float]
            devs = []  # type: List[float]
            avail_names = []  # type: List[str]
            for name in names:
                if name in records:
                    avail_names.append(name)
                    _, avg, dev = records[name]

                    if dev is None:
                        dev = 0

                    vals.append(avg)
                    devs.append(dev)

            # synchronously sort values and names, values is a key
            vals, names, devs = map(list, zip(*sorted(zip(vals, names, devs)))) # type: ignore

            ds = DataSource(suite_id=suite.storage_id,
                            job_id=job.storage_id,
                            node_id=AGG_TAG,
                            sensor='resources',
                            dev=AGG_TAG,
                            metric=tp.replace(' ', "_") + '2service_bar',
                            tag=default_format)

            fname = self.plt(plot_simple_bars, ds, tp.capitalize(),
                             [name.replace(" nodes", "") for name in names],
                             vals, devs)

            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fname))


class BottleNeck(JobReporter):
    """Statistic info for job results"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, job: JobConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:

        nodes = list(find_nodes_by_roles(self.rstorage.storage, STORAGE_ROLES))

        sensor = 'block-io'
        metric = 'io_queue'
        bn_val = 16

        for node_id in nodes:
            bn = 0
            tot = 0
            for ds in self.rstorage.iter_sensors(node_id=node_id, sensor=sensor, metric=metric):
                if ds.dev in ('sdb', 'sdc', 'sdd', 'sde'):
                    ts = self.rstorage.get_sensor(ds, job.reliable_info_range_s)
                    bn += (ts.data > bn_val).sum()
                    tot += len(ts.data)
            print(node_id, bn, tot)

        yield Menu1st.per_job, job.summary, HTMLBlock("")


# CPU load
class CPULoadPlot(JobReporter):
    def get_divs(self, suite: SuiteConfig, job: JobConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:

        # plot CPU time
        for rt, roles in [('storage', STORAGE_ROLES), ('test', ['testnode'])]:
            cpu_ts = get_cluster_cpu_load(self.rstorage, roles, job.reliable_info_range_s)
            tss = [(name, ts.data * 100 / cpu_ts['total'].data)
                   for name, ts in cpu_ts.items()
                   if name in {'user', 'sys', 'idle', 'iowait'}]


            ds = cpu_ts['idle'].source(job_id=job.storage_id, suite_id=suite.storage_id,
                                       node_id=AGG_TAG, metric='allcpu', tag=rt + '.plt.' + default_format)

            fname = self.plt(plot_simple_over_time, ds, tss=tss, average=True, ylabel="CPU time %",
                             title="{} nodes CPU usage".format(rt.capitalize()),
                             xlabel="Time from test begin")

            yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fname))


class DevRoles:
    client_disk = 'client_disk'
    client_net = 'client_net'
    client_cpu = 'client_cpu'

    storage_disk = 'storage_disk'
    storage_client_net = 'storage_client_net'
    storage_replication_net = 'storage_replication_net'
    storage_cpu = 'storage_disk'
    ceph_storage = 'ceph_storage'
    ceph_journal = 'ceph_journal'

    compute_disk = 'compute_disk'
    compute_net = 'compute_net'
    compute_cpu = 'compute_cpu'


def roles_for_sensors(storage: IWallyStorage) -> Dict[str, List[DataSource]]:
    role2ds = defaultdict(list)

    for node in storage.load_nodes():
        ds = DataSource(node_id=node.node_id)
        if 'ceph-osd' in node.roles:
            for jdev in node.params.get('ceph_journal_devs', []):
                role2ds[DevRoles.ceph_journal].append(ds(dev=jdev))
                role2ds[DevRoles.storage_disk].append(ds(dev=jdev))

            for sdev in node.params.get('ceph_storage_devs', []):
                role2ds[DevRoles.ceph_storage].append(ds(dev=sdev))
                role2ds[DevRoles.storage_disk].append(ds(dev=sdev))

            if node.hw_info:
                for dev in node.hw_info.disks_info:
                    role2ds[DevRoles.storage_disk].append(ds(dev=dev))

        if 'testnode' in node.roles:
            role2ds[DevRoles.client_disk].append(ds(dev='rbd0'))

    return role2ds


def get_sources_for_roles(roles: Iterable[str]) -> List[DataSource]:
    return []


# IO time and QD
class QDIOTimeHeatmap(JobReporter):
    def get_divs(self, suite: SuiteConfig, job: JobConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        trange = (job.reliable_info_range[0] // 1000, job.reliable_info_range[1] // 1000)
        test_nc = len(list(find_nodes_by_roles(self.rstorage.storage, ['testnode'])))

        for dev_role in (DevRoles.ceph_storage, DevRoles.ceph_journal, DevRoles.client_disk):

            caption = "{} IO heatmaps - {}".format(dev_role.capitalize(), cast(FioJobParams, job).params.long_summary)
            if test_nc != 1:
                caption += " * {} nodes".format(test_nc)

            yield Menu1st.engineering_per_job, job.summary, HTMLBlock(html.H3(html.center(caption)))

            # QD heatmap
            ioq2d = find_sensors_to_2d(self.rstorage, trange, dev_role=dev_role, sensor='block-io', metric='io_queue')

            ds = DataSource(suite.storage_id, job.storage_id, AGG_TAG, 'block-io', dev_role,
                            tag="hmap." + default_format)

            fname = self.plt(plot_hmap_from_2d, ds(metric='io_queue'), data2d=ioq2d, xlabel='Time', ylabel="IO QD",
                             title=dev_role.capitalize() + " devs QD", bins=StyleProfile.qd_bins)
            yield Menu1st.engineering_per_job, job.summary, HTMLBlock(html.img(fname))

            # Block size heatmap
            wc2d = find_sensors_to_2d(self.rstorage, trange, dev_role=dev_role, sensor='block-io',
                                      metric='writes_completed')
            wc2d[wc2d < 1E-3] = 1
            sw2d = find_sensors_to_2d(self.rstorage, trange, dev_role=dev_role, sensor='block-io',
                                      metric='sectors_written')
            data2d = sw2d / wc2d / 1024
            fname = self.plt(plot_hmap_from_2d, ds(metric='wr_block_size'),
                             data2d=data2d, title=dev_role.capitalize() + " write block size",
                             ylabel="IO bsize, KiB", xlabel='Time', bins=StyleProfile.block_size_bins)
            yield Menu1st.engineering_per_job, job.summary, HTMLBlock(html.img(fname))

            # iotime heatmap
            wtime2d = find_sensors_to_2d(self.rstorage, trange, dev_role=dev_role, sensor='block-io',
                                         metric='io_time')
            fname = self.plt(plot_hmap_from_2d, ds(metric='io_time'), data2d=wtime2d,
                             xlabel='Time', ylabel="IO time (ms) per second",
                             title=dev_role.capitalize() + " iotime", bins=StyleProfile.iotime_bins)
            yield Menu1st.engineering_per_job, job.summary, HTMLBlock(html.img(fname))


# IOPS/latency over test time for each job
class LoadToolResults(JobReporter):
    """IOPS/latency during test"""
    suite_types = {'fio'}

    def get_divs(self, suite: SuiteConfig, job: JobConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:
        fjob = cast(FioJobConfig, job)

        # caption = "Load tool results, " + job.params.long_summary
        caption = "Load tool results"
        yield Menu1st.per_job, job.summary, HTMLBlock(html.H3(html.center(caption)))

        agg_io = get_aggregated(self.rstorage, suite.storage_id, fjob.storage_id, "bw", job.reliable_info_range_s)

        if fjob.bsize >= DefStyleProfile.large_blocks:
            title = "Fio measured bandwidth over time"
            units = "MiBps"
            agg_io.data //= int(unit_conversion_coef_f(units, agg_io.units))
        else:
            title = "Fio measured IOPS over time"
            agg_io.data //= (int(unit_conversion_coef_f("KiBps", agg_io.units)) * fjob.bsize)
            units = "IOPS"

        fpath = self.plt(plot_v_over_time, agg_io.source(tag='ts.' + default_format), title, units, agg_io)
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        title = "BW distribution" if fjob.bsize >= DefStyleProfile.large_blocks else "IOPS distribution"
        io_stat_prop = calc_norm_stat_props(agg_io, bins_count=StyleProfile.hist_boxes)
        fpath = self.plt(plot_hist, agg_io.source(tag='hist.' + default_format), title, units, io_stat_prop)
        yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

        if fjob.bsize < DefStyleProfile.large_blocks:
            agg_lat = get_aggregated(self.rstorage, suite.storage_id, fjob.storage_id, "lat",
                                     job.reliable_info_range_s)
            TARGET_UNITS = 'ms'
            coef = unit_conversion_coef_f(agg_lat.units, TARGET_UNITS)
            agg_lat.histo_bins = agg_lat.histo_bins.copy() * coef
            agg_lat.units = TARGET_UNITS

            fpath = self.plt(plot_lat_over_time, agg_lat.source(tag='ts.' + default_format), "Latency", agg_lat,
                             ylabel="Latency, " + agg_lat.units)
            yield Menu1st.per_job, fjob.summary, HTMLBlock(html.img(fpath))

            fpath = self.plt(plot_histo_heatmap, agg_lat.source(tag='hmap.' + default_format),
                             "Latency heatmap", agg_lat, ylabel="Latency, " + agg_lat.units, xlabel='Test time')

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

    def get_divs(self, suite: SuiteConfig, job: JobConfig) -> Iterator[Tuple[str, str, HTMLBlock]]:

        yield Menu1st.per_job, job.summary, HTMLBlock(html.H3(html.center("Cluster load")))

        sensors = []
        max_iop = 0
        max_bytes = 0
        stor_nodes = find_nodes_by_roles(self.rstorage.storage, STORAGE_ROLES)
        for sensor, metric, op, units in self.storage_sensors:
            ts = sum_sensors(self.rstorage, job.reliable_info_range_s, node_id=stor_nodes, sensor=sensor, metric=metric)
            if ts is not None:
                ds = DataSource(suite_id=suite.storage_id,
                                job_id=job.storage_id,
                                node_id="storage",
                                sensor=sensor,
                                dev=AGG_TAG,
                                metric=metric,
                                tag="ts." + default_format)

                data = ts.data if units != 'MiB' else ts.data * unit_conversion_coef_f(ts.units, 'MiB')
                ts = TimeSeries(times=numpy.arange(*job.reliable_info_range_s),
                                data=data,
                                units=units if ts.units is None else ts.units,
                                time_units=ts.time_units,
                                source=ds,
                                histo_bins=ts.histo_bins)

                sensors.append(("{} {}".format(op, units), ds, ts, units))

                if units == 'iop':
                    max_iop = max(max_iop, data.sum())
                else:
                    assert units == 'MiB'
                    max_bytes = max(max_bytes, data.sum())

        for title, ds, ts, units in sensors:
            if ts.data.sum() >= (max_iop if units == 'iop' else max_bytes) * DefStyleProfile.min_load_diff:
                fpath = self.plt(plot_v_over_time, ds, title, units, ts=ts)
                yield Menu1st.per_job, job.summary, HTMLBlock(html.img(fpath))
            else:
                logger.info("Hide '%s' plot for %s, as it's load is less then %s%% from maximum",
                            title, job.summary, int(DefStyleProfile.min_load_diff * 100))


# ------------------------------------------  REPORT STAGES  -----------------------------------------------------------


def add_devroles(ctx: TestRun):
    # TODO: need to detect all devices for node on this stage using hw info
    detected_selectors = collections.defaultdict(
        lambda: collections.defaultdict(list))  # type: Dict[str, Dict[str, List[str]]]

    for node in ctx.nodes:
        if NodeRole.osd in node.info.roles:
            all_devs = set()

            jdevs = node.info.params.get('ceph_journal_devs')
            if jdevs:
                all_devs.update(jdevs)
                detected_selectors[node.info.hostname]["|".join(jdevs)].append(DevRoles.osd_journal)

            sdevs = node.info.params.get('ceph_storage_devs')
            if sdevs:
                all_devs.update(sdevs)
                detected_selectors[node.info.hostname]["|".join(sdevs)].append(DevRoles.osd_storage)

            if all_devs:
                detected_selectors[node.info.hostname]["|".join(all_devs)].append(DevRoles.storage_block)

    for hostname, dev_rules in detected_selectors.items():
        dev_locs = []  # type: List[Dict[str, List[str]]]
        ctx.devs_locator.append({hostname: dev_locs})
        for dev_names, roles in dev_rules.items():
            dev_locs.append({dev_names: roles})


class HtmlReportStage(Stage):
    priority = StepOrder.REPORT

    def run(self, ctx: TestRun) -> None:
        nodes = ctx.rstorage.load_nodes()
        update_storage_selector(ctx.rstorage, ctx.devs_locator, nodes)

        # role2ds = roles_for_sensors(ctx.rstorage)

        job_reporters_cls = [StatInfo, LoadToolResults, Resources, ClusterLoad, CPULoadPlot]  #, QDIOTimeHeatmap]
        # job_reporters_cls = [QDIOTimeHeatmap]
        job_reporters = [rcls(ctx.rstorage, DefStyleProfile, DefColorProfile)
                         for rcls in job_reporters_cls] # type: ignore

        suite_reporters_cls = [IOQD,
                               ResourceQD,
                               PerformanceSummary,
                               EngineeringSummary,
                               ResourceConsumptionSummary]  # type: List[Type[SuiteReporter]]
        # suite_reporters_cls = []  # type: List[Type[SuiteReporter]]
        suite_reporters = [rcls(ctx.rstorage, DefStyleProfile, DefColorProfile)
                           for rcls in suite_reporters_cls]  # type: ignore

        root_dir = os.path.dirname(os.path.dirname(wally.__file__))
        doc_templ_path = os.path.join(root_dir, "report_templates/index.html")
        report_template = open(doc_templ_path, "rt").read()
        css_file_src = os.path.join(root_dir, "report_templates/main.css")
        css_file = open(css_file_src, "rt").read()

        menu_block = []
        content_block = []
        link_idx = 0

        items = defaultdict(lambda: defaultdict(list))  # type: Dict[str, Dict[str, List[HTMLBlock]]]
        DEBUG = False
        job_summ_sort_order = []

        # TODO: filter reporters
        for suite in ctx.rstorage.iter_suite(FioTest.name):
            all_jobs = list(ctx.rstorage.iter_job(suite))
            all_jobs.sort(key=lambda job: job.params)

            new_jobs_in_order = [job.summary for job in all_jobs]
            same = set(new_jobs_in_order).intersection(set(job_summ_sort_order))
            assert not same, "Job with same names in different suits found: " + ",".join(same)
            job_summ_sort_order.extend(new_jobs_in_order)

            for job in all_jobs:
                try:
                    for reporter in job_reporters:  # type: JobReporter
                        logger.debug("Start reporter %s on job %s suite %s",
                                     reporter.__class__.__name__, job.summary, suite.test_type)
                        for block, item, html in reporter.get_divs(suite, job):
                            items[block][item].append(html)
                    if DEBUG:
                        break
                except Exception:
                    logger.exception("Failed to generate report for %s", job.summary)

            for sreporter in suite_reporters:  # type: SuiteReporter
                try:
                    logger.debug("Start reporter %s on suite %s", sreporter.__class__.__name__, suite.test_type)
                    for block, item, html in sreporter.get_divs(suite):
                        items[block][item].append(html)
                except Exception:
                    logger.exception("Failed to generate report for suite %s", suite.storage_id)

            if DEBUG:
                break

        logger.debug("Generating result html")

        for idx_1st, menu_1st in enumerate(sorted(items, key=lambda x: menu_1st_order.index(x))):
            menu_block.append(
                '<a href="#item{}" class="nav-group" data-toggle="collapse" data-parent="#MainMenu">{}</a>'
                .format(idx_1st, menu_1st)
            )
            menu_block.append('<div class="collapse" id="item{}">'.format(idx_1st))

            if menu_1st in (Menu1st.per_job, Menu1st.engineering_per_job):
                key = job_summ_sort_order.index
            elif menu_1st == Menu1st.engineering:
                key = Menu2ndEng.order.index
            elif menu_1st == Menu1st.summary:
                key = Menu2ndSumm.order.index
            else:
                key = lambda x: x

            in_order = sorted(items[menu_1st], key=key)

            for menu_2nd in in_order:
                menu_block.append('    <a href="#content{}" class="nav-group-item">{}</a>'
                                  .format(link_idx, menu_2nd))
                content_block.append('<div id="content{}">'.format(link_idx))
                content_block.extend("    " + x.data for x in items[menu_1st][menu_2nd])
                content_block.append('</div>')
                link_idx += 1
            menu_block.append('</div>')

        report = report_template.replace("{{{menu}}}", ("\n" + " " * 16).join(menu_block))
        report = report.replace("{{{content}}}", ("\n" + " " * 16).join(content_block))
        report_path = ctx.rstorage.put_report(report, "index.html")
        ctx.rstorage.put_report(css_file, "main.css")
        logger.info("Report is stored into %r", report_path)
