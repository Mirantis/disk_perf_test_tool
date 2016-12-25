import os
import csv
import abc
import bisect
import logging
import itertools
import collections
from io import StringIO
from typing import Dict, Any, Iterator, Tuple, cast, List

try:
    import numpy
    import scipy
    import matplotlib
    matplotlib.use('svg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import wally
from .utils import ssize2b
from .storage import Storage
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .result_classes import TestInfo, FullTestResult, SensorInfo
from .suits.io.fio_task_parser import (get_test_sync_mode,
                                       get_test_summary,
                                       parse_all_in_1,
                                       abbv_name_to_full)


logger = logging.getLogger("wally")


def load_test_results(storage: Storage) -> Iterator[FullTestResult]:
    raise NotImplementedError()
    # sensors_data = {}  # type: Dict[Tuple[str, str, str], SensorInfo]
    #
    # mstorage = storage.sub_storage("metric")
    # for _, node_id in mstorage.list():
    #     for _, dev_name in mstorage.list(node_id):
    #         for _, sensor_name in mstorage.list(node_id, dev_name):
    #             key = (node_id, dev_name, sensor_name)
    #             si = SensorInfo(*key)
    #             si.begin_time, si.end_time, si.data = storage[node_id, dev_name, sensor_name]  # type: ignore
    #             sensors_data[key] = si
    #
    # rstorage = storage.sub_storage("result")
    # for _, run_id in rstorage.list():
    #     ftr = FullTestResult()
    #     ftr.test_info = rstorage.load(TestInfo, run_id, "info")
    #     ftr.performance_data = {}
    #
    #     p1 = "{}/measurement".format(run_id)
    #     for _, node_id in rstorage.list(p1):
    #         for _, measurement_name in rstorage.list(p1, node_id):
    #             perf_key = (node_id, measurement_name)
    #             ftr.performance_data[perf_key] = rstorage["{}/{}/{}".format(p1, *perf_key)]  # type: ignore
    #
    #     yield ftr


class ConsoleReportStage(Stage):

    priority = StepOrder.REPORT

    def run(self, ctx: TestRun) -> None:
        # TODO(koder): load data from storage
        raise NotImplementedError("...")


class HtmlReportStage(Stage):

    priority = StepOrder.REPORT

    def run(self, ctx: TestRun) -> None:
        # TODO(koder): load data from storage
        raise NotImplementedError("...")


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


class HTMLBlock:
    data = None  # type: str
    js_links = []  # type: List[str]
    css_links = []  # type: List[str]


class Reporter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_divs(self, config, storage) -> Iterator[Tuple[str, str, HTMLBlock]]:
        pass


# Main performance report
class PerformanceSummary(Reporter):
    """Creates graph, which show how IOPS and Latency depend on QD"""


# Main performance report
class IOPS_QD(Reporter):
    """Creates graph, which show how IOPS and Latency depend on QD"""


# Linearization report
class IOPS_Bsize(Reporter):
    """Creates graphs, which show how IOPS and Latency depend on block size"""


# IOPS/latency distribution
class IOPSHist(Reporter):
    """IOPS.latency distribution histogram"""


# IOPS/latency over test time
class IOPSTime(Reporter):
    """IOPS/latency during test"""


# Cluster load over test time
class ClusterLoad(Reporter):
    """IOPS/latency during test"""


# Node load over test time
class NodeLoad(Reporter):
    """IOPS/latency during test"""


# Ceph cluster summary
class CephClusterSummary(Reporter):
    """IOPS/latency during test"""


# TODO: Resource consumption report
# TODO: Ceph operation breakout report
# TODO: Resource consumption for different type of test


#
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
