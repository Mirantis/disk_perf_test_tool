import ctypes
import logging
import os.path
from typing import Tuple, List, Iterable, Iterator, Optional, Union
from fractions import Fraction

import numpy

from cephlib.numeric import auto_edges2

import wally
from .hlstorage import ResultStorage
from .node_interfaces import NodeInfo
from .result_classes import DataSource, TimeSeries, SuiteConfig, JobConfig
from .suits.io.fio import FioJobConfig
from .suits.io.fio_hist import expected_lat_bins
from .utils import unit_conversion_coef


logger = logging.getLogger("wally")

# Separately for each test heatmaps & agg acroos whole time histos:
#   * fio latency heatmap for all instances
#   * data dev iops across all osd
#   * data dev bw across all osd
#   * date dev qd across all osd
#   * journal dev iops across all osd
#   * journal dev bw across all osd
#   * journal dev qd across all osd
#   * net dev pps across all hosts
#   * net dev bps across all hosts

# Main API's
#   get sensors by pattern
#   allign values to seconds
#   cut ranges for particular test
#   transform into 2d histos (either make histos or rebin them) and clip outliers same time


AGG_TAG = 'ALL'


def find_nodes_by_roles(rstorage: ResultStorage, node_roles: Iterable[str]) -> List[NodeInfo]:
    nodes = rstorage.storage.load_list(NodeInfo, 'all_nodes')  # type: List[NodeInfo]
    node_roles_s = set(node_roles)
    return [node for node in nodes if node.roles.intersection(node_roles_s)]


def find_all_sensors(rstorage: ResultStorage,
                     node_roles: Iterable[str],
                     sensor: str,
                     metric: str) -> Iterator[TimeSeries]:
    all_nodes_rr = "|".join(node.node_id for node in find_nodes_by_roles(rstorage, node_roles))
    all_nodes_rr = "(?P<node>{})".format(all_nodes_rr)

    for path, ds in rstorage.iter_sensors(all_nodes_rr, sensor=sensor, metric=metric):
        ts = rstorage.load_sensor(ds)

        # for sensors ts.times is array of pairs - collection_start_at, colelction_finished_at
        # to make this array consistent with times in load data second item if each pair is dropped
        ts.times = ts.times[::2]
        yield ts


def find_all_series(rstorage: ResultStorage, suite: SuiteConfig, job: JobConfig, metric: str) -> Iterator[TimeSeries]:
    "Iterated over selected metric for all nodes for given Suite/job"
    return rstorage.iter_ts(suite, job, metric=metric)


def get_aggregated(rstorage: ResultStorage, suite: SuiteConfig, job: FioJobConfig, metric: str) -> TimeSeries:
    "Sum selected metric for all nodes for given Suite/job"

    tss = list(find_all_series(rstorage, suite, job, metric))

    if len(tss) == 0:
        raise NameError("Can't found any TS for {},{},{}".format(suite, job, metric))

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
                        units=tss[0].units,
                        histo_bins=tss[0].histo_bins,
                        time_units=tss[0].time_units)

    for ts in tss:
        if metric == 'lat' and (len(ts.data.shape) != 2 or ts.data.shape[1] != expected_lat_bins):
            msg = "Sensor {}.{} on node %s has shape={}. Can only process sensors with shape=[X, {}].".format(
                         ts.source.dev, ts.source.sensor, ts.source.node_id, ts.data.shape, expected_lat_bins)
            logger.error(msg)
            raise ValueError(msg)

        if metric != 'lat' and len(ts.data.shape) != 1:
            msg = "Sensor {}.{} on node {} has shape={}. Can only process 1D sensors.".format(
                         ts.source.dev, ts.source.sensor, ts.source.node_id, ts.data.shape)
            logger.error(msg)
            raise ValueError(msg)

        # TODO: match times on different ts
        agg_ts.data += ts.data

    return agg_ts


interpolated_cache = {}


def interpolate_ts_on_seconds_border(ts: TimeSeries, nc: bool = False) -> TimeSeries:
    "Interpolate time series to values on seconds borders"
    logging.warning("This implementation of interpolate_ts_on_seconds_border is deplricated and should be updated")

    if not nc and ts.source.tpl in interpolated_cache:
        return interpolated_cache[ts.source.tpl]

    assert len(ts.times) == len(ts.data), "Time(={}) and data(={}) sizes doesn't equal for {!s}"\
            .format(len(ts.times), len(ts.data), ts.source)

    rcoef = 1 / unit_conversion_coef(ts.time_units, 's')  # type: Union[int, Fraction]

    if isinstance(rcoef, Fraction):
        assert rcoef.denominator == 1, "Incorrect conversion coef {!r}".format(rcoef)
        rcoef = rcoef.numerator

    assert rcoef >= 1 and isinstance(rcoef, int), "Incorrect conversion coef {!r}".format(rcoef)
    coef = int(rcoef)   # make typechecker happy

    # round to seconds border
    begin = int(ts.times[0] / coef + 1) * coef
    end = int(ts.times[-1] / coef) * coef

    # current real data time chunk begin time
    edge_it = iter(ts.times)

    # current real data value
    val_it = iter(ts.data)

    # result array, cumulative value per second
    result = numpy.empty([(end - begin) // coef], dtype=ts.data.dtype)
    idx = 0
    curr_summ = 0

    # end of current time slot
    results_cell_ends = begin + coef

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
            results_cell_ends += coef

        # don't lost any real data
        curr_summ += real_val_left

    assert idx == len(result), "Wrong output array size - idx(={}) != len(result)(={})".format(idx, len(result))

    res_ts = TimeSeries(ts.name, None, result,
                        times=int(begin // coef) + numpy.arange(idx, dtype=ts.times.dtype),
                        units=ts.units,
                        time_units='s',
                        source=ts.source(),
                        histo_bins=ts.histo_bins)

    if not nc:
        interpolated_cache[ts.source.tpl] = res_ts

    return res_ts


c_interp_func = None
c_interp_func_qd = None


def c_interpolate_ts_on_seconds_border(ts: TimeSeries, nc: bool = False, qd: bool = False) -> TimeSeries:
    "Interpolate time series to values on seconds borders"
    key = (ts.source.tpl, qd)
    if not nc and key in interpolated_cache:
        return interpolated_cache[key].copy()

    # both data and times must be 1d compact arrays
    assert len(ts.data.strides) == 1, "ts.data.strides must be 1D, not " + repr(ts.data.strides)
    assert ts.data.dtype.itemsize == ts.data.strides[0], "ts.data array must be compact"
    assert len(ts.times.strides) == 1, "ts.times.strides must be 1D, not " + repr(ts.times.strides)
    assert ts.times.dtype.itemsize == ts.times.strides[0], "ts.times array must be compact"

    assert len(ts.times) == len(ts.data), "Time(={}) and data(={}) sizes doesn't equal for {!s}"\
            .format(len(ts.times), len(ts.data), ts.source)

    rcoef = 1 / unit_conversion_coef(ts.time_units, 's')  # type: Union[int, Fraction]

    if isinstance(rcoef, Fraction):
        assert rcoef.denominator == 1, "Incorrect conversion coef {!r}".format(rcoef)
        rcoef = rcoef.numerator

    assert rcoef >= 1 and isinstance(rcoef, int), "Incorrect conversion coef {!r}".format(rcoef)
    coef = int(rcoef)   # make typechecker happy

    global c_interp_func
    global c_interp_func_qd

    uint64_p = ctypes.POINTER(ctypes.c_uint64)

    if c_interp_func is None:
        dirname = os.path.dirname(os.path.dirname(wally.__file__))
        path = os.path.join(dirname, 'clib', 'libwally.so')
        cdll = ctypes.CDLL(path)

        c_interp_func = cdll.interpolate_ts_on_seconds_border
        c_interp_func.argtypes = [
            ctypes.c_uint,  # input_size
            ctypes.c_uint,  # output_size
            uint64_p,  # times
            uint64_p,  # values
            ctypes.c_uint,  # time_scale_coef
            uint64_p,  # output
        ]
        c_interp_func.restype = None

        c_interp_func_qd = cdll.interpolate_ts_on_seconds_border_qd
        c_interp_func_qd.argtypes = [
            ctypes.c_uint,  # input_size
            ctypes.c_uint,  # output_size
            uint64_p,  # times
            uint64_p,  # values
            ctypes.c_uint,  # time_scale_coef
            uint64_p,  # output
        ]
        c_interp_func_qd.restype = ctypes.c_uint

    assert ts.data.dtype.name == 'uint64', "Data dtype for {}=={} != uint64".format(ts.source, ts.data.dtype.name)
    assert ts.times.dtype.name == 'uint64', "Time dtype for {}=={} != uint64".format(ts.source, ts.times.dtype.name)

    output_sz = int(ts.times[-1]) // coef - int(ts.times[0]) // coef + 2
    # print("output_sz =", output_sz, "coef =", coef)
    result = numpy.zeros(output_sz, dtype=ts.data.dtype.name)

    if qd:
        func = c_interp_func_qd
    else:
        func = c_interp_func

    sz = func(ts.data.size,
              output_sz,
              ts.times.ctypes.data_as(uint64_p),
              ts.data.ctypes.data_as(uint64_p),
              coef,
              result.ctypes.data_as(uint64_p))

    if qd:
        result = result[:sz]
        output_sz = sz
    else:
        assert sz is None

    rtimes = int(ts.times[0] // coef) + numpy.arange(output_sz, dtype=ts.times.dtype)
    res_ts = TimeSeries(ts.name, None, result,
                        times=rtimes,
                        units=ts.units,
                        time_units='s',
                        source=ts.source(),
                        histo_bins=ts.histo_bins)

    if not nc:
        interpolated_cache[ts.source.tpl] = res_ts.copy()

    return res_ts


def get_ts_for_time_range(ts: TimeSeries, time_range: Tuple[int, int]) -> TimeSeries:
    """Return sensor values for given node for given period. Return per second estimated values array
    Raise an error if required range is not full covered by data in storage"""

    assert ts.time_units == 's', "{} != s for {!s}".format(ts.time_units, ts.source)
    assert len(ts.times) == len(ts.data), "Time(={}) and data(={}) sizes doesn't equal for {!s}"\
            .format(len(ts.times), len(ts.data), ts.source)

    if time_range[0] < ts.times[0] or time_range[1] > ts.times[-1]:
        raise AssertionError(("Incorrect data for get_sensor - time_range={!r}, collected_at=[{}, ..., {}]," +
                              "sensor = {}_{}.{}.{}").format(time_range, ts.times[0], ts.times[-1],
                                                             ts.source.node_id, ts.source.sensor, ts.source.dev,
                                                             ts.source.metric))
    idx1, idx2 = numpy.searchsorted(ts.times, time_range)
    return TimeSeries(ts.name, None,
                      ts.data[idx1:idx2],
                      times=ts.times[idx1:idx2],
                      units=ts.units,
                      time_units=ts.time_units,
                      source=ts.source,
                      histo_bins=ts.histo_bins)


def make_2d_histo(tss: List[TimeSeries],
                  outliers_range: Tuple[float, float] = (0.02, 0.98),
                  bins_count: int = 20,
                  log_bins: bool = False) -> TimeSeries:

    # validate input data
    for ts in tss:
        assert len(ts.times) == len(ts.data), "Time(={}) and data(={}) sizes doesn't equal for {!s}"\
                .format(len(ts.times), len(ts.data), ts.source)
        assert ts.time_units == 's', "All arrays should have the same data units"
        assert ts.units == tss[0].units, "All arrays should have the same data units"
        assert ts.data.shape == tss[0].data.shape, "All arrays should have the same data size"
        assert len(ts.data.shape) == 1, "All arrays should be 1d"

    whole_arr = numpy.concatenate([ts.data for ts in tss])
    whole_arr.shape = [len(tss), -1]

    if outliers_range is not None:
        max_vl, begin, end, min_vl = numpy.percentile(whole_arr,
                                                      [0, outliers_range[0] * 100, outliers_range[1] * 100, 100])
        bins_edges = auto_edges2(begin, end, bins=bins_count, log_space=log_bins)
        fixed_bins_edges = bins_edges.copy()
        fixed_bins_edges[0] = begin
        fixed_bins_edges[-1] = end
    else:
        begin, end = numpy.percentile(whole_arr, [0, 100])
        bins_edges = auto_edges2(begin, end, bins=bins_count, log_space=log_bins)
        fixed_bins_edges = bins_edges

    res_data = numpy.concatenate(numpy.histogram(column, fixed_bins_edges) for column in whole_arr.T)
    res_data.shape = (len(tss), -1)
    res = TimeSeries(name=tss[0].name,
                     raw=None,
                     data=res_data,
                     times=tss[0].times,
                     units=tss[0].units,
                     source=tss[0].source,
                     time_units=tss[0].time_units,
                     histo_bins=bins_edges)
    return res


def aggregate_histograms(tss: List[TimeSeries],
                         outliers_range: Tuple[float, float] = (0.02, 0.98),
                         bins_count: int = 20,
                         log_bins: bool = False) -> TimeSeries:

    # validate input data
    for ts in tss:
        assert len(ts.times) == len(ts.data), "Need to use stripped time"
        assert ts.time_units == 's', "All arrays should have the same data units"
        assert ts.units == tss[0].units, "All arrays should have the same data units"
        assert ts.data.shape == tss[0].data.shape, "All arrays should have the same data size"
        assert len(ts.data.shape) == 2, "All arrays should be 2d"
        assert ts.histo_bins is not None, "All arrays should be 2d"

    whole_arr = numpy.concatenate([ts.data for ts in tss])
    whole_arr.shape = [len(tss), -1]

    max_val = whole_arr.min()
    min_val = whole_arr.max()

    if outliers_range is not None:
        begin, end = numpy.percentile(whole_arr, [outliers_range[0] * 100, outliers_range[1] * 100])
    else:
        begin = min_val
        end = max_val

    bins_edges = auto_edges2(begin, end, bins=bins_count, log_space=log_bins)

    if outliers_range is not None:
        fixed_bins_edges = bins_edges.copy()
        fixed_bins_edges[0] = begin
        fixed_bins_edges[-1] = end
    else:
        fixed_bins_edges = bins_edges

    res_data = numpy.concatenate(numpy.histogram(column, fixed_bins_edges) for column in whole_arr.T)
    res_data.shape = (len(tss), -1)
    return TimeSeries(name=tss[0].name,
                      raw=None,
                      data=res_data,
                      times=tss[0].times,
                      units=tss[0].units,
                      source=tss[0].source,
                      time_units=tss[0].time_units,
                      histo_bins=fixed_bins_edges)


qd_metrics = {'io_queue'}


def summ_sensors(rstorage: ResultStorage,
                 roles: List[str],
                 sensor: str,
                 metric: str,
                 time_range: Tuple[int, int]) -> Optional[TimeSeries]:

    res = None  # type: Optional[TimeSeries]
    for node in find_nodes_by_roles(rstorage, roles):
        for _, ds in rstorage.iter_sensors(node_id=node.node_id, sensor=sensor, metric=metric):
            data = rstorage.load_sensor(ds)
            data = c_interpolate_ts_on_seconds_border(data, qd=metric in qd_metrics)
            data = get_ts_for_time_range(data, time_range)
            if res is None:
                res = data
                res.data = res.data.copy()
            else:
                res.data += data.data
    return res


def find_sensors_to_2d(rstorage: ResultStorage,
                       roles: List[str],
                       sensor: str,
                       devs: List[str],
                       metric: str,
                       time_range: Tuple[int, int]) -> numpy.ndarray:

    res = []  # type: List[TimeSeries]
    for node in find_nodes_by_roles(rstorage, roles):
        for dev in devs:
            for _, ds in rstorage.iter_sensors(node_id=node.node_id, sensor=sensor, dev=dev, metric=metric):
                data = rstorage.load_sensor(ds)
                data = c_interpolate_ts_on_seconds_border(data, qd=metric in qd_metrics)
                data = get_ts_for_time_range(data, time_range)
                res.append(data.data)
    res2d = numpy.concatenate(res)
    res2d.shape = ((len(res), -1))
    return res2d
