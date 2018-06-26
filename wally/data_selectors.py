import logging
from typing import Tuple, Iterator, List, Iterable, Dict, Union, Callable, Set

import numpy

from cephlib.numeric_types import DataSource, TimeSeries
from cephlib.storage_selectors import c_interpolate_ts_on_seconds_border
from cephlib.node import NodeInfo

from .result_classes import IWallyStorage
from .suits.io.fio_hist import expected_lat_bins


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


def find_all_series(rstorage: IWallyStorage, suite_id: str, job_id: str, metric: str) -> Iterator[TimeSeries]:
    "Iterated over selected metric for all nodes for given Suite/job"
    return (rstorage.get_ts(ds) for ds in rstorage.iter_ts(suite_id=suite_id, job_id=job_id, metric=metric))


def get_aggregated(rstorage: IWallyStorage, suite_id: str, job_id: str, metric: str,
                   trange: Tuple[int, int]) -> TimeSeries:
    "Sum selected fio metric for all nodes for given Suite/job"

    key = (id(rstorage), suite_id, job_id, metric, trange)
    aggregated_cache = rstorage.storage.other_caches['aggregated']
    if key in aggregated_cache:
        return aggregated_cache[key].copy()

    tss = list(find_all_series(rstorage, suite_id, job_id, metric))

    if len(tss) == 0:
        raise NameError(f"Can't found any TS for {suite_id},{job_id},{metric}")

    c_intp = c_interpolate_ts_on_seconds_border
    tss_inp = [c_intp(ts.select(trange), tp='fio', allow_broken_step=(metric == 'lat')) for ts in tss]

    res = None
    res_times = None

    for ts, ts_orig in zip(tss_inp, tss):
        if ts.time_units != 's':
            msg = "time_units must be 's' for fio sensor"
            logger.error(msg)
            raise ValueError(msg)

        if metric == 'lat' and (len(ts.data.shape) != 2 or ts.data.shape[1] != expected_lat_bins):
            msg = f"Sensor {ts.source.dev}.{ts.source.sensor} on node {ts.source.node_id} " + \
                f"has shape={ts.data.shape}. Can only process sensors with shape=[X, {expected_lat_bins}]."
            logger.error(msg)
            raise ValueError(msg)

        if metric != 'lat' and len(ts.data.shape) != 1:
            msg = f"Sensor {ts.source.dev}.{ts.source.sensor} on node {ts.source.node_id} " + \
                f"has shape={ts.data.shape}. Can only process 1D sensors."
            logger.error(msg)
            raise ValueError(msg)

        assert trange[0] >= ts.times[0] and trange[1] <= ts.times[-1], \
            f"[{ts.times[0]}, {ts.times[-1]}] not in [{trange[0]}, {trange[-1]}]"


        idx1, idx2 = numpy.searchsorted(ts.times, trange)
        idx2 += 1

        assert (idx2 - idx1) == (trange[1] - trange[0] + 1), \
            "Broken time array at {} for {}".format(trange, ts.source)

        dt = ts.data[idx1: idx2]
        if res is None:
            res = dt.copy()
            res_times = ts.times[idx1: idx2].copy()
        else:
            assert res.shape == dt.shape, f"res.shape(={res.shape}) != dt.shape(={dt.shape})"
            res += dt

    ds = DataSource(suite_id=suite_id, job_id=job_id, node_id=AGG_TAG, sensor='fio',
                    dev=AGG_TAG, metric=metric, tag='csv')
    agg_ts = TimeSeries(res, source=ds,
                        times=res_times,
                        units=tss_inp[0].units,
                        histo_bins=tss_inp[0].histo_bins,
                        time_units=tss_inp[0].time_units)
    aggregated_cache[key] = agg_ts
    return agg_ts.copy()


def get_nodes(storage: IWallyStorage, roles: Iterable[str]) -> List[NodeInfo]:
    return [node for node in storage.load_nodes() if node.roles.intersection(roles)]

