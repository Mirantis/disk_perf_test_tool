import logging
from typing import Tuple, Iterator

import numpy

from cephlib.numeric_types import DataSource, TimeSeries
from cephlib.storage_selectors import c_interpolate_ts_on_seconds_border

from .result_classes import IResultStorage
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


def find_all_series(rstorage: IResultStorage, suite_id: str, job_id: str, metric: str) -> Iterator[TimeSeries]:
    "Iterated over selected metric for all nodes for given Suite/job"
    return (rstorage.get_ts(ds) for ds in rstorage.iter_ts(suite_id=suite_id, job_id=job_id, metric=metric))


def get_aggregated(rstorage: IResultStorage, suite_id: str, job_id: str, metric: str,
                   trange: Tuple[int, int]) -> TimeSeries:
    "Sum selected metric for all nodes for given Suite/job"

    tss = list(find_all_series(rstorage, suite_id, job_id, metric))

    if len(tss) == 0:
        raise NameError("Can't found any TS for {},{},{}".format(suite_id, job_id, metric))

    ds = DataSource(suite_id=suite_id, job_id=job_id, node_id=AGG_TAG, sensor='fio',
                    dev=AGG_TAG, metric=metric, tag='csv')

    tss_inp = [c_interpolate_ts_on_seconds_border(ts, tp='fio', allow_broken_step=(metric == 'lat')) for ts in tss]
    res = None

    for ts in tss_inp:
        if ts.time_units != 's':
            msg = "time_units must be 's' for fio sensor"
            logger.error(msg)
            raise ValueError(msg)

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

        assert trange[0] >= ts.times[0] and trange[1] <= ts.times[-1], \
            "[{}, {}] not in [{}, {}]".format(ts.times[0], ts.times[-1], trange[0], trange[-1])

        idx1, idx2 = numpy.searchsorted(ts.times, trange)
        idx2 += 1

        assert (idx2 - idx1) == (trange[1] - trange[0] + 1), \
            "Broken time array at {} for {}".format(trange, ts.source)

        dt = ts.data[idx1: idx2]
        if res is None:
            res = dt
        else:
            assert res.shape == dt.shape, "res.shape(={}) != dt.shape(={})".format(res.shape, dt.shape)
            res += dt

    agg_ts = TimeSeries(res, source=ds,
                        times=tss_inp[0].times.copy(),
                        units=tss_inp[0].units,
                        histo_bins=tss_inp[0].histo_bins,
                        time_units=tss_inp[0].time_units)

    return agg_ts

