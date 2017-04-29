import numpy
from wally.statistic import rebin_histogram
from wally.result_classes import DataSource, TimeSeries
from wally.data_selectors import interpolate_ts_on_seconds_border, c_interpolate_ts_on_seconds_border


def array_eq(x: numpy.array, y: numpy.array, max_diff: float = 1E-3) -> bool:
    return numpy.abs(x - y).max() <= max_diff


def test_rebin_histo():
    curr_histo = numpy.empty((100,), dtype=int)
    curr_histo[:] = 1
    edges = numpy.arange(100)
    new_histo, new_edges = rebin_histogram(curr_histo, edges, 10)

    assert new_edges.shape == (10,)
    assert new_histo.shape == (10,)
    assert new_edges.dtype.name.startswith('float')
    assert new_histo.dtype.name.startswith('int')

    assert array_eq(new_edges, numpy.arange(10) * 9.9)
    assert new_histo.sum() == curr_histo.sum()
    assert list(new_histo) == [10] * 10

    new_histo, new_edges = rebin_histogram(curr_histo, edges, 3,
                                           left_tail_idx=20,
                                           right_tail_idx=50)

    assert new_edges.shape == (3,)
    assert new_histo.shape == (3,)
    assert array_eq(new_edges, numpy.array([20, 30, 40]))
    assert new_histo.sum() == curr_histo.sum()
    assert list(new_histo) == [30, 10, 60]


SUITE_ID = "suite1"
JOB_ID = "job1"
NODE_ID = "node1"
SENSOR = "sensor"
DEV = "dev"
METRIC = "metric"
TAG = "csv"
DATA_UNITS = "x"
TIME_UNITS = "ms"


def test_interpolate():
    ds = DataSource(node_id=NODE_ID, sensor=SENSOR, dev=DEV, metric=METRIC)
    samples = 200
    ms_coef = 1000
    s_offset = 377 * ms_coef
    ms_offset = 300 + s_offset

    for i in range(16):
        source_times = numpy.random.randint(100, size=samples, dtype='uint64') + \
            ms_coef * numpy.arange(samples, dtype='uint64') + s_offset + ms_offset
        source_values = numpy.random.randint(30, 60, size=samples, dtype='uint64')

        ts = TimeSeries("test", raw=None, data=source_values, times=source_times, units=DATA_UNITS,
                        source=ds, time_units=TIME_UNITS)

        # ts2 = interpolate_ts_on_seconds_border(ts)
        ts2 = c_interpolate_ts_on_seconds_border(ts, nc=True)

        # print()
        # print(ts.times)
        # print(ts.data, ts.data.sum())
        # print(ts2.times)
        # print(ts2.data, ts2.data.sum())

        assert ts.time_units == 'ms'
        assert ts2.time_units == 's'
        assert ts2.times.dtype == ts.times.dtype
        assert ts2.data.dtype == ts.data.dtype

        assert ts.data.sum() == ts2.data.sum()

        borders = 5
        block_size = samples // 10
        for begin_idx in numpy.random.randint(borders, samples - borders, size=20):
            begin_idx = int(begin_idx)
            end_idx = min(begin_idx + block_size, ts.times.size - 1)

            first_cell_begin_time = ts.times[begin_idx - 1]
            last_cell_end_time = ts.times[end_idx]
            ts_sum = ts.data[begin_idx:end_idx].sum()

            ts2_begin_idx = numpy.searchsorted(ts2.times, first_cell_begin_time // ms_coef)
            ts2_end_idx = numpy.searchsorted(ts2.times, last_cell_end_time // ms_coef) + 1
            ts2_max = ts.data[ts2_begin_idx: ts2_end_idx].sum()
            ts2_min = ts.data[ts2_begin_idx + 1: ts2_end_idx - 1].sum()

            assert ts2_min <= ts_sum <= ts2_max, "NOT {} <= {} <= {}".format(ts2_min, ts_sum, ts2_max)
