import os
import shutil
import tempfile
import contextlib
from typing import Tuple, Union, Dict, Any

import numpy

from wally.result_classes import DataSource, TimeSeries, SuiteConfig
from wally.suits.job import JobConfig, JobParams
from wally.hlstorage import ResultStorage

from cephlib.storage import make_storage

@contextlib.contextmanager
def in_temp_dir():
    dname = tempfile.mkdtemp()
    try:
        yield dname
    finally:
        shutil.rmtree(dname)


SUITE_ID = "suite_1"
JOB_ID = "job_11"
NODE_ID = "11.22.33.44:223"
SENSOR = "sensor"
DEV = "dev"
METRIC = "metric"
TAG = "csv"
DATA_UNITS = "x"
TIME_UNITS = "us"


class TJobParams(JobParams):
    def __init__(self) -> None:
        JobParams.__init__(self)

    @property
    def summary(self) -> str:
        return "UT_Job_CFG"

    @property
    def long_summary(self) -> str:
        return "UT_Job_Config"

    def copy(self, **updated) -> 'TJobParams':
        return self.__class__()

    @property
    def char_tpl(self) -> Tuple[Union[str, int, float, bool], ...]:
        return (1, 2, 3)


class TJobConfig(JobConfig):
    @property
    def storage_id(self) -> str:
        return JOB_ID

    @property
    def params(self) -> JobParams:
        return TJobParams()

    def raw(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'TJobConfig':
        return cls()


class TSuiteConfig(SuiteConfig):
    def __init__(self):
        SuiteConfig.__init__(self, "UT", {}, "run_uuid", [], "/tmp", 0, False)
        self.storage_id = SUITE_ID


def test_sensor_ts():
    with in_temp_dir() as root:
        size = 5
        sensor_data = numpy.arange(size)
        collected_at = numpy.arange(size * 2) + 100

        ds = DataSource(node_id=NODE_ID, sensor=SENSOR, dev=DEV, metric=METRIC, tag='csv')
        cds = DataSource(node_id=NODE_ID, metric='collected_at', tag='csv')

        with make_storage(root, existing=False) as storage:
            rstorage = ResultStorage(storage)

            rstorage.append_sensor(sensor_data, ds, units=DATA_UNITS)
            rstorage.append_sensor(sensor_data, ds, units=DATA_UNITS)

            rstorage.append_sensor(collected_at, cds, units=TIME_UNITS)
            rstorage.append_sensor(collected_at + size * 2, cds, units=TIME_UNITS)

        with make_storage(root, existing=True) as storage2:
            rstorage2 = ResultStorage(storage2)
            ts = rstorage2.get_sensor(ds)
            assert numpy.array_equal(ts.data, numpy.concatenate((sensor_data, sensor_data)))
            assert numpy.array_equal(ts.times, numpy.concatenate((collected_at, collected_at + size * 2))[::2])


def test_result_ts():
    with in_temp_dir() as root:
        sensor_data = numpy.arange(5, dtype=numpy.uint32)
        collected_at = numpy.arange(5, dtype=numpy.uint32) + 100
        ds = DataSource(suite_id=SUITE_ID, job_id=JOB_ID,
                        node_id=NODE_ID, sensor=SENSOR, dev=DEV, metric=METRIC, tag=TAG)
        ds.verify()

        ts = TimeSeries(sensor_data, times=collected_at, units=DATA_UNITS, source=ds, time_units=TIME_UNITS)

        suite = TSuiteConfig()
        job = TJobConfig(1)

        with make_storage(root, existing=False) as storage:
            rstorage = ResultStorage(storage)
            rstorage.put_or_check_suite(suite)
            rstorage.put_job(suite, job)
            rstorage.put_ts(ts)

        with make_storage(root, existing=True) as storage2:
            rstorage2 = ResultStorage(storage2)
            suits = list(rstorage2.iter_suite('UT'))
            suits2 = list(rstorage2.iter_suite())
            assert len(suits) == 1
            assert len(suits2) == 1
