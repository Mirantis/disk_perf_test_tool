import shutil
import tempfile
import contextlib
from typing import Tuple, Union, Dict, Any

import numpy
from oktest import ok


from wally.result_classes import DataSource, TimeSeries, SuiteConfig
from wally.suits.job import JobConfig, JobParams
from wally.storage import make_storage
from wally.hlstorage import ResultStorage


@contextlib.contextmanager
def in_temp_dir():
    dname = tempfile.mkdtemp()
    try:
        yield dname
    finally:
        shutil.rmtree(dname)


SUITE_ID = "suite1"
JOB_ID = "job1"
NODE_ID = "node1"
SENSOR = "sensor"
DEV = "dev"
METRIC = "metric"
TAG = "csv"
DATA_UNITS = "x"
TIME_UNITS = "us"


class TestJobParams(JobParams):
    def __init__(self) -> None:
        JobParams.__init__(self)

    @property
    def summary(self) -> str:
        return "UT_Job_CFG"

    @property
    def long_summary(self) -> str:
        return "UT_Job_Config"

    def copy(self, **updated) -> 'JobParams':
        return self.__class__()

    @property
    def char_tpl(self) -> Tuple[Union[str, int, float, bool], ...]:
        return (1, 2, 3)


class TestJobConfig(JobConfig):
    @property
    def storage_id(self) -> str:
        return JOB_ID

    @property
    def params(self) -> JobParams:
        return TestJobParams()

    def raw(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'TestJobConfig':
        return cls()


class TestSuiteConfig(SuiteConfig):
    def __init__(self):
        SuiteConfig.__init__(self, "UT", {}, "run_uuid", [], "/tmp", 0, False)
        self.storage_id = SUITE_ID



def test_sensor_ts():
    with in_temp_dir() as root:
        sensor_data = numpy.arange(5)
        collected_at = numpy.arange(5) + 100

        ds = DataSource(node_id=NODE_ID, sensor=SENSOR, dev=DEV, metric=METRIC, tag='csv')
        cds = DataSource(node_id=NODE_ID, metric='collected_at', tag='csv')

        with make_storage(root, existing=False) as storage:
            rstorage = ResultStorage(storage)

            rstorage.append_sensor(sensor_data, ds, units=DATA_UNITS, histo_bins=None)
            rstorage.append_sensor(sensor_data, ds, units=DATA_UNITS, histo_bins=None)

            rstorage.append_sensor(collected_at, cds, units=TIME_UNITS, histo_bins=None)
            rstorage.append_sensor(collected_at + 5, cds, units=TIME_UNITS, histo_bins=None)

        with make_storage(root, existing=True) as storage2:
            rstorage2 = ResultStorage(storage2)
            ts = rstorage2.load_sensor(ds)
            assert (ts.data == numpy.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])).all()
            assert (ts.times == numpy.arange(10) + 100).all()


def test_result_ts():
    with in_temp_dir() as root:
        sensor_data = numpy.arange(5, dtype=numpy.uint32)
        collected_at = numpy.arange(5, dtype=numpy.uint32) + 100
        ds = DataSource(suite_id=SUITE_ID, job_id=JOB_ID,
                        node_id=NODE_ID, sensor=SENSOR, dev=DEV, metric=METRIC, tag=TAG)

        ts = TimeSeries("xxxx", None, sensor_data, collected_at, DATA_UNITS, ds, TIME_UNITS)

        suite = TestSuiteConfig()
        job = TestJobConfig(1)

        with make_storage(root, existing=False) as storage:
            rstorage = ResultStorage(storage)
            rstorage.put_or_check_suite(suite)
            rstorage.put_job(suite, job)
            rstorage.put_ts(ts)

        with make_storage(root, existing=True) as storage2:
            rstorage2 = ResultStorage(storage2)
            suits = list(rstorage2.iter_suite('UT'))
            suits2 = list(rstorage2.iter_suite())
            ok(len(suits)) == 1
            ok(len(suits2)) == 1
            
