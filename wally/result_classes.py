import abc
import array
from typing import Dict, List, Any, Optional, Tuple, cast, Type, Iterator
from collections import OrderedDict


import numpy
from scipy.stats.mstats_basic import NormaltestResult


from .node_interfaces import IRPCNode
from .common_types import IStorable, Storable
from .utils import round_digits, Number


class TestJobConfig(Storable, metaclass=abc.ABCMeta):
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.reliable_info_time_range = None  # type: Tuple[int, int]
        self.vals = OrderedDict()  # type: Dict[str, Any]

    @property
    def storage_id(self) -> str:
        return "{}_{}".format(self.summary, self.idx)

    @abc.abstractproperty
    def characterized_tuple(self) -> Tuple:
        pass

    @abc.abstractproperty
    def summary(self, *excluded_fields) -> str:
        pass

    @abc.abstractproperty
    def long_summary(self, *excluded_fields) -> str:
        pass


class TestSuiteConfig(IStorable):
    """
    Test suite input configuration.

    test_type - test type name
    params - parameters from yaml file for this test
    run_uuid - UUID to be used to create file names & Co
    nodes - nodes to run tests on
    remote_dir - directory on nodes to be used for local files
    """
    def __init__(self,
                 test_type: str,
                 params: Dict[str, Any],
                 run_uuid: str,
                 nodes: List[IRPCNode],
                 remote_dir: str,
                 idx: int) -> None:
        self.test_type = test_type
        self.params = params
        self.run_uuid = run_uuid
        self.nodes = nodes
        self.nodes_ids = [node.node_id for node in nodes]
        self.remote_dir = remote_dir
        self.storage_id = "{}_{}".format(self.test_type, idx)

    def __eq__(self, o: object) -> bool:
        if type(o) is not self.__class__:
            return False

        other = cast(TestSuiteConfig, o)

        return (self.test_type == other.test_type and
                self.params == other.params and
                set(self.nodes_ids) == set(other.nodes_ids))

    def raw(self) -> Dict[str, Any]:
        res = self.__dict__.copy()
        del res['nodes']
        del res['run_uuid']
        del res['remote_dir']
        return res

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'IStorable':
        obj = cls.__new__(cls)
        data = data.copy()
        data['nodes'] = None
        data['run_uuid'] = None
        data['remote_dir'] = None
        obj.__dict__.update(data)
        return obj


class DataSource:
    def __init__(self,
                 suite_id: str = None,
                 job_id: str = None,
                 node_id: str = None,
                 dev: str = None,
                 sensor: str = None,
                 tag: str = None) -> None:
        self.suite_id = suite_id
        self.job_id = job_id
        self.node_id = node_id
        self.dev = dev
        self.sensor = sensor
        self.tag = tag

    def __call__(self, **kwargs) -> 'DataSource':
        dct = self.__dict__.copy()
        dct.update(kwargs)
        return self.__class__(**dct)

    def __str__(self) -> str:
        return "{0.suite_id}.{0.job_id}/{0.node_id}/{0.dev}.{0.sensor}.{0.tag}".format(self)

    def __repr__(self) -> str:
        return str(self)


class TimeSeries:
    """Data series from sensor - either system sensor or from load generator tool (e.g. fio)"""

    def __init__(self,
                 name: str,
                 raw: Optional[bytes],
                 data: numpy.array,
                 times: numpy.array,
                 second_axis_size: int = 1,
                 source: DataSource = None) -> None:

        # Sensor name. Typically DEV_NAME.METRIC
        self.name = name

        # Time series times and values. Time in ms from Unix epoch.
        self.times = times
        self.data = data

        # Not equal to 1 in case of 2d sensors, like latency, when each measurement is a histogram.
        self.second_axis_size = second_axis_size

        # Raw sensor data (is provided). Like log file for fio iops/bw/lat.
        self.raw = raw

        self.source = source

    def __str__(self) -> str:
        res = "TS({}):\n".format(self.name)
        res += "    source={}\n".format(self.source)
        res += "    times_size={}\n".format(len(self.times))
        res += "    data_size={}\n".format(len(self.data))
        res += "    data_shape={}x{}\n".format(len(self.data) // self.second_axis_size, self.second_axis_size)
        return res

    def __repr__(self) -> str:
        return str(self)


# (node_name, source_dev, metric_name) => metric_results
JobMetrics = Dict[Tuple[str, str, str], TimeSeries]


class StatProps(IStorable):
    "Statistic properties for timeseries with unknown data distribution"
    def __init__(self, data: numpy.array) -> None:
        self.perc_99 = None  # type: float
        self.perc_95 = None  # type: float
        self.perc_90 = None  # type: float
        self.perc_50 = None   # type: float

        self.min = None  # type: Number
        self.max = None  # type: Number

        # bin_center: bin_count
        self.bins_populations = None # type: numpy.array
        self.bins_mids = None  # type: numpy.array
        self.data = data

    def __str__(self) -> str:
        res = ["{}(size = {}):".format(self.__class__.__name__, len(self.data))]
        for name in ["perc_50", "perc_90", "perc_95", "perc_99"]:
            res.append("    {} = {}".format(name, round_digits(getattr(self, name))))
        res.append("    range {} {}".format(round_digits(self.min), round_digits(self.max)))
        return "\n".join(res)

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        del data['data']
        data['bins_mids'] = list(self.bins_mids)
        data['bins_populations'] = list(self.bins_populations)
        return data

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'StatProps':
        data['bins_mids'] = numpy.array(data['bins_mids'])
        data['bins_populations'] = numpy.array(data['bins_populations'])
        data['data'] = None
        res = cls.__new__(cls)
        res.__dict__.update(data)
        return res


class HistoStatProps(StatProps):
    """Statistic properties for 2D timeseries with unknown data distribution and histogram as input value.
    Used for latency"""
    def __init__(self, data: numpy.array, second_axis_size: int) -> None:
        self.second_axis_size = second_axis_size
        StatProps.__init__(self, data)


class NormStatProps(StatProps):
    "Statistic properties for timeseries with normal data distribution. Used for iops/bw"
    def __init__(self, data: numpy.array) -> None:
        StatProps.__init__(self, data)

        self.average = None  # type: float
        self.deviation = None  # type: float
        self.confidence = None  # type: float
        self.confidence_level = None  # type: float
        self.normtest = None  # type: NormaltestResult
        self.skew = None  # type: float
        self.kurt = None  # type: float

    def __str__(self) -> str:
        res = ["NormStatProps(size = {}):".format(len(self.data)),
               "    distr = {} ~ {}".format(round_digits(self.average), round_digits(self.deviation)),
               "    confidence({0.confidence_level}) = {1}".format(self, round_digits(self.confidence)),
               "    perc_50 = {}".format(round_digits(self.perc_50)),
               "    perc_90 = {}".format(round_digits(self.perc_90)),
               "    perc_95 = {}".format(round_digits(self.perc_95)),
               "    perc_99 = {}".format(round_digits(self.perc_99)),
               "    range {} {}".format(round_digits(self.min), round_digits(self.max)),
               "    normtest = {0.normtest}".format(self),
               "    skew ~ kurt = {0.skew} ~ {0.kurt}".format(self)]
        return "\n".join(res)

    def raw(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data['normtest'] = (data['nortest'].statistic, data['nortest'].pvalue)
        del data['data']
        data['bins_mids'] = list(self.bins_mids)
        data['bins_populations'] = list(self.bins_populations)
        return data

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'NormStatProps':
        data['normtest'] = NormaltestResult(*data['normtest'])
        obj = StatProps.fromraw(data)
        obj.__class__ = cls
        return cast('NormStatProps', obj)


JobStatMetrics = Dict[Tuple[str, str, str], StatProps]


class TestJobResult:
    """Contains done test job information"""

    def __init__(self,
                 info: TestJobConfig,
                 begin_time: int,
                 end_time: int,
                 raw: JobMetrics) -> None:
        self.info = info
        self.run_interval = (begin_time, end_time)
        self.raw = raw  # type: JobMetrics
        self.processed = None  # type: JobStatMetrics


class IResultStorage(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def put_or_check_suite(self, suite: TestSuiteConfig) -> None:
        pass

    @abc.abstractmethod
    def put_job(self, suite: TestSuiteConfig, job: TestJobConfig) -> None:
        pass

    @abc.abstractmethod
    def put_ts(self, ts: TimeSeries) -> None:
        pass

    @abc.abstractmethod
    def put_extra(self, data: bytes, source: DataSource) -> None:
        pass

    @abc.abstractmethod
    def put_stat(self, data: StatProps, source: DataSource) -> None:
        pass

    @abc.abstractmethod
    def get_stat(self, stat_cls: Type[StatProps], source: DataSource) -> StatProps:
        pass

    @abc.abstractmethod
    def iter_suite(self, suite_type: str = None) -> Iterator[TestSuiteConfig]:
        pass

    @abc.abstractmethod
    def iter_job(self, suite: TestSuiteConfig) -> Iterator[TestJobConfig]:
        pass

    @abc.abstractmethod
    def iter_ts(self, suite: TestSuiteConfig, job: TestJobConfig) -> Iterator[TimeSeries]:
        pass

    # return path to file to be inserted into report
    @abc.abstractmethod
    def put_plot_file(self, data: bytes, source: DataSource) -> str:
        pass
