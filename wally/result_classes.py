import abc
import copy
from typing import Dict, List, Any, Optional, Tuple, cast, Type, Iterator, NamedTuple


import numpy
from scipy.stats.mstats_basic import NormaltestResult

from .suits.job import JobConfig
from .node_interfaces import IRPCNode
from .common_types import Storable
from .utils import round_digits, Number


class SuiteConfig(Storable):
    """
    Test suite input configuration.

    test_type - test type name
    params - parameters from yaml file for this test
    run_uuid - UUID to be used to create file names & Co
    nodes - nodes to run tests on
    remote_dir - directory on nodes to be used for local files
    """
    __ignore_fields__ = ['nodes', 'run_uuid', 'remote_dir']

    def __init__(self,
                 test_type: str,
                 params: Dict[str, Any],
                 run_uuid: str,
                 nodes: List[IRPCNode],
                 remote_dir: str,
                 idx: int,
                 keep_raw_files: bool) -> None:
        self.test_type = test_type
        self.params = params
        self.run_uuid = run_uuid
        self.nodes = nodes
        self.nodes_ids = [node.node_id for node in nodes]
        self.remote_dir = remote_dir
        self.keep_raw_files = keep_raw_files

        if 'load' in self.params:
            self.storage_id = "{}_{}_{}".format(self.test_type, self.params['load'], idx)
        else:
            self.storage_id = "{}_{}".format(self.test_type, idx)

    def __eq__(self, o: object) -> bool:
        if type(o) is not self.__class__:
            return False

        other = cast(SuiteConfig, o)

        return (self.test_type == other.test_type and
                self.params == other.params and
                set(self.nodes_ids) == set(other.nodes_ids))


class DataSource:
    def __init__(self,
                 suite_id: str = None,
                 job_id: str = None,
                 node_id: str = None,
                 sensor: str = None,
                 dev: str = None,
                 metric: str = None,
                 tag: str = None) -> None:
        self.suite_id = suite_id
        self.job_id = job_id
        self.node_id = node_id
        self.sensor = sensor
        self.dev = dev
        self.metric = metric
        self.tag = tag

    @property
    def metric_fqdn(self) -> str:
        return "{0.sensor}.{0.dev}.{0.metric}".format(self)

    def __call__(self, **kwargs) -> 'DataSource':
        dct = self.__dict__.copy()
        dct.update(kwargs)
        return self.__class__(**dct)

    def __str__(self) -> str:
        return ("suite={0.suite_id},job={0.job_id},node={0.node_id}," +
                "path={0.sensor}.{0.dev}.{0.metric},tag={0.tag}").format(self)

    def __repr__(self) -> str:
        return str(self)

    @property
    def tpl(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str],
                           Optional[str], Optional[str], Optional[str]]:
        return self.suite_id, self.job_id, self.node_id, self.sensor, self.dev, self.metric, self.tag

    def __eq__(self, o: object) -> bool:
        return self.tpl == cast(DataSource, o).tpl

    def __hash__(self) -> int:
        return hash(self.tpl)


class TimeSeries:
    """Data series from sensor - either system sensor or from load generator tool (e.g. fio)"""

    def __init__(self,
                 name: str,
                 raw: Optional[bytes],
                 data: numpy.ndarray,
                 times: numpy.ndarray,
                 units: str,
                 source: DataSource,
                 time_units: str = 'us',
                 raw_tag: str = 'txt',
                 histo_bins: numpy.ndarray = None) -> None:

        # Sensor name. Typically DEV_NAME.METRIC
        self.name = name

        # units for data
        self.units = units

        # units for time
        self.time_units = time_units

        # Time series times and values. Time in ms from Unix epoch.
        self.times = times
        self.data = data

        # Raw sensor data (is provided). Like log file for fio iops/bw/lat.
        self.raw = raw
        self.raw_tag = raw_tag
        self.source = source
        self.histo_bins = histo_bins

    def __str__(self) -> str:
        res = "TS({}):\n".format(self.name)
        res += "    source={}\n".format(self.source)
        res += "    times_size={}\n".format(len(self.times))
        res += "    data_shape={}\n".format(*self.data.shape)
        return res

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> 'TimeSeries':
        cp = copy.copy(self)
        cp.times = self.times.copy()
        cp.data = self.data.copy()
        cp.source = self.source()
        return cp


# (node_name, source_dev, metric_name) => metric_results
JobMetrics = Dict[Tuple[str, str, str], TimeSeries]


class StatProps(Storable):
    "Statistic properties for timeseries with unknown data distribution"

    __ignore_fields__ = ['data']

    def __init__(self, data: numpy.array, units: str) -> None:
        self.perc_99 = None  # type: float
        self.perc_95 = None  # type: float
        self.perc_90 = None  # type: float
        self.perc_50 = None   # type: float
        self.perc_10 = None  # type: float
        self.perc_5 = None   # type: float
        self.perc_1 = None   # type: float

        self.min = None  # type: Number
        self.max = None  # type: Number

        # bin_center: bin_count
        self.log_bins = False
        self.bins_populations = None # type: numpy.array

        # bin edges, one more element that in bins_populations
        self.bins_edges = None  # type: numpy.array

        self.data = data
        self.units = units

    def __str__(self) -> str:
        res = ["{}(size = {}):".format(self.__class__.__name__, len(self.data))]
        for name in ["perc_1", "perc_5", "perc_10", "perc_50", "perc_90", "perc_95", "perc_99"]:
            res.append("    {} = {}".format(name, round_digits(getattr(self, name))))
        res.append("    range {} {}".format(round_digits(self.min), round_digits(self.max)))
        return "\n".join(res)

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        data = super().raw()
        data['bins_mids'] = list(data['bins_mids'])
        data['bins_populations'] = list(data['bins_populations'])
        return data

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'StatProps':
        data['bins_mids'] = numpy.array(data['bins_mids'])
        data['bins_populations'] = numpy.array(data['bins_populations'])
        return cast(StatProps, super().fromraw(data))


class HistoStatProps(StatProps):
    """Statistic properties for 2D timeseries with unknown data distribution and histogram as input value.
    Used for latency"""
    def __init__(self, data: numpy.array, units: str) -> None:
        StatProps.__init__(self, data, units)


class NormStatProps(StatProps):
    "Statistic properties for timeseries with normal data distribution. Used for iops/bw"
    def __init__(self, data: numpy.array, units: str) -> None:
        StatProps.__init__(self, data, units)

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
               "    perc_1 = {}".format(round_digits(self.perc_1)),
               "    perc_5 = {}".format(round_digits(self.perc_5)),
               "    perc_10 = {}".format(round_digits(self.perc_10)),
               "    perc_50 = {}".format(round_digits(self.perc_50)),
               "    perc_90 = {}".format(round_digits(self.perc_90)),
               "    perc_95 = {}".format(round_digits(self.perc_95)),
               "    perc_99 = {}".format(round_digits(self.perc_99)),
               "    range {} {}".format(round_digits(self.min), round_digits(self.max)),
               "    normtest = {0.normtest}".format(self),
               "    skew ~ kurt = {0.skew} ~ {0.kurt}".format(self)]
        return "\n".join(res)

    def raw(self) -> Dict[str, Any]:
        data = super().raw()
        data['normtest'] = (data['nortest'].statistic, data['nortest'].pvalue)
        return data

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'NormStatProps':
        data['normtest'] = NormaltestResult(*data['normtest'])
        return cast(NormStatProps, super().fromraw(data))


JobStatMetrics = Dict[Tuple[str, str, str], StatProps]


class JobResult:
    """Contains done test job information"""

    def __init__(self,
                 info: JobConfig,
                 begin_time: int,
                 end_time: int,
                 raw: JobMetrics) -> None:
        self.info = info
        self.run_interval = (begin_time, end_time)
        self.raw = raw  # type: JobMetrics
        self.processed = None  # type: JobStatMetrics


ArrayData = NamedTuple("ArrayData",
                       [('header', List[str]),
                        ('histo_bins', Optional[numpy.ndarray]),
                        ('data', Optional[numpy.ndarray])])


class IResultStorage(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def append_sensor(self, data: numpy.array, ds: DataSource, units: str, histo_bins: numpy.ndarray = None) -> None:
        pass

    @abc.abstractmethod
    def load_sensor(self, ds: DataSource) -> TimeSeries:
        pass

    @abc.abstractmethod
    def iter_sensors(self, ds: DataSource) -> Iterator[TimeSeries]:
        pass

    @abc.abstractmethod
    def put_or_check_suite(self, suite: SuiteConfig) -> None:
        pass

    @abc.abstractmethod
    def put_job(self, suite: SuiteConfig, job: JobConfig) -> None:
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
    def iter_suite(self, suite_type: str = None) -> Iterator[SuiteConfig]:
        pass

    @abc.abstractmethod
    def iter_job(self, suite: SuiteConfig) -> Iterator[JobConfig]:
        pass

    @abc.abstractmethod
    def iter_ts(self, suite: SuiteConfig, job: JobConfig) -> Iterator[TimeSeries]:
        pass

    # return path to file to be inserted into report
    @abc.abstractmethod
    def put_plot_file(self, data: bytes, source: DataSource) -> str:
        pass
