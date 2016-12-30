import array
from typing import Dict, List, Any, Optional, Tuple, cast


import numpy
from scipy.stats.mstats_basic import NormaltestResult


from .node_interfaces import IRPCNode
from .istorable import IStorable, Storable
from .utils import round_digits, Number


class TestJobConfig(Storable):
    def __init__(self) -> None:
        self.summary = None  # type: str


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
                 remote_dir: str) -> None:
        self.test_type = test_type
        self.params = params
        self.run_uuid = run_uuid
        self.nodes = nodes
        self.nodes_ids = [node.info.node_id() for node in nodes]
        self.remote_dir = remote_dir

    def __eq__(self, other: 'TestSuiteConfig') -> bool:
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


class TimeSeries:
    """Data series from sensor - either system sensor or from load generator tool (e.g. fio)"""

    def __init__(self,
                 name: str,
                 raw: Optional[bytes],
                 data: array.array,
                 times: array.array,
                 second_axis_size: int = 1,
                 bins_edges: List[float] = None) -> None:

        # Sensor name. Typically DEV_NAME.METRIC
        self.name = name

        # Time series times and values. Time in ms from Unix epoch.
        self.times = times  # type: List[int]
        self.data = data  # type: List[int]

        # Not equal to 1 in case of 2d sensors, like latency, when each measurement is a histogram.
        self.second_axis_size = second_axis_size

        # Raw sensor data (is provided). Like log file for fio iops/bw/lat.
        self.raw = raw

        # bin edges for historgam timeseries
        self.bins_edges = bins_edges


# (node_name, source_dev, metric_name) => metric_results
JobMetrics = Dict[Tuple[str, str, str], TimeSeries]


class StatProps(IStorable):
    "Statistic properties for timeseries with unknown data distribution"
    def __init__(self, data: List[Number]) -> None:
        self.perc_99 = None  # type: float
        self.perc_95 = None  # type: float
        self.perc_90 = None  # type: float
        self.perc_50 = None   # type: float

        self.min = None  # type: Number
        self.max = None  # type: Number

        # bin_center: bin_count
        self.bins_populations = None  # type: List[int]
        self.bins_edges = None  # type: List[float]
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
        data['bins_edges'] = list(self.bins_edges)
        data['bins_populations'] = list(self.bins_populations)
        return data

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'StatProps':
        data['bins_edges'] = numpy.array(data['bins_edges'])
        data['bins_populations'] = numpy.array(data['bins_populations'])
        res = cls.__new__(cls)
        res.__dict__.update(data)
        return res


class HistoStatProps(StatProps):
    """Statistic properties for 2D timeseries with unknown data distribution and histogram as input value.
    Used for latency"""
    def __init__(self, data: List[Number], second_axis_size: int) -> None:
        self.second_axis_size = second_axis_size
        StatProps.__init__(self, data)


class NormStatProps(StatProps):
    "Statistic properties for timeseries with normal data distribution. Used for iops/bw"
    def __init__(self, data: List[Number]) -> None:
        StatProps.__init__(self, data)

        self.average = None  # type: float
        self.deviation = None  # type: float
        self.confidence = None  # type: float
        self.confidence_level = None  # type: float
        self.normtest = None  # type: NormaltestResult

    def __str__(self) -> str:
        res = ["NormStatProps(size = {}):".format(len(self.data)),
               "    distr = {} ~ {}".format(round_digits(self.average), round_digits(self.deviation)),
               "    confidence({0.confidence_level}) = {1}".format(self, round_digits(self.confidence)),
               "    perc_50 = {}".format(round_digits(self.perc_50)),
               "    perc_90 = {}".format(round_digits(self.perc_90)),
               "    perc_95 = {}".format(round_digits(self.perc_95)),
               "    perc_99 = {}".format(round_digits(self.perc_99)),
               "    range {} {}".format(round_digits(self.min), round_digits(self.max)),
               "    normtest = {0.normtest}".format(self)]
        return "\n".join(res)

    def raw(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data['normtest'] = (data['nortest'].statistic, data['nortest'].pvalue)
        data['bins_edges'] = list(self.bins_edges)
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
