import abc
import array
from typing import Dict, List, Any, Optional

import numpy
from scipy import stats


from .utils import IStorable, Number, round_digits


class TimeSerie:
    name = None  # type: str
    start_at = None  # type: int
    step = None  # type: int
    data = None  # type: List[int]
    second_axis_size = None  # type: int
    raw = None  # type: Optional[bytes]

    def __init__(self, name: str, raw: Optional[bytes], second_axis_size: int,
                 start_at: int, step: int, data: array.array) -> None:
        self.name = name
        self.start_at = start_at
        self.step = step
        self.second_axis_size = second_axis_size
        self.data = data # type: ignore
        self.raw = raw

    def meta(self) -> Dict[str, Any]:
        return {
            "start_at": self.start_at,
            "step": self.step,
            "second_axis_size": self.second_axis_size
        }


class SensorInfo:
    """Holds information from a single sensor from a single node"""
    node_id = None  # type: str
    source_id = None  # type: str
    sensor_name = None  # type: str
    begin_time = None  # type: int
    end_time = None  # type: int
    data = None  # type: List[int]

    def __init__(self, node_id: str, source_id: str, sensor_name: str) -> None:
        self.node_id = node_id
        self.source_id = source_id
        self.sensor_name = sensor_name


class TestInfo:
    """Contains done test information"""
    name = None  # type: str
    iteration_name = None # type: str
    nodes = None  # type: List[str]
    start_time = None  # type: int
    stop_time = None  # type: int
    params = None  # type: Dict[str, Any]
    config = None  # type: str
    node_ids = None # type: List[str]


class NodeTestResults:
    name = None  # type: str
    node_id = None  # type: str
    summary = None  # type: str

    load_start_at = None  # type: int
    load_stop_at = None  # type: int

    series = None  # type: Dict[str, TimeSerie]

    def __init__(self, name: str, node_id: str, summary: str) -> None:
        self.name = name
        self.node_id = node_id
        self.summary = summary
        self.series = {}
        self.extra_logs = {}  # type: Dict[str, bytes]


class NormStatProps(IStorable):
    "Statistic properties for timeserie"
    def __init__(self, data: List[Number]) -> None:
        self.average = None  # type: float
        self.deviation = None  # type: float
        self.confidence = None  # type: float
        self.confidence_level = None  # type: float

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

        self.normtest = None  # type: Any

    def __str__(self) -> str:
        res = ["StatProps(size = {}):".format(len(self.data)),
               "    distr = {} ~ {}".format(round_digits(self.average), round_digits(self.deviation)),
               "    confidence({0.confidence_level}) = {1}".format(self, round_digits(self.confidence)),
               "    perc_50 = {}".format(round_digits(self.perc_50)),
               "    perc_90 = {}".format(round_digits(self.perc_90)),
               "    perc_95 = {}".format(round_digits(self.perc_95)),
               "    perc_99 = {}".format(round_digits(self.perc_99)),
               "    range {} {}".format(round_digits(self.min), round_digits(self.max)),
               "    normtest = {0.normtest}".format(self)]
        return "\n".join(res)

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data['nortest'] = (data['nortest'].statistic, data['nortest'].pvalue)
        data['bins_edges'] = list(self.bins_edges)
        return data

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'NormStatProps':
        data['nortest'] = stats.mstats.NormaltestResult(data['nortest'].statistic, data['nortest'].pvalue)
        data['bins_edges'] = numpy.array(data['bins_edges'])
        res = cls.__new__(cls)
        res.__dict__.update(data)
        return res


class ProcessedTestResults:
    def __init__(self, info: Dict[str, Any],
                 metrics: Dict[str, NormStatProps]) -> None:
        self.test = info['test']
        self.profile = info['profile']
        self.suite = info['suite']
        self.name = "{0.suite}.{0.test}.{0.profile}".format(self)
        self.info = info
        self.metrics = metrics  # mapping {metrics_name: StatProps}


# class FullTestResult:
#     test_info = None  # type: TestInfo
#
#     # TODO(koder): array.array or numpy.array?
#     # {(node_id, perf_metrics_name): values}
#     performance_data = None  # type: Dict[Tuple[str, str], List[int]]
#
#     # {(node_id, perf_metrics_name): values}
#     sensors_data = None  # type: Dict[Tuple[str, str, str], SensorInfo]


