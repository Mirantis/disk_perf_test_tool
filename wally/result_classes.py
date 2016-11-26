from typing import Union, Dict, List, Any, Tuple

# Stores test result for integral value, which
# can be expressed as a single value for given time period,
# like IO, BW, etc.
TimeSeriesIntegral = List[float]


# Stores test result for value, which
# requires distribution to be stored for any time period,
# like latency.
TimeSeriesHistogram = List[List[float]]


TimeSeries = Union[TimeSeriesIntegral, TimeSeriesHistogram]
RawTestResults = Dict[str, TimeSeries]


class SensorInfo:
    """Holds information from a single sensor from a single node"""
    node_id = None  # type: str
    source_id = None  # type: str
    sensor_name = None  # type: str
    begin_time = None  # type: int
    end_time = None  # type: int
    data = None  # type: TimeSeries

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


class FullTestResult:
    test_info = None  # type: TestInfo

    # TODO(koder): array.array or numpy.array?
    # {(node_id, perf_metrics_name): values}
    performance_data = None  # type: Dict[Tuple[str, str], TimeSeries]

    # {(node_id, perf_metrics_name): values}
    sensors_data = None  # type: Dict[Tuple[str, str, str], SensorInfo]
