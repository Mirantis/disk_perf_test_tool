import array
from typing import Dict, List, Any, Tuple, Optional


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


class FullTestResult:
    test_info = None  # type: TestInfo

    # TODO(koder): array.array or numpy.array?
    # {(node_id, perf_metrics_name): values}
    performance_data = None  # type: Dict[Tuple[str, str], List[int]]

    # {(node_id, perf_metrics_name): values}
    sensors_data = None  # type: Dict[Tuple[str, str, str], SensorInfo]
