import abc
from typing import Dict, List, Any, Tuple, cast, Type, Iterator, Union

from cephlib.numeric_types import TimeSeries, DataSource
from cephlib.statistic import StatProps
from cephlib.istorage import IImagesStorage, Storable, ISensorStorage

from .suits.job import JobConfig
from .node_interfaces import IRPCNode, NodeInfo


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


# (node_name, source_dev, metric_name) => metric_results
JobMetrics = Dict[Tuple[str, str, str], TimeSeries]
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


class IResultStorage(ISensorStorage, IImagesStorage, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def put_or_check_suite(self, suite: SuiteConfig) -> None:
        pass

    @abc.abstractmethod
    def put_job(self, suite: SuiteConfig, job: JobConfig) -> None:
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

    # return path to file to be inserted into report
    @abc.abstractmethod
    def put_plot_file(self, data: bytes, source: DataSource) -> str:
        pass

    @abc.abstractmethod
    def get_job_info(self, suite: SuiteConfig, job: JobConfig, key: str) -> Any:
        pass

    @abc.abstractmethod
    def get_ts(self, ds: DataSource) -> TimeSeries:
        pass

    @abc.abstractmethod
    def put_ts(self, ts: TimeSeries) -> None:
        pass

    @abc.abstractmethod
    def iter_ts(self, **ds_parts) -> Iterator[DataSource]:
        pass

    @abc.abstractmethod
    def put_job_info(self, suite: SuiteConfig, job: JobConfig, key: str, data: Any) -> None:
        pass

    @abc.abstractmethod
    def find_nodes(self, roles: Union[str, List[str]]) -> List[NodeInfo]:
        pass