import re
import abc
import time
import array
import struct
import logging
import os.path
import datetime
from typing import Any, List, Optional, Callable, cast, Iterator, Tuple, Iterable

from concurrent.futures import ThreadPoolExecutor, wait, Future

from ..utils import StopTestError, sec_to_str, get_time_interval_printable_info
from ..node_interfaces import IRPCNode
from ..storage import Storage
from ..result_classes import TestSuiteConfig, TestJobConfig, JobMetrics, TimeSeries


logger = logging.getLogger("wally")


__doc__ = "Contains base classes for performance tests"


class ResultStorage:
    ts_header_format = "!IIIcc"

    def __init__(self, storage: Storage, job_config_cls: type) -> None:
        self.storage = storage
        self.job_config_cls = job_config_cls

    def get_suite_root(self, suite_type: str, idx: int) -> str:
        return "results/{}_{}".format(suite_type, idx)

    def get_job_root(self, suite_root: str, summary: str, run_id: int) -> str:
        return "{}/{}_{}".format(suite_root, summary, run_id)

    # store
    def put_suite_config(self, config: TestSuiteConfig, root: str) -> None:
        self.storage.put(config, root, "config.yml")

    def put_job_config(self, config: TestJobConfig, root: str) -> None:
        self.storage.put(config, root, "config.yml")

    def get_suite_config(self, suite_root: str) -> TestSuiteConfig:
        return self.storage.load(TestSuiteConfig, suite_root, "config.yml")

    def get_job_node_prefix(self, job_root_path: str, node_id: str) -> str:
        return "{}/{}".format(job_root_path, node_id)

    def get_ts_path(self, job_root_path: str, node_id: str, dev: str, sensor_name: str) -> str:
        return "{}_{}.{}".format(self.get_job_node_prefix(job_root_path, node_id), dev, sensor_name)

    def put_ts(self, ts: TimeSeries, job_root_path: str, node_id: str, dev: str, sensor_name: str) -> None:
        # TODO: check that 'metrics', 'dev' and 'node_id' match required patterns
        root_path = self.get_ts_path(job_root_path, node_id, dev, sensor_name)

        if len(ts.data) / ts.second_axis_size != len(ts.times):
            logger.error("Unbalanced time series data. Array size has % elements, while time size has %",
                         len(ts.data) / ts.second_axis_size, len(ts.times))
            raise StopTestError()

        with self.storage.get_fd(root_path, "cb") as fd:
            header = struct.pack(self.ts_header_format,
                                 ts.second_axis_size,
                                 len(ts.data),
                                 len(ts.times),
                                 cast(array.array, ts.data).typecode.encode("ascii"),
                                 cast(array.array, ts.times).typecode.encode("ascii"))
            fd.write(header)
            cast(array.array, ts.data).tofile(fd)
            cast(array.array, ts.times).tofile(fd)

        if ts.raw is not None:
            self.storage.put_raw(ts.raw, root_path + ":raw")

    def put_extra(self, job_root: str, node_id: str, key: str, data: bytes) -> None:
        self.storage.put_raw(data, job_root, node_id + "_" + key)

    def list_suites(self) -> Iterator[Tuple[TestSuiteConfig, str]]:
        """iterates over (suite_name, suite_id, suite_root_path)
        primary this function output should be used as input into list_jobs_in_suite method
        """
        ts_re = re.compile(r"[a-zA-Z]+_\d+$")
        for is_file, name in self.storage.list("results"):
            if not is_file:
                rr = ts_re.match(name)
                if rr:
                    path = "results/" + name
                    yield self.get_suite_config(path), path

    def list_jobs_in_suite(self, suite_root_path: str) -> Iterator[Tuple[TestJobConfig, str, int]]:
        """iterates over (job_summary, job_root_path)
        primary this function output should be used as input into list_ts_in_job method
        """
        ts_re = re.compile(r"(?P<job_summary>[a-zA-Z0-9]+)_(?P<id>\d+)$")
        for is_file, name in self.storage.list(suite_root_path):
            if is_file:
                continue
            rr = ts_re.match(name)
            if rr:
                config_path = "{}/{}/config.yml".format(suite_root_path, name)
                if config_path in self.storage:
                    cfg = self.storage.load(self.job_config_cls, config_path)
                    yield cfg, "{}/{}".format(suite_root_path, name), int(rr.group("id"))

    def list_ts_in_job(self, job_root_path: str) -> Iterator[Tuple[str, str, str]]:
        """iterates over (node_id, device_name, sensor_name)
        primary this function output should be used as input into load_ts method
        """
        # TODO: check that all TS files available
        ts_re = re.compile(r"(?P<node_id>\d+\.\d+\.\d+\.\d+:\d+)_(?P<dev>[^.]+)\.(?P<sensor>[a-z_]+)$")
        already_found = set()
        for is_file, name in self.storage.list(job_root_path):
            if not is_file:
                continue
            rr = ts_re.match(name)
            if rr:
                key = (rr.group("node_id"), rr.group("dev"), rr.group("sensor"))
                if key not in already_found:
                    already_found.add(key)
                    yield key

    def load_ts(self, root_path: str, node_id: str, dev: str, sensor_name: str) -> TimeSeries:
        path = self.get_ts_path(root_path, node_id, dev, sensor_name)

        with self.storage.get_fd(path, "rb") as fd:
            header = fd.read(struct.calcsize(self.ts_header_format))
            second_axis_size, data_sz, time_sz, data_typecode, time_typecode = \
                struct.unpack(self.ts_header_format, header)

            data = array.array(data_typecode.decode("ascii"))
            times = array.array(time_typecode.decode("ascii"))

            data.fromfile(fd, data_sz)
            times.fromfile(fd, time_sz)

            # calculate number of elements
            return TimeSeries("{}.{}".format(dev, sensor_name),
                              raw=None,
                              data=data,
                              times=times,
                              second_axis_size=second_axis_size)


class PerfTest(metaclass=abc.ABCMeta):
    """Base class for all tests"""
    name = None  # type: str
    max_retry = 3
    retry_time = 30
    job_config_cls = None  # type: type

    def __init__(self, storage: Storage, config: TestSuiteConfig, idx: int, on_idle: Callable[[], None] = None) -> None:
        self.config = config
        self.stop_requested = False
        self.sorted_nodes_ids = sorted(node.info.node_id() for node in self.config.nodes)
        self.on_idle = on_idle
        self.storage = storage
        self.rstorage = ResultStorage(self.storage, self.job_config_cls)
        self.idx = idx

    def request_stop(self) -> None:
        self.stop_requested = True

    def join_remote(self, path: str) -> str:
        return os.path.join(self.config.remote_dir, path)

    @abc.abstractmethod
    def run(self) -> None:
        pass

    @abc.abstractmethod
    def format_for_console(self, data: Any) -> str:
        pass


class ThreadedTest(PerfTest, metaclass=abc.ABCMeta):
    """Base class for tests, which spawn separated thread for each node"""

    # max allowed time difference between starts and stops of run of the same test on different test nodes
    # used_max_diff = max((min_run_time * max_rel_time_diff), max_time_diff)
    max_time_diff = 5
    max_rel_time_diff = 0.05
    load_profile_name = None  # type: str

    def __init__(self, *args, **kwargs) -> None:
        PerfTest.__init__(self, *args, **kwargs)
        self.job_configs = [None]  # type: List[Optional[TestJobConfig]]
        self.suite_root_path = self.rstorage.get_suite_root(self.config.test_type, self.idx)

    @abc.abstractmethod
    def get_expected_runtime(self, iter_cfg: TestJobConfig) -> Optional[int]:
        pass

    def get_not_done_stages(self) -> Iterable[Tuple[int, TestJobConfig]]:
        all_jobs = dict(enumerate(self.job_configs))
        for db_config, path, jid in self.rstorage.list_jobs_in_suite(self.suite_root_path):
            if jid in all_jobs:
                job_config = all_jobs[jid]
                if job_config != db_config:
                    logger.error("Test info at path '%s/config' is not equal to expected config for iteration %s.%s." +
                                 " Maybe configuration was changed before test was restarted. " +
                                 "DB cfg is:\n    %s\nExpected cfg is:\n    %s\nFix DB or rerun test from beginning",
                                 path, self.name, job_config.summary,
                                 str(db_config).replace("\n", "\n    "),
                                 str(job_config).replace("\n", "\n    "))
                    raise StopTestError()

                logger.info("Test iteration %s.%s found in storage and will be skipped",
                            self.name, job_config.summary)
                del all_jobs[jid]
        return all_jobs.items()

    def run(self) -> None:
        try:
            cfg = self.rstorage.get_suite_config(self.suite_root_path)
        except KeyError:
            cfg = None

        if cfg is not None and cfg != self.config:
            logger.error("Current suite %s config is not equal to found in storage at %s",
                         self.config.test_type, self.suite_root_path)
            raise StopTestError()

        not_in_storage = list(self.get_not_done_stages())

        if not not_in_storage:
            logger.info("All test iteration in storage already. Skip test")
            return

        self.rstorage.put_suite_config(self.config, self.suite_root_path)

        logger.debug("Run test %s with profile %r on nodes %s.", self.name,
                                                                 self.load_profile_name,
                                                                 ",".join(self.sorted_nodes_ids))
        logger.debug("Prepare nodes")


        with ThreadPoolExecutor(len(self.config.nodes)) as pool:
            # config nodes
            list(pool.map(self.config_node, self.config.nodes))

            run_times = [self.get_expected_runtime(job_config) for _, job_config in not_in_storage]

            if None not in run_times:
                # +5% - is a rough estimation for additional operations
                expected_run_time = int(sum(run_times) * 1.05)

                exec_time_s, end_dt_s = get_time_interval_printable_info(expected_run_time)
                logger.info("Entire test should takes around %s and finished at %s", exec_time_s, end_dt_s)

            for run_id, job_config in not_in_storage:
                job_path = self.rstorage.get_job_root(self.suite_root_path, job_config.summary, run_id)

                jfutures = []  # type: List[Future]
                for idx in range(self.max_retry):
                    logger.debug("Prepare job %s", job_config.summary)

                    # prepare nodes for new iterations
                    wait([pool.submit(self.prepare_iteration, node, job_config) for node in self.config.nodes])

                    expected_job_time = self.get_expected_runtime(job_config)
                    exec_time_s, end_dt_s = get_time_interval_printable_info(expected_job_time)
                    logger.info("Job should takes around %s and finished at %s", exec_time_s, end_dt_s)

                    try:
                        jfutures = []
                        for node in self.config.nodes:
                            future = pool.submit(self.run_iteration, node, job_config, job_path)
                            jfutures.append(future)
                        # test completed successfully, stop retrying
                        break
                    except EnvironmentError:
                        if self.max_retry - 1 == idx:
                            logger.exception("Fio failed")
                            raise StopTestError()
                        logger.exception("During fio run")
                        logger.info("Sleeping %ss and retrying job", self.retry_time)
                        time.sleep(self.retry_time)

                start_times = []  # type: List[int]
                stop_times = []  # type: List[int]

                for future in jfutures:
                    for (node_id, dev, sensor_name), ts in future.result().items():
                        self.rstorage.put_ts(ts, job_path, node_id=node_id, dev=dev, sensor_name=sensor_name)

                        if len(ts.times) >= 2:
                            start_times.append(ts.times[0])
                            stop_times.append(ts.times[-1])

                if len(start_times) > 0:
                    min_start_time = min(start_times)
                    max_start_time = max(start_times)
                    min_stop_time = min(stop_times)
                    max_stop_time = max(stop_times)

                    max_allowed_time_diff = int((min_stop_time - max_start_time) * self.max_rel_time_diff)
                    max_allowed_time_diff = max(max_allowed_time_diff, self.max_time_diff)

                    if min_start_time + self.max_time_diff < max_allowed_time_diff:
                        logger.warning("Too large difference in %s:%s start time - %s. " +
                                       "Max recommended difference is %s",
                                       self.name, job_config.summary,
                                       max_start_time - min_start_time, self.max_time_diff)

                    if min_stop_time + self.max_time_diff < max_allowed_time_diff:
                        logger.warning("Too large difference in %s:%s stop time - %s. " +
                                       "Max recommended difference is %s",
                                       self.name, job_config.summary,
                                       max_start_time - min_start_time, self.max_time_diff)

                self.rstorage.put_job_config(job_config, job_path)
                self.storage.sync()

                if self.on_idle is not None:
                    self.on_idle()

    @abc.abstractmethod
    def config_node(self, node: IRPCNode) -> None:
        pass

    @abc.abstractmethod
    def prepare_iteration(self, node: IRPCNode, iter_config: TestJobConfig) -> None:
        pass

    @abc.abstractmethod
    def run_iteration(self, node: IRPCNode, iter_config: TestJobConfig, stor_prefix: str) -> JobMetrics:
        pass


class TwoScriptTest(ThreadedTest, metaclass=abc.ABCMeta):
    def __init__(self, *dt, **mp) -> None:
        ThreadedTest.__init__(self, *dt, **mp)
        self.prerun_script = self.config.params['prerun_script']
        self.run_script = self.config.params['run_script']
        self.prerun_tout = self.config.params.get('prerun_tout', 3600)
        self.run_tout = self.config.params.get('run_tout', 3600)
        self.iterations_configs = [None]

    def get_expected_runtime(self, iter_cfg: TestJobConfig) -> Optional[int]:
        return None

    def config_node(self, node: IRPCNode) -> None:
        node.copy_file(self.run_script, self.join_remote(self.run_script))
        node.copy_file(self.prerun_script, self.join_remote(self.prerun_script))

        cmd = self.join_remote(self.prerun_script)
        cmd += ' ' + self.config.params.get('prerun_opts', '')
        node.run(cmd, timeout=self.prerun_tout)

    def prepare_iteration(self, node: IRPCNode, iter_config: TestJobConfig) -> None:
        pass

    def run_iteration(self, node: IRPCNode, iter_config: TestJobConfig, stor_prefix: str) -> JobMetrics:
        # TODO: have to store logs
        cmd = self.join_remote(self.run_script)
        cmd += ' ' + self.config.params.get('run_opts', '')
        return self.parse_results(node.run(cmd, timeout=self.run_tout))

    @abc.abstractmethod
    def parse_results(self, data: str) -> JobMetrics:
        pass

