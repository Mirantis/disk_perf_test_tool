import abc
import time
import logging
import os.path
from typing import Any, List, Optional, Callable, Iterable, cast, Tuple

from concurrent.futures import ThreadPoolExecutor, wait

from cephlib.node import IRPCNode

from ..utils import StopTestError, get_time_interval_printable_info
from ..result_classes import SuiteConfig, JobConfig, TimeSeries, IWallyStorage


logger = logging.getLogger("wally")


__doc__ = "Contains base classes for performance tests"


class PerfTest(metaclass=abc.ABCMeta):
    """Base class for all tests"""
    name = None  # type: str
    max_retry = 3
    retry_time = 30
    job_config_cls = None  # type: type

    def __init__(self, storage: IWallyStorage, suite: SuiteConfig, on_idle: Callable[[], None] = None) -> None:
        self.suite = suite
        self.stop_requested = False
        self.sorted_nodes_ids = sorted(node.node_id for node in self.suite.nodes)
        self.on_idle = on_idle
        self.storage = storage

    def request_stop(self) -> None:
        self.stop_requested = True

    def join_remote(self, path: str) -> str:
        return os.path.join(self.suite.remote_dir, path)

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
        self.job_configs = None  # type: List[JobConfig]

    @abc.abstractmethod
    def get_expected_runtime(self, iter_cfg: JobConfig) -> Optional[int]:
        pass

    def get_not_done_jobs(self) -> Iterable[JobConfig]:
        jobs_map = {job.storage_id: job for job in self.job_configs}
        already_in_storage = set()
        for db_config in cast(List[JobConfig], self.storage.iter_job(self.suite)):
            if db_config.storage_id in jobs_map:
                job = jobs_map[db_config.storage_id]
                if job != db_config:
                    logger.error("Test info at '%s.%s' is not equal to expected config for iteration %s.%s." +
                                 " Maybe configuration was changed before test was restarted. " +
                                 "DB cfg is:\n    %s\nExpected cfg is:\n    %s\nFix DB or rerun test from beginning",
                                 self.suite.storage_id, job.storage_id, self.name, job.summary,
                                 str(db_config).replace("\n", "\n    "),
                                 str(job).replace("\n", "\n    "))
                    raise StopTestError()

                logger.info("Test iteration %s.%s found in storage and will be skipped", self.name, job.summary)
                already_in_storage.add(db_config.storage_id)

        return [job for job in self.job_configs if job.storage_id not in already_in_storage]

    def run(self) -> None:
        self.storage.put_or_check_suite(self.suite)

        not_in_storage = list(self.get_not_done_jobs())
        if not not_in_storage:
            logger.info("All test iteration in storage already. Skip test")
            return

        logger.debug("Run test %s with profile %r on nodes %s.", self.name,
                                                                 self.load_profile_name,
                                                                 ",".join(self.sorted_nodes_ids))
        logger.debug("Prepare nodes")


        with ThreadPoolExecutor(len(self.suite.nodes)) as pool:
            # config nodes
            list(pool.map(self.config_node, self.suite.nodes))

            run_times = list(map(self.get_expected_runtime, not_in_storage))

            if None not in run_times:
                # +5% - is a rough estimation for additional operations
                expected_run_time = int(sum(run_times) * 1.05)

                exec_time_s, end_dt_s = get_time_interval_printable_info(expected_run_time)
                logger.info("Entire test should takes around %s and finish at %s", exec_time_s, end_dt_s)

            for job in not_in_storage:
                results = []  # type: List[TimeSeries]
                for retry_idx in range(self.max_retry):
                    logger.info("Preparing job %s", job.params.summary)

                    # prepare nodes for new iterations
                    wait([pool.submit(self.prepare_iteration, node, job) for node in self.suite.nodes])

                    expected_job_time = self.get_expected_runtime(job)
                    exec_time_s, end_dt_s = get_time_interval_printable_info(expected_job_time)
                    logger.info("Job should takes around %s and finish at %s", exec_time_s, end_dt_s)

                    jfutures = [pool.submit(self.run_iteration, node, job) for node in self.suite.nodes]
                    failed = False
                    for future in jfutures:
                        try:
                            results.extend(future.result())
                        except EnvironmentError:
                            failed = True

                    if not failed:
                        break

                    if self.max_retry - 1 == retry_idx:
                        logger.exception("Fio failed")
                        raise StopTestError()

                    logger.exception("During fio run")
                    logger.info("Sleeping %ss and retrying job", self.retry_time)
                    time.sleep(self.retry_time)
                    results = []

                # per node jobs start and stop times
                start_times = []  # type: List[int]
                stop_times = []  # type: List[int]

                for ts in results:
                    self.storage.put_ts(ts)
                    if len(ts.times) >= 2:  # type: ignore
                        start_times.append(ts.times[0])
                        stop_times.append(ts.times[-1])

                if len(start_times) > 0:
                    min_start_time = min(start_times)
                    max_start_time = max(start_times)
                    min_stop_time = min(stop_times)

                    max_allowed_time_diff = int((min_stop_time - max_start_time) * self.max_rel_time_diff)
                    max_allowed_time_diff = max(max_allowed_time_diff, self.max_time_diff)

                    if min_start_time + self.max_time_diff < max_allowed_time_diff:
                        logger.warning("Too large difference in %s:%s start time - %s. " +
                                       "Max recommended difference is %s",
                                       self.name, job.summary,
                                       max_start_time - min_start_time, self.max_time_diff)

                    if min_stop_time + self.max_time_diff < max_allowed_time_diff:
                        logger.warning("Too large difference in %s:%s stop time - %s. " +
                                       "Max recommended difference is %s",
                                       self.name, job.summary,
                                       max_start_time - min_start_time, self.max_time_diff)

                    job.reliable_info_range = (int(max_start_time), int(min_stop_time))

                self.storage.put_job(self.suite, job)
                self.storage.sync()

                if self.on_idle is not None:
                    self.on_idle()

    @abc.abstractmethod
    def config_node(self, node: IRPCNode) -> None:
        pass

    @abc.abstractmethod
    def prepare_iteration(self, node: IRPCNode, job: JobConfig) -> None:
        pass

    @abc.abstractmethod
    def run_iteration(self, node: IRPCNode, job: JobConfig) -> List[TimeSeries]:
        pass


class TwoScriptTest(ThreadedTest, metaclass=abc.ABCMeta):
    def __init__(self, *dt, **mp) -> None:
        ThreadedTest.__init__(self, *dt, **mp)
        self.prerun_script = self.suite.params['prerun_script']
        self.run_script = self.suite.params['run_script']
        self.prerun_tout = self.suite.params.get('prerun_tout', 3600)
        self.run_tout = self.suite.params.get('run_tout', 3600)
        # TODO: fix job_configs field
        raise NotImplementedError("Fix job configs")

    def get_expected_runtime(self, job: JobConfig) -> Optional[int]:
        return None

    def config_node(self, node: IRPCNode) -> None:
        node.copy_file(self.run_script, self.join_remote(self.run_script))
        node.copy_file(self.prerun_script, self.join_remote(self.prerun_script))

        cmd = self.join_remote(self.prerun_script)
        cmd += ' ' + self.suite.params.get('prerun_opts', '')
        node.run(cmd, timeout=self.prerun_tout)

    def prepare_iteration(self, node: IRPCNode, job: JobConfig) -> None:
        pass

    def run_iteration(self, node: IRPCNode, job: JobConfig) -> List[TimeSeries]:
        # TODO: have to store logs
        cmd = self.join_remote(self.run_script)
        cmd += ' ' + self.suite.params.get('run_opts', '')
        return self.parse_results(node.run(cmd, timeout=self.run_tout))

    @abc.abstractmethod
    def parse_results(self, data: str) -> List[TimeSeries]:
        pass

