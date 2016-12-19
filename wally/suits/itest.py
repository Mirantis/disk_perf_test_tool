import abc
import time
import logging
import os.path
import datetime
from typing import Dict, Any, List, Optional, Tuple, cast

from concurrent.futures import ThreadPoolExecutor

from ..utils import Barrier, StopTestError, sec_to_str
from ..node_interfaces import IRPCNode
from ..storage import Storage
from ..result_classes import RawTestResults

import agent


logger = logging.getLogger("wally")


__doc__ = "Contains base classes for performance tests"


class TestInputConfig:
    """
    this class describe test input configuration

    test_type - test type name
    params - parameters from yaml file for this test
    test_uuid - UUID to be used to create file names & Co
    log_directory - local directory to store results
    nodes - nodes to run tests on
    remote_dir - directory on nodes to be used for local files
    """
    def __init__(self,
                 test_type: str,
                 params: Dict[str, Any],
                 run_uuid: str,
                 nodes: List[IRPCNode],
                 storage: Storage,
                 remote_dir: str) -> None:
        self.test_type = test_type
        self.params = params
        self.run_uuid = run_uuid
        self.nodes = nodes
        self.storage = storage
        self.remote_dir = remote_dir


class IterationConfig:
    name = None  # type: str


class PerfTest:
    """Base class for all tests"""
    name = None  # type: str
    max_retry = 3
    retry_time = 30

    def __init__(self, config: TestInputConfig) -> None:
        self.config = config
        self.stop_requested = False
        self.nodes = self.config.nodes  # type: List[IRPCNode]
        self.sorted_nodes_ids = sorted(node.info.node_id() for node in self.nodes)

    def request_stop(self) -> None:
        self.stop_requested = True

    def join_remote(self, path: str) -> str:
        return os.path.join(self.config.remote_dir, path)

    @abc.abstractmethod
    def run(self, storage: Storage) -> None:
        pass

    @abc.abstractmethod
    def format_for_console(self, data: Any) -> str:
        pass


RunTestRes = Tuple[RawTestResults, Tuple[int, int]]


class ThreadedTest(PerfTest, metaclass=abc.ABCMeta):
    """Base class for tests, which spawn separated thread for each node"""

    # max allowed time difference between starts and stops of run of the same test on different test nodes
    # used_max_diff = max((min_run_time * max_rel_time_diff), max_time_diff)
    max_time_diff = 5
    max_rel_time_diff = 0.05

    def __init__(self, config: TestInputConfig) -> None:
        PerfTest.__init__(self, config)
        self.iterations_configs = [None]  # type: List[Optional[IterationConfig]]

    @abc.abstractmethod
    def get_expected_runtime(self, iter_cfg: IterationConfig) -> Optional[int]:
        pass

    def get_not_done_stages(self, storage: Storage) -> Dict[int, IterationConfig]:
        start_run_id = max(int(name) for _, name in storage.list('result')) + 1
        not_in_storage = {}  # type: Dict[int, IterationConfig]
        for run_id, iteration_config in enumerate(self.iterations_configs, start_run_id):
            info_path = "result/{}/info".format(run_id)
            if info_path in storage:
                info = cast(Dict[str, Any], storage[info_path]) # type: Dict[str, Any]

                assert isinstance(info, dict), \
                    "Broken storage at path {}. Expect test info dict, obtain {!r}".format(info_path, info)

                info = info.copy()
                del info['begin_time']
                del info['end_time']

                iter_name = "Unnamed" if iteration_config is None else iteration_config.name
                expected_config = {
                    'name': self.name,
                    'iteration_name': iter_name,
                    'iteration_config': iteration_config,
                    'params': self.config.params,
                    'nodes': self.sorted_nodes_ids
                }

                assert info == expected_config, \
                    ("Test info at path {} is not equal to expected config." +
                     "Maybe configuration was changed before test was restarted. " +
                     "Current cfg is {!r}, expected cfg is {!r}").format(info_path, info, expected_config)

                logger.info("Test iteration {} found in storage and will be skipped".format(iter_name))
            else:
                not_in_storage[run_id] = iteration_config
        return not_in_storage

    def run(self, storage: Storage) -> None:
        not_in_storage = self.get_not_done_stages(storage)

        if not not_in_storage:
            logger.info("All test iteration in storage already. Skip test")
            return

        logger.debug("Run test {} on nodes {}.".format(self.name, ",".join(self.sorted_nodes_ids)))

        barrier = Barrier(len(self.nodes))

        logger.debug("Run preparation")

        with ThreadPoolExecutor(len(self.nodes)) as pool:
            list(pool.map(self.config_node, self.nodes))

            # +5% - is a rough estimation for additional operations
            run_times = [self.get_expected_runtime(iteration_config) for iteration_config in not_in_storage.values()]
            if None not in run_times:
                expected_run_time = int(sum(run_times) * 1.05)
                exec_time_s = sec_to_str(expected_run_time)
                now_dt = datetime.datetime.now()
                end_dt = now_dt + datetime.timedelta(0, expected_run_time)
                logger.info("Entire test should takes aroud: {} and finished at {:%H:%M:%S}"
                            .format(exec_time_s, end_dt))

            for run_id, iteration_config in sorted(not_in_storage.items()):
                iter_name = "Unnamed" if iteration_config is None else iteration_config.name
                logger.info("Run test iteration {} ".format(iter_name))

                results = []  # type: List[RunTestRes]
                for idx in range(self.max_retry):
                    barrier.wait()
                    try:
                        futures = [pool.submit(self.do_test, node, iteration_config) for node in self.nodes]
                        results = [fut.result() for fut in futures]
                    except EnvironmentError as exc:
                        if self.max_retry - 1 == idx:
                            raise StopTestError("Fio failed") from exc
                        logger.exception("During fio run")
                    else:
                        if all(results):
                            break

                    logger.info("Sleeping %ss and retrying", self.retry_time)
                    time.sleep(self.retry_time)

                start_times = []  # type: List[int]
                stop_times = []  # type: List[int]

                mstorage = storage.sub_storage("result", str(run_id), "measurement")
                for (result, (t_start, t_stop)), node in zip(results, self.config.nodes):
                    for metrics_name, data in result.items():
                        mstorage[node.info.node_id(), metrics_name] = data  # type: ignore
                    start_times.append(t_start)
                    stop_times.append(t_stop)

                min_start_time = min(start_times)
                max_start_time = max(start_times)
                min_stop_time = min(stop_times)
                max_stop_time = max(stop_times)

                max_allowed_time_diff = int((min_stop_time - max_start_time) * self.max_rel_time_diff)
                max_allowed_time_diff = max(max_allowed_time_diff, self.max_time_diff)

                if min_start_time + self.max_time_diff < max_allowed_time_diff:
                    logger.warning("Too large difference in {}:{} start time - {}. Max recommended difference is {}"
                                   .format(self.name, iter_name, max_start_time - min_start_time, self.max_time_diff))

                if min_stop_time + self.max_time_diff < max_allowed_time_diff:
                    logger.warning("Too large difference in {}:{} stop time - {}. Max recommended difference is {}"
                                   .format(self.name, iter_name, max_start_time - min_start_time, self.max_time_diff))

                test_config = {
                    'name': self.name,
                    'iteration_name': iter_name,
                    'iteration_config': iteration_config,
                    'params': self.config.params,
                    'nodes': self.sorted_nodes_ids,
                    'begin_time': min_start_time,
                    'end_time': max_stop_time
                }

                storage["result", str(run_id), "info"] = test_config  # type: ignore

    @abc.abstractmethod
    def config_node(self, node: IRPCNode) -> None:
        pass

    @abc.abstractmethod
    def do_test(self, node: IRPCNode, iter_config: IterationConfig) -> RunTestRes:
        pass


class TwoScriptTest(ThreadedTest, metaclass=abc.ABCMeta):
    def __init__(self, *dt, **mp) -> None:
        ThreadedTest.__init__(self, *dt, **mp)
        self.prerun_script = self.config.params['prerun_script']
        self.run_script = self.config.params['run_script']
        self.prerun_tout = self.config.params.get('prerun_tout', 3600)
        self.run_tout = self.config.params.get('run_tout', 3600)
        self.iterations_configs = [None]

    def get_expected_runtime(self, iter_cfg: IterationConfig) -> Optional[int]:
        return None

    def config_node(self, node: IRPCNode) -> None:
        node.copy_file(self.run_script, self.join_remote(self.run_script))
        node.copy_file(self.prerun_script, self.join_remote(self.prerun_script))

        cmd = self.join_remote(self.prerun_script)
        cmd += ' ' + self.config.params.get('prerun_opts', '')
        node.run(cmd, timeout=self.prerun_tout)

    def do_test(self, node: IRPCNode, iter_config: IterationConfig) -> RunTestRes:
        cmd = self.join_remote(self.run_script)
        cmd += ' ' + self.config.params.get('run_opts', '')
        t1 = time.time()
        res = self.parse_results(node.run(cmd, timeout=self.run_tout))
        t2 = time.time()
        return res, (int(t1), int(t2))

    @abc.abstractmethod
    def parse_results(self, data: str) -> RawTestResults:
        pass

