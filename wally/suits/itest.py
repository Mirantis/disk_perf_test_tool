import abc
import time
import logging
import os.path
import functools
from typing import Dict, Any, List, Tuple

from concurrent.futures import ThreadPoolExecutor

from ..utils import Barrier, StopTestError
from ..statistic import data_property
from ..inode import INode
from ..storage import Storage


logger = logging.getLogger("wally")


class TestConfig:
    """
    this class describe test input configuration

    test_type:str - test type name
    params:{str:Any} - parameters from yaml file for this test
    test_uuid:str - UUID to be used to create filenames and Co
    log_directory:str - local directory to store results
    nodes:[Node] - node to run tests on
    remote_dir:str - directory on nodes to be used for local files
    """
    def __init__(self,
                 test_type: str,
                 params: Dict[str, Any],
                 run_uuid: str,
                 nodes: List[INode],
                 storage: Storage,
                 remote_dir: str):
        self.test_type = test_type
        self.params = params
        self.run_uuid = run_uuid
        self.nodes = nodes
        self.storage = storage
        self.remote_dir = remote_dir


class TestResults:
    """
    this class describe test results

    config:TestConfig - test config object
    params:dict - parameters from yaml file for this test
    results:{str:MeasurementMesh} - test results object
    raw_result:Any - opaque object to store raw results
    run_interval:(float, float) - test tun time, used for sensors
    """
    def __init__(self,
                 config: TestConfig,
                 results: Dict[str, Any],
                 raw_result: Any,
                 run_interval: Tuple[float, float]):
        self.config = config
        self.params = config.params
        self.results = results
        self.raw_result = raw_result
        self.run_interval = run_interval

    def __str__(self):
        res = "{0}({1}):\n    results:\n".format(
                    self.__class__.__name__,
                    self.summary())

        for name, val in self.results.items():
            res += "        {0}={1}\n".format(name, val)

        res += "    params:\n"

        for name, val in self.params.items():
            res += "        {0}={1}\n".format(name, val)

        return res

    @abc.abstractmethod
    def summary(self):
        pass

    @abc.abstractmethod
    def get_yamable(self):
        pass


# class MeasurementMatrix:
#     """
#     data:[[MeasurementResult]] - VM_COUNT x TH_COUNT matrix of MeasurementResult
#     """
#     def __init__(self, data, connections_ids):
#         self.data = data
#         self.connections_ids = connections_ids
#
#     def per_vm(self):
#         return self.data
#
#     def per_th(self):
#         return sum(self.data, [])


class MeasurementResults:
    def stat(self):
        return data_property(self.data)

    def __str__(self):
        return 'TS([' + ", ".join(map(str, self.data)) + '])'


class SimpleVals(MeasurementResults):
    """
    data:[float] - list of values
    """
    def __init__(self, data):
        self.data = data


class TimeSeriesValue(MeasurementResults):
    """
    data:[(float, float, float)] - list of (start_time, lenght, average_value_for_interval)
    odata: original values
    """
    def __init__(self, data: List[Tuple[float, float, float]]):
        assert len(data) > 0
        self.odata = data[:]
        self.data = []

        cstart = 0
        for nstart, nval in data:
            self.data.append((cstart, nstart - cstart, nval))
            cstart = nstart

    @property
    def values(self) -> List[float]:
        return [val[2] for val in self.data]

    def average_interval(self) -> float:
        return float(sum([val[1] for val in self.data])) / len(self.data)

    def skip(self, seconds) -> 'TimeSeriesValue':
        nres = []
        for start, ln, val in self.data:
            nstart = start + ln - seconds
            if nstart > 0:
                nres.append([nstart, val])
        return self.__class__(nres)

    def derived(self, tdelta) -> 'TimeSeriesValue':
        end = self.data[-1][0] + self.data[-1][1]
        tdelta = float(tdelta)

        ln = end / tdelta

        if ln - int(ln) > 0:
            ln += 1

        res = [[tdelta * i, 0.0] for i in range(int(ln))]

        for start, lenght, val in self.data:
            start_idx = int(start / tdelta)
            end_idx = int((start + lenght) / tdelta)

            for idx in range(start_idx, end_idx + 1):
                rstart = tdelta * idx
                rend = tdelta * (idx + 1)

                intersection_ln = min(rend, start + lenght) - max(start, rstart)
                if intersection_ln > 0:
                    try:
                        res[idx][1] += val * intersection_ln / tdelta
                    except IndexError:
                        raise

        return self.__class__(res)


class PerfTest:
    """
    Very base class for tests
    config:TestConfig - test configuration
    stop_requested:bool - stop for test requested
    """
    def __init__(self, config):
        self.config = config
        self.stop_requested = False

    def request_stop(self) -> None:
        self.stop_requested = True

    def join_remote(self, path: str) -> str:
        return os.path.join(self.config.remote_dir, path)

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def format_for_console(cls, data: Any) -> str:
        pass


class ThreadedTest(PerfTest):
    """
    Base class for tests, which spawn separated thread for each node
    """

    def run(self) -> List[TestResults]:
        barrier = Barrier(len(self.config.nodes))
        th_test_func = functools.partial(self.th_test_func, barrier)

        with ThreadPoolExecutor(len(self.config.nodes)) as pool:
            return list(pool.map(th_test_func, self.config.nodes))

    @abc.abstractmethod
    def do_test(self, node: INode) -> TestResults:
        pass

    def th_test_func(self, barrier: Barrier, node: INode) -> TestResults:
        test_name = self.__class__.__name__
        logger.debug("Starting {} test on {}".format(test_name , node))
        logger.debug("Run test preparation on {}".format(node))
        self.pre_run(node)

        # wait till all thread became ready
        barrier.wait()

        logger.debug("Run test on {}".format(node))
        try:
            return self.do_test(node)
        except Exception as exc:
            msg = "In test {} for {}".format(test_name, node)
            logger.exception(msg)
            raise StopTestError(msg) from exc

    def pre_run(self, node: INode) -> None:
        pass


class TwoScriptTest(ThreadedTest):
    def __init__(self, *dt, **mp):
        ThreadedTest.__init__(self, *dt, **mp)
        self.remote_dir = '/tmp'
        self.prerun_script = self.config.params['prerun_script']
        self.run_script = self.config.params['run_script']

        self.prerun_tout = self.config.params.get('prerun_tout', 3600)
        self.run_tout = self.config.params.get('run_tout', 3600)

    def get_remote_for_script(self, script: str) -> str:
        return os.path.join(self.remote_dir, os.path.basename(script))

    def pre_run(self, node: INode) -> None:
        copy_paths(node.connection,
                   {self.run_script: self.get_remote_for_script(self.run_script),
                    self.prerun_script: self.get_remote_for_script(self.prerun_script)})

        cmd = self.get_remote_for_script(self.prerun_script)
        cmd += ' ' + self.config.params.get('prerun_opts', '')
        node.run(cmd, timeout=self.prerun_tout)

    def do_test(self, node: INode) -> TestResults:
        cmd = self.get_remote_for_script(self.run_script)
        cmd += ' ' + self.config.params.get('run_opts', '')
        t1 = time.time()
        res = node.run(cmd, timeout=self.run_tout)
        t2 = time.time()
        return TestResults(self.config, None, res, (t1, t2))
