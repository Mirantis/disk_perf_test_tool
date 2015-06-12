import abc
import time
import logging
import os.path
import functools

from concurrent.futures import ThreadPoolExecutor

from wally.utils import Barrier, StopTestError
from wally.statistic import data_property
from wally.ssh_utils import run_over_ssh, copy_paths


logger = logging.getLogger("wally")


class TestConfig(object):
    """
    this class describe test input configuration

    test_type:str - test type name
    params:{str:Any} - parameters from yaml file for this test
    test_uuid:str - UUID to be used to create filenames and Co
    log_directory:str - local directory to store results
    nodes:[Node] - node to run tests on
    remote_dir:str - directory on nodes to be used for local files
    """
    def __init__(self, test_type, params, test_uuid, nodes,
                 log_directory, remote_dir):
        self.test_type = test_type
        self.params = params
        self.test_uuid = test_uuid
        self.log_directory = log_directory
        self.nodes = nodes
        self.remote_dir = remote_dir


class TestResults(object):
    """
    this class describe test results

    config:TestConfig - test config object
    params:dict - parameters from yaml file for this test
    results:{str:MeasurementMesh} - test results object
    raw_result:Any - opaque object to store raw results
    run_interval:(float, float) - test tun time, used for sensors
    """
    def __init__(self, config, results, raw_result, run_interval):
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


class MeasurementMatrix(object):
    """
    data:[[MeasurementResult]] - VM_COUNT x TH_COUNT matrix of MeasurementResult
    """
    def __init__(self, data):
        self.data = data

    def per_vm(self):
        return self.data

    def per_th(self):
        return sum(self.data, [])


class MeasurementResults(object):
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
    values:[(float, float, float)] - list of (start_time, lenght, average_value_for_interval)
    """
    def __init__(self, data):
        assert len(data) > 0
        data = [(0, 0)] + data

        self.values = []
        for (cstart, cval), (nstart, nval) in zip(data[:-1], data[1:]):
            self.values.append((cstart, nstart - cstart, nval))

    @property
    def values(self):
        return [val[2] for val in self.data]

    def skip(self, seconds):
        nres = []
        for start, ln, val in enumerate(self.data):
            if start + ln < seconds:
                continue
            elif start > seconds:
                nres.append([start + ln - seconds, val])
            else:
                nres.append([0, val])
        return self.__class__(nres)

    def derived(self, tdelta):
        end = tdelta
        res = [[end, 0.0]]
        tdelta = float(tdelta)

        for start, lenght, val in self.data:
            if start < end:
                ln = min(end, start + lenght) - start
                res[-1][1] += val * ln / tdelta

            if end <= start + lenght:
                end += tdelta
                res.append([end, 0.0])
                while end < start + lenght:
                    res[-1][1] = val
                    res.append([end, 0.0])
                    end += tdelta

        if res[-1][1] < 1:
            res = res[:-1]

        return self.__class__(res)


class PerfTest(object):
    """
    Very base class for tests
    config:TestConfig - test configuration
    stop_requested:bool - stop for test requested
    """
    def __init__(self, config):
        self.config = config
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def join_remote(self, path):
        return os.path.join(self.config.remote_dir, path)

    @classmethod
    @abc.abstractmethod
    def load(cls, path):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def format_for_console(cls, data):
        pass


def run_on_node(node):
    def closure(*args, **kwargs):
        return run_over_ssh(node.connection,
                            *args,
                            node=node.get_conn_id(),
                            **kwargs)
    return closure


class ThreadedTest(PerfTest):
    """
    Base class for tests, which spawn separated thread for each node
    """

    def run(self):
        barrier = Barrier(len(self.nodes))
        th_test_func = functools.partial(self.th_test_func, barrier)

        with ThreadPoolExecutor(len(self.nodes)) as pool:
            return list(pool.map(th_test_func, self.config.nodes))

    @abc.abstractmethod
    def do_test(self, node):
        pass

    def th_test_func(self, barrier, node):
        logger.debug("Starting {0} test on {1} node".format(self.__class__.__name__,
                                                            node.conn_url))

        logger.debug("Run preparation for {0}".format(node.get_conn_id()))
        self.pre_run(node)
        barrier.wait()
        try:
            logger.debug("Run test for {0}".format(node.get_conn_id()))
            return self.do_test(node)
        except StopTestError as exc:
            pass
        except Exception as exc:
            msg = "In test {0} for node {1}".format(self, node.get_conn_id())
            logger.exception(msg)
            exc = StopTestError(msg, exc)

        try:
            self.cleanup()
        except StopTestError as exc1:
            if exc is None:
                exc = exc1
        except Exception as exc1:
            if exc is None:
                msg = "Duringf cleanup - in test {0} for node {1}".format(self, node)
                logger.exception(msg)
                exc = StopTestError(msg, exc)

        if exc is not None:
            raise exc

    def pre_run(self, node):
        pass

    def cleanup(self, node):
        pass


class TwoScriptTest(ThreadedTest):
    def __init__(self, *dt, **mp):
        ThreadedTest.__init__(self, *dt, **mp)

        self.prerun_script = self.config.params['prerun_script']
        self.run_script = self.config.params['run_script']

        self.prerun_tout = self.config.params.get('prerun_tout', 3600)
        self.run_tout = self.config.params.get('run_tout', 3600)

    def get_remote_for_script(self, script):
        return os.path.join(self.options.remote_dir,
                            os.path.basename(script))

    def pre_run(self, node):
        copy_paths(node.connection,
                   {
                       self.run_script: self.get_remote_for_script(self.run_script),
                       self.prerun_script: self.get_remote_for_script(self.prerun_script),
                   })

        cmd = self.get_remote_for_script(self.pre_run_script)
        cmd += ' ' + self.config.params.get('prerun_opts', '')
        run_on_node(node)(cmd, timeout=self.prerun_tout)

    def run(self, node):
        cmd = self.get_remote_for_script(self.run_script)
        cmd += ' ' + self.config.params.get('run_opts', '')
        t1 = time.time()
        res = run_on_node(node)(cmd, timeout=self.run_tout)
        t2 = time.time()
        return TestResults(self.config, None, res, (t1, t2))
