import abc
import os.path
import functools


from wally.ssh_utils import run_over_ssh, copy_paths


def cached_prop(func):
    @property
    @functools.wraps(func)
    def closure(self):
        val = getattr(self, "_" + func.__name__)
        if val is NoData:
            val = func(self)
            setattr(self, "_" + func.__name__, val)
        return val
    return closure


class NoData(object):
    pass


class VMThData(object):
    "store set of values for VM_COUNT * TH_COUNT"


class IOTestResult(object):
    def __init__(self):
        self.run_config = None
        self.suite_config = None
        self.run_interval = None

        self.bw = None
        self.lat = None
        self.iops = None
        self.slat = None
        self.clat = None

        self.fio_section = None

        self._lat_log = NoData
        self._iops_log = NoData
        self._bw_log = NoData

        self._sensors_data = NoData
        self._raw_resuls = NoData

    def to_jsonable(self):
        pass

    @property
    def thread_count(self):
        pass

    @property
    def sync_mode(self):
        pass

    @property
    def abbrev_name(self):
        pass

    @property
    def full_name(self):
        pass

    @cached_prop
    def lat_log(self):
        pass

    @cached_prop
    def iops_log(self):
        pass

    @cached_prop
    def bw_log(self):
        pass

    @cached_prop
    def sensors_data(self):
        pass

    @cached_prop
    def raw_resuls(self):
        pass


class TestResults(object):
    def __init__(self, config, params, results,
                 raw_result, run_interval, vm_count,
                 test_name, **attrs):
        self.config = config
        self.params = params
        self.results = results
        self.raw_result = raw_result
        self.run_interval = run_interval
        self.vm_count = vm_count
        self.test_name = test_name
        self.__dict__.update(attrs)

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


class IPerfTest(object):
    def __init__(self, options, is_primary, on_result_cb, test_uuid, node,
                 total_nodes_count,
                 log_directory=None,
                 coordination_queue=None,
                 remote_dir="/tmp/wally"):
        self.options = options
        self.on_result_cb = on_result_cb
        self.log_directory = log_directory
        self.node = node
        self.test_uuid = test_uuid
        self.coordination_queue = coordination_queue
        self.remote_dir = remote_dir
        self.is_primary = is_primary
        self.stop_requested = False
        self.total_nodes_count = total_nodes_count

    def request_stop(self):
        self.stop_requested = True

    def join_remote(self, path):
        return os.path.join(self.remote_dir, path)

    def coordinate(self, data):
        if self.coordination_queue is not None:
            self.coordination_queue.put((self.node.get_conn_id(), data))

    def pre_run(self):
        pass

    def cleanup(self):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, data):
        pass

    @abc.abstractmethod
    def run(self, barrier):
        pass

    @classmethod
    def format_for_console(cls, data):
        msg = "{0}.format_for_console".format(cls.__name__)
        raise NotImplementedError(msg)

    def run_over_ssh(self, cmd, **kwargs):
        return run_over_ssh(self.node.connection, cmd,
                            node=self.node.get_conn_id(), **kwargs)

    @classmethod
    def coordination_th(cls, coord_q, barrier, num_threads):
        pass


class TwoScriptTest(IPerfTest):
    def __init__(self, *dt, **mp):
        IPerfTest.__init__(self, *dt, **mp)

        if 'scripts_path' in self.options:
            self.root = self.options['scripts_path']
            self.run_script = self.options['run_script']
            self.prerun_script = self.options['prerun_script']

    def get_remote_for_script(self, script):
        return os.path.join(self.remote_dir, script.rpartition('/')[2])

    def pre_run(self):
        copy_paths(self.node.connection, {self.root: self.remote_dir})
        cmd = self.get_remote_for_script(self.pre_run_script)
        self.run_over_ssh(cmd, timeout=2000)

    def run(self, barrier):
        remote_script = self.get_remote_for_script(self.run_script)
        cmd_opts = ' '.join(["%s %s" % (key, val) for key, val
                             in self.options.items()])
        cmd = remote_script + ' ' + cmd_opts
        out_err = self.run_over_ssh(cmd, timeout=6000)
        self.on_result(out_err, cmd)

    def parse_results(self, out):
        for line in out.split("\n"):
            key, separator, value = line.partition(":")
            if key and value:
                self.on_result_cb((key, float(value)))

    def on_result(self, out_err, cmd):
        try:
            self.parse_results(out_err)
        except Exception as exc:
            msg_templ = "Error during postprocessing results: {0!s}. {1}"
            raise RuntimeError(msg_templ.format(exc, out_err))

    def merge_results(self, results):
        tpcm = sum([val[1] for val in results])
        return {"res": {"TpmC": tpcm}}
