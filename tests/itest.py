import abc
import time
import os.path
import logging

from disk_perf_test_tool.tests import disk_test_agent
from disk_perf_test_tool.tests.disk_test_agent import parse_fio_config_full
from disk_perf_test_tool.tests.disk_test_agent import estimate_cfg, sec_to_str
from disk_perf_test_tool.tests.io_results_loader import parse_output
from disk_perf_test_tool.ssh_utils import copy_paths, run_over_ssh
from disk_perf_test_tool.utils import ssize_to_b


logger = logging.getLogger("io-perf-tool")


class IPerfTest(object):
    def __init__(self, on_result_cb):
        self.on_result_cb = on_result_cb

    def pre_run(self, conn):
        pass

    @abc.abstractmethod
    def run(self, conn, barrier):
        pass


class TwoScriptTest(IPerfTest):
    def __init__(self, opts, on_result_cb):
        super(TwoScriptTest, self).__init__(on_result_cb)
        self.opts = opts
        self.pre_run_script = None
        self.run_script = None
        self.tmp_dir = "/tmp/"
        self.set_run_script()
        self.set_pre_run_script()

    def set_run_script(self):
        self.pre_run_script = self.opts.pre_run_script

    def set_pre_run_script(self):
        self.run_script = self.opts.run_script

    def get_remote_for_script(self, script):
        return os.path.join(self.tmp_dir, script.rpartition('/')[2])

    def copy_script(self, conn, src):
        remote_path = self.get_remote_for_script(src)
        copy_paths(conn, {src: remote_path})
        return remote_path

    def pre_run(self, conn):
        remote_script = self.copy_script(conn, self.pre_run_script)
        cmd = remote_script
        run_over_ssh(conn, cmd)

    def run(self, conn, barrier):
        remote_script = self.copy_script(conn, self.run_script)
        cmd_opts = ' '.join(["%s %s" % (key, val) for key, val
                             in self.opts.items()])
        cmd = remote_script + ' ' + cmd_opts
        out_err = run_over_ssh(conn, cmd)
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
            msg_templ = "Error during postprocessing results: {0!r}. {1}"
            raise RuntimeError(msg_templ.format(exc.message, out_err))


class PgBenchTest(TwoScriptTest):

    def set_run_script(self):
        self.pre_run_script = "tests/postgres/prepare.sh"

    def set_pre_run_script(self):
        self.run_script = "tests/postgres/run.sh"


class IOPerfTest(IPerfTest):
    io_py_remote = "/tmp/disk_test_agent.py"

    def __init__(self,
                 test_options,
                 on_result_cb):
        IPerfTest.__init__(self, on_result_cb)
        self.options = test_options
        self.config_fname = test_options['cfg']
        self.config_params = test_options.get('params', {})
        self.tool = test_options.get('tool', 'fio')
        self.raw_cfg = open(self.config_fname).read()
        self.configs = list(parse_fio_config_full(self.raw_cfg,
                                                  self.config_params))

    def pre_run(self, conn):

        try:
            run_over_ssh(conn, 'which fio')
        except OSError:
            # TODO: install fio, if not installed
            cmd = "sudo apt-get -y install fio"

            for i in range(3):
                try:
                    run_over_ssh(conn, cmd)
                    break
                except OSError as err:
                    time.sleep(3)
            else:
                raise OSError("Can't install fio - " + err.message)

        local_fname = disk_test_agent.__file__.rsplit('.')[0] + ".py"
        self.files_to_copy = {local_fname: self.io_py_remote}
        copy_paths(conn, self.files_to_copy)

        cmd_templ = "sudo dd if=/dev/zero of={0} bs={1} count={2}"
        files = {}

        for secname, params in self.configs:
            sz = ssize_to_b(params['size'])
            msz = msz = sz / (1024 ** 2)
            if sz % (1024 ** 2) != 0:
                msz += 1

            fname = params['filename']
            files[fname] = max(files.get(fname, 0), msz)

        for fname, sz in files.items():
            cmd = cmd_templ.format(fname, 1024 ** 2, msz)
            run_over_ssh(conn, cmd, timeout=msz)

    def run(self, conn, barrier):
        cmd_templ = "sudo env python2 {0} --type {1} {2} --json -"

        params = " ".join("{0}={1}".format(k, v)
                          for k, v in self.config_params.items())

        if "" != params:
            params = "--params " + params

        cmd = cmd_templ.format(self.io_py_remote, self.tool, params)
        logger.debug("Waiting on barrier")

        exec_time = estimate_cfg(self.raw_cfg, self.config_params)
        exec_time_str = sec_to_str(exec_time)

        try:
            if barrier.wait():
                logger.info("Test will takes about {0}".format(exec_time_str))

            out_err = run_over_ssh(conn, cmd,
                                   stdin_data=self.raw_cfg,
                                   timeout=int(exec_time * 1.1 + 300))
        finally:
            barrier.exit()

        self.on_result(out_err, cmd)

    def on_result(self, out_err, cmd):
        try:
            for data in parse_output(out_err):
                self.on_result_cb(data)
        except Exception as exc:
            msg_templ = "Error during postprocessing results: {0!r}"
            raise RuntimeError(msg_templ.format(exc.message))

    def merge_results(self, results):
        merged_result = results[0]
        merged_data = merged_result['res']
        expected_keys = set(merged_data.keys())
        mergable_fields = ['bw', 'clat', 'iops', 'lat', 'slat']

        for res in results[1:]:
            assert res['__meta__'] == merged_result['__meta__']

            data = res['res']
            diff = set(data.keys()).symmetric_difference(expected_keys)

            msg = "Difference: {0}".format(",".join(diff))
            assert len(diff) == 0, msg

            for testname, test_data in data.items():
                res_test_data = merged_data[testname]

                diff = set(test_data.keys()).symmetric_difference(
                            res_test_data.keys())

                msg = "Difference: {0}".format(",".join(diff))
                assert len(diff) == 0, msg

                for k, v in test_data.items():
                    if k in mergable_fields:
                        res_test_data[k].extend(v)
                    else:
                        msg = "{0!r} != {1!r}".format(res_test_data[k], v)
                        assert res_test_data[k] == v, msg

        return merged_result
