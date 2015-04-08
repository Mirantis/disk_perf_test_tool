import re
import abc
import json
import os.path
import logging

from disk_perf_test_tool.tests import disk_test_agent
from disk_perf_test_tool.ssh_utils import copy_paths
from disk_perf_test_tool.utils import run_over_ssh, ssize_to_b

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
        code, out_err = run_over_ssh(conn, cmd)
        if code != 0:
            raise Exception("Pre run failed. %s" % out_err)

    def run(self, conn, barrier):
        remote_script = self.copy_script(conn, self.run_script)
        cmd_opts = ' '.join(["%s %s" % (key, val) for key, val
                             in self.opts.items()])
        cmd = remote_script + ' ' + cmd_opts
        code, out_err = run_over_ssh(conn, cmd)
        self.on_result(code, out_err, cmd)

    def parse_results(self, out):
        for line in out.split("\n"):
            key, separator, value = line.partition(":")
            if key and value:
                self.on_result_cb((key, float(value)))

    def on_result(self, code, out_err, cmd):
        if 0 == code:
            try:
                self.parse_results(out_err)
            except Exception as exc:
                msg_templ = "Error during postprocessing results: {0!r}"
                raise RuntimeError(msg_templ.format(exc.message))
        else:
            templ = "Command {0!r} failed with code {1}. Error output is:\n{2}"
            logger.error(templ.format(cmd, code, out_err))


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

        parse_func = disk_test_agent.parse_fio_config_full
        self.configs = parse_func(self.raw_cfg, self.config_params)

    def pre_run(self, conn):

        # TODO: install fio, if not installed
        run_over_ssh(conn, "apt-get -y install fio")

        local_fname = disk_test_agent.__file__.rsplit('.')[0] + ".py"
        self.files_to_copy = {local_fname: self.io_py_remote}
        copy_paths(conn, self.files_to_copy)

        cmd_templ = "dd if=/dev/zero of={0} bs={1} count={2}"
        for secname, params in self.configs:
            sz = ssize_to_b(params['size'])
            msz = msz = sz / (1024 ** 2)
            if sz % (1024 ** 2) != 0:
                msz += 1

            cmd = cmd_templ.format(params['filename'], 1024 ** 2, msz)
            code, out_err = run_over_ssh(conn, cmd)

        if code != 0:
            raise RuntimeError("Preparation failed " + out_err)

    def run(self, conn, barrier):
        cmd_templ = "env python2 {0} --type {1} --json -"
        cmd = cmd_templ.format(self.io_py_remote, self.tool)
        logger.debug("Run {0}".format(cmd))
        try:
            barrier.wait()
            code, out_err = run_over_ssh(conn, cmd, stdin_data=self.raw_cfg)
            self.on_result(code, out_err, cmd)
        finally:
            barrier.exit()

    def on_result(self, code, out_err, cmd):
        if 0 == code:
            try:
                for data in disk_test_agent.parse_output(out_err):
                    self.on_result_cb(data)
            except Exception as exc:
                msg_templ = "Error during postprocessing results: {0!r}"
                raise RuntimeError(msg_templ.format(exc.message))
        else:
            templ = "Command {0!r} failed with code {1}. Output is:\n{2}"
            logger.error(templ.format(cmd, code, out_err))
