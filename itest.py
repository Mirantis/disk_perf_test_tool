import abc
import json
import os.path
import logging
from StringIO import StringIO
from ConfigParser import RawConfigParser

from tests import io
from ssh_utils import copy_paths
from utils import run_over_ssh, ssize_to_b


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
    def __init__(self, opts, testtool, on_result_cb, keep_tmp_files):
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
        cmd = remote_script + ' ' + ' '.join(self.opts)
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
        self.pre_run_script = "hl_tests/postgres/prepare.sh"

    def set_pre_run_script(self):
        self.run_script = "hl_tests/postgres/run.sh"


class IOPerfTest(IPerfTest):
    io_py_remote = "/tmp/io.py"

    def __init__(self,
                 test_options,
                 on_result_cb):
        IPerfTest.__init__(self, on_result_cb)
        self.options = test_options
        self.config_fname = test_options['config_file']
        self.tool = test_options['tool']
        self.configs = []

        cp = RawConfigParser()
        cp.readfp(open(self.config_fname))

        for secname in cp.sections():
            params = dict(cp.items(secname))
            self.configs.append((secname, params))

    def pre_run(self, conn):
        local_fname = io.__file__.rsplit('.')[0] + ".py"
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
        try:
            for secname, _params in self.configs:
                params = _params.copy()
                count = params.pop('count', 1)

                config = RawConfigParser()
                config.add_section(secname)

                for k, v in params.items():
                    config.set(secname, k, v)

                cfg = StringIO()
                config.write(cfg)

                # FIX python config parser-fio incompatibility
                # remove spaces around '='
                new_cfg = []
                config_data = cfg.getvalue()
                for line in config_data.split("\n"):
                    if '=' in line:
                        name, val = line.split('=', 1)
                        name = name.strip()
                        val = val.strip()
                        line = "{0}={1}".format(name, val)
                    new_cfg.append(line)

                for _ in range(count):
                    barrier.wait()
                    code, out_err = run_over_ssh(conn, cmd,
                                                 stdin_data="\n".join(new_cfg))
                    self.on_result(code, out_err, cmd)
        finally:
            barrier.exit()

    def on_result(self, code, out_err, cmd):
        if 0 == code:
            try:
                for line in out_err.split("\n"):
                    if line.strip() != "":
                        self.on_result_cb(json.loads(line))
            except Exception as exc:
                msg_templ = "Error during postprocessing results: {0!r}"
                raise RuntimeError(msg_templ.format(exc.message))
        else:
            templ = "Command {0!r} failed with code {1}. Output is:\n{2}"
            logger.error(templ.format(cmd, code, out_err))
