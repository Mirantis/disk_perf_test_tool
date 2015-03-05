import abc
import json
import types
import os.path
import logging

from io_scenario import io
from ssh_copy_directory import copy_paths
from utils import run_over_ssh


logger = logging.getLogger("io-perf-tool")


class IPerfTest(object):
    def __init__(self, on_result_cb):
        self.set_result_cb(on_result_cb)

    def set_result_cb(self, on_result_cb):
        self.on_result_cb = on_result_cb

    def build(self, conn):
        self.pre_run(conn)

    def pre_run(self, conn):
        pass

    @abc.abstractmethod
    def run(self, conn):
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
        code, out, err = run_over_ssh(conn, cmd)
        if code != 0:
            raise Exception("Pre run failed. %s" % err)

    def run(self, conn):
        remote_script = self.copy_script(conn, self.run_script)
        cmd = remote_script + ' ' + ' '.join(self.opts)
        code, out, err = run_over_ssh(conn, cmd)
        self.on_result(code, out, err, cmd)

    def parse_results(self, out):
        for line in out.split("\n"):
            key, separator, value = line.partition(":")
            if key and value:
                self.on_result_cb((key, float(value)))

    def on_result(self, code, out, err, cmd):
        if 0 == code:
            try:
                self.parse_results(out)
            except Exception as err:
                msg_templ = "Error during postprocessing results: {0!r}"
                raise RuntimeError(msg_templ.format(err.message))
        else:
            templ = "Command {0!r} failed with code {1}. Error output is:\n{2}"
            logger.error(templ.format(cmd, code, err))


class PgBenchTest(TwoScriptTest):

    def set_run_script(self):
        self.pre_run_script = "hl_tests/postgres/prepare.sh"

    def set_pre_run_script(self):
        self.run_script = "hl_tests/postgres/run.sh"


def run_test_iter(obj, conn):
    logger.debug("Run preparation")
    yield obj.pre_run(conn)
    logger.debug("Run test")
    res = obj.run(conn)
    if isinstance(res, types.GeneratorType):
        for vl in res:
            yield vl
    else:
        yield res


class IOPerfTest(IPerfTest):
    def __init__(self,
                 script_opts,
                 testtool_local,
                 on_result_cb,
                 keep_tmp_files):

        IPerfTest.__init__(self, on_result_cb)

        dst_testtool_path = '/tmp/io_tool'
        self.script_opts = script_opts + ["--binary-path", dst_testtool_path]
        io_py_local = os.path.join(os.path.dirname(io.__file__), "io.py")
        self.io_py_remote = "/tmp/io.py"

        self.files_to_copy = {testtool_local: dst_testtool_path,
                              io_py_local: self.io_py_remote}

    def pre_run(self, conn):
        copy_paths(conn, self.files_to_copy)

        args = ['env', 'python2', self.io_py_remote] + \
            self.script_opts + ['--prepare-only']

        code, self.prep_results, err = run_over_ssh(conn, " ".join(args))
        if code != 0:
            raise RuntimeError("Preparation failed " + err)

    def run(self, conn):
        args = ['env', 'python2', self.io_py_remote] + self.script_opts
        args.append('--preparation-results')
        args.append("'{0}'".format(self.prep_results))
        cmd = " ".join(args)
        code, out, err = run_over_ssh(conn, cmd)
        self.on_result(code, out, err, cmd)
        args = ['env', 'python2', self.io_py_remote, '--clean',
                "'{0}'".format(self.prep_results)]
        logger.debug(" ".join(args))
        code, _, err = run_over_ssh(conn, " ".join(args))
        if 0 != code:
            logger.error("Cleaning failed: " + err)

    def on_result(self, code, out, err, cmd):
        if 0 == code:
            try:
                for line in out.split("\n"):
                    if line.strip() != "":
                        self.on_result_cb(json.loads(line))
            except Exception as err:
                msg_templ = "Error during postprocessing results: {0!r}"
                raise RuntimeError(msg_templ.format(err.message))
        else:
            templ = "Command {0!r} failed with code {1}. Error output is:\n{2}"
            logger.error(templ.format(cmd, code, err))
