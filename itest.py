import abc
import json
import types
import os.path

from io_scenario import io
from ssh_copy_directory import copy_paths


class IPerfTest(object):
    def __init__(self, on_result_cb):
        self.on_result_cb = on_result_cb

    def build(self, conn):
        self.pre_run(conn)

    def pre_run(self, conn):
        pass

    @abc.abstractmethod
    def run(self, conn):
        pass


def run_test_iter(obj, conn):
    yield obj.pre_run(conn)
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
        self.files_to_copy = {testtool_local: dst_testtool_path}
        self.script_opts = script_opts + ["--binary-path", dst_testtool_path]

    def pre_run(self, conn):
        copy_paths(self.files_to_copy)

    def run(self, conn):
        io_py = os.path.dirname(io.__file__)

        if io_py.endswith('.pyc'):
            io_py = io_py[:-1]

        args = ['env', 'python2', io_py] + self.script_opts
        code, out, err = conn.execute(args)
        self.on_result(code, out, err)
        return code, out, err

    def on_result(self, code, out, err):
        if 0 == code:
            try:
                for line in out.split("\n"):
                    if line.strip() != "":
                        self.on_result_cb(json.loads(line))
            except Exception as err:
                msg = "Error during postprocessing results: {0!r}".format(err)
                raise RuntimeError(msg)
