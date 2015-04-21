import abc
import time
import os.path
import logging
import datetime

from wally.ssh_utils import copy_paths, run_over_ssh, delete_file
from wally.utils import ssize_to_b, open_for_append_or_create, sec_to_str

from . import postgres
from .io import agent as io_agent
from .io import formatter as io_formatter
from .io.results_loader import parse_output


logger = logging.getLogger("wally")


class IPerfTest(object):
    def __init__(self, on_result_cb, log_directory=None, node=None):
        self.on_result_cb = on_result_cb
        self.log_directory = log_directory
        self.node = node

    def pre_run(self, conn):
        pass

    def cleanup(self, conn):
        pass

    @abc.abstractmethod
    def run(self, conn, barrier):
        pass

    @classmethod
    def format_for_console(cls, data):
        msg = "{0}.format_for_console".format(cls.__name__)
        raise NotImplementedError(msg)


class TwoScriptTest(IPerfTest):
    remote_tmp_dir = '/tmp'

    def __init__(self, opts, on_result_cb, log_directory=None, node=None):
        IPerfTest.__init__(self, on_result_cb, log_directory, node=node)
        self.opts = opts

        if 'run_script' in self.opts:
            self.run_script = self.opts['run_script']
            self.prepare_script = self.opts['prepare_script']

    def get_remote_for_script(self, script):
        return os.path.join(self.tmp_dir, script.rpartition('/')[2])

    def copy_script(self, conn, src):
        remote_path = self.get_remote_for_script(src)
        copy_paths(conn, {src: remote_path})
        return remote_path

    def pre_run(self, conn):
        remote_script = self.copy_script(conn, self.pre_run_script)
        cmd = remote_script
        run_over_ssh(conn, cmd, node=self.node)

    def run(self, conn, barrier):
        remote_script = self.copy_script(conn, self.run_script)
        cmd_opts = ' '.join(["%s %s" % (key, val) for key, val
                             in self.opts.items()])
        cmd = remote_script + ' ' + cmd_opts
        out_err = run_over_ssh(conn, cmd, node=self.node)
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
    root = os.path.dirname(postgres.__file__)
    prepare_script = os.path.join(root, "prepare.sh")
    run_script = os.path.join(root, "run.sh")


class IOPerfTest(IPerfTest):
    io_py_remote = "/tmp/disk_test_agent.py"

    def __init__(self, test_options, on_result_cb,
                 log_directory=None, node=None):
        IPerfTest.__init__(self, on_result_cb, log_directory, node=node)
        self.options = test_options
        self.config_fname = test_options['cfg']
        self.alive_check_interval = test_options.get('alive_check_interval')
        self.config_params = test_options.get('params', {})
        self.tool = test_options.get('tool', 'fio')
        self.raw_cfg = open(self.config_fname).read()
        self.configs = list(io_agent.parse_all_in_1(self.raw_cfg,
                                                    self.config_params))

        cmd_log = os.path.join(self.log_directory, "task_compiled.cfg")
        raw_res = os.path.join(self.log_directory, "raw_results.txt")

        fio_command_file = open_for_append_or_create(cmd_log)

        cfg_s_it = io_agent.compile_all_in_1(self.raw_cfg, self.config_params)
        splitter = "\n\n" + "-" * 60 + "\n\n"
        fio_command_file.write(splitter.join(cfg_s_it))
        self.fio_raw_results_file = open_for_append_or_create(raw_res)

    def cleanup(self, conn):
        delete_file(conn, self.io_py_remote)
        # Need to remove tempo files, used for testing

    def pre_run(self, conn):
        try:
            run_over_ssh(conn, 'which fio', node=self.node)
        except OSError:
            # TODO: install fio, if not installed
            cmd = "sudo apt-get -y install fio"

            for i in range(3):
                try:
                    run_over_ssh(conn, cmd, node=self.node)
                    break
                except OSError as err:
                    time.sleep(3)
            else:
                raise OSError("Can't install fio - " + err.message)

        local_fname = io_agent.__file__.rsplit('.')[0] + ".py"
        self.files_to_copy = {local_fname: self.io_py_remote}
        copy_paths(conn, self.files_to_copy)

        if self.options.get('prefill_files', True):
            files = {}

            for section in self.configs:
                sz = ssize_to_b(section.vals['size'])
                msz = sz / (1024 ** 2)

                if sz % (1024 ** 2) != 0:
                    msz += 1

                fname = section.vals['filename']

                # if already has other test with the same file name
                # take largest size
                files[fname] = max(files.get(fname, 0), msz)

            # logger.warning("dd run DISABLED")
            cmd_templ = "dd if=/dev/zero of={0} bs={1} count={2}"

            # cmd_templ = "sudo dd if=/dev/zero of={0} bs={1} count={2}"
            ssize = 0
            stime = time.time()

            for fname, curr_sz in files.items():
                cmd = cmd_templ.format(fname, 1024 ** 2, curr_sz)
                ssize += curr_sz
                run_over_ssh(conn, cmd, timeout=curr_sz, node=self.node)

            ddtime = time.time() - stime
            if ddtime > 1E-3:
                fill_bw = int(ssize / ddtime)
                mess = "Initiall dd fill bw is {0} MiBps for this vm"
                logger.info(mess.format(fill_bw))
        else:
            logger.warning("Test files prefill disabled")

    def run(self, conn, barrier):
        # logger.warning("No tests runned")
        # return
        # cmd_templ = "sudo env python2 {0} --type {1} {2} --json -"
        cmd_templ = "env python2 {0} --type {1} {2} --json -"

        params = " ".join("{0}={1}".format(k, v)
                          for k, v in self.config_params.items())

        if "" != params:
            params = "--params " + params

        cmd = cmd_templ.format(self.io_py_remote, self.tool, params)
        logger.debug("Waiting on barrier")

        exec_time = io_agent.calculate_execution_time(self.configs)
        exec_time_str = sec_to_str(exec_time)

        try:
            timeout = int(exec_time * 1.2 + 300)
            if barrier.wait():
                templ = "Test should takes about {0}." + \
                        " Should finish at {1}," + \
                        " will wait at most till {2}"
                now_dt = datetime.datetime.now()
                end_dt = now_dt + datetime.timedelta(0, exec_time)
                wait_till = now_dt + datetime.timedelta(0, timeout)

                logger.info(templ.format(exec_time_str,
                                         end_dt.strftime("%H:%M:%S"),
                                         wait_till.strftime("%H:%M:%S")))

            out_err = run_over_ssh(conn, cmd,
                                   stdin_data=self.raw_cfg,
                                   timeout=timeout,
                                   node=self.node)
            logger.info("Done")
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
        if len(results) == 0:
            return None

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

    @classmethod
    def format_for_console(cls, data):
        return io_formatter.format_results_for_console(data)
