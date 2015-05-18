import time
import json
import os.path
import logging
import datetime

from wally.utils import (ssize2b, open_for_append_or_create,
                         sec_to_str, StopTestError)

from wally.ssh_utils import save_to_remote, read_from_remote, BGSSHTask

from ..itest import IPerfTest, TestResults
from .formatter import format_results_for_console
from .fio_task_parser import (execution_time, fio_cfg_compile,
                              get_test_summary, FioJobSection)


logger = logging.getLogger("wally")


class IOTestResults(TestResults):
    def summary(self):
        return get_test_summary(self.config) + "vm" + str(self.vm_count)

    def get_yamable(self):
        return {
            'type': "fio_test",
            'params': self.params,
            'config': (self.config.name, self.config.vals),
            'results': self.results,
            'raw_result': self.raw_result,
            'run_interval': self.run_interval,
            'vm_count': self.vm_count
        }

    @classmethod
    def from_yaml(cls, data):
        name, vals = data['config']
        sec = FioJobSection(name)
        sec.vals = vals

        return cls(sec, data['params'], data['results'],
                   data['raw_result'], data['run_interval'],
                   data['vm_count'])


def get_slice_parts_offset(test_slice, real_inteval):
    calc_exec_time = sum(map(execution_time, test_slice))
    coef = (real_inteval[1] - real_inteval[0]) / calc_exec_time
    curr_offset = real_inteval[0]
    for section in test_slice:
        slen = execution_time(section) * coef
        yield (curr_offset, curr_offset + slen)
        curr_offset += slen


class IOPerfTest(IPerfTest):
    tcp_conn_timeout = 30
    max_pig_timeout = 5
    soft_runcycle = 5 * 60

    def __init__(self, *dt, **mp):
        IPerfTest.__init__(self, *dt, **mp)
        self.config_fname = self.options['cfg']

        if '/' not in self.config_fname and '.' not in self.config_fname:
            cfgs_dir = os.path.dirname(__file__)
            self.config_fname = os.path.join(cfgs_dir,
                                             self.config_fname + '.cfg')

        self.alive_check_interval = self.options.get('alive_check_interval')

        self.config_params = self.options.get('params', {}).copy()
        self.tool = self.options.get('tool', 'fio')

        raw_res = os.path.join(self.log_directory, "raw_results.txt")
        self.fio_raw_results_file = open_for_append_or_create(raw_res)

        self.io_py_remote = self.join_remote("agent.py")
        self.results_file = self.join_remote("results.json")
        self.pid_file = self.join_remote("pid")
        self.task_file = self.join_remote("task.cfg")
        self.use_sudo = self.options.get("use_sudo", True)
        self.test_logging = self.options.get("test_logging", False)
        self.raw_cfg = open(self.config_fname).read()
        self.fio_configs = fio_cfg_compile(self.raw_cfg,
                                           self.config_fname,
                                           self.config_params,
                                           split_on_names=self.test_logging)
        self.fio_configs = list(self.fio_configs)

        cmd_log = os.path.join(self.log_directory, "task_compiled.cfg")
        fio_command_file = open_for_append_or_create(cmd_log)
        splitter = "\n\n" + "-" * 60 + "\n\n"
        fio_command_file.write(splitter.join(map(str, self.fio_configs)))

    def __str__(self):
        return "{0}({1})".format(self.__class__.__name__,
                                 self.node.get_conn_id())

    @classmethod
    def load(cls, data):
        return IOTestResults.from_yaml(data)

    def cleanup(self):
        # delete_file(conn, self.io_py_remote)
        # Need to remove tempo files, used for testing
        pass

    def prefill_test_files(self):
        files = {}
        for cfg_slice in self.fio_configs:
            for section in cfg_slice:
                sz = ssize2b(section.vals['size'])
                msz = sz / (1024 ** 2)

                if sz % (1024 ** 2) != 0:
                    msz += 1

                fname = section.vals['filename']

                # if already has other test with the same file name
                # take largest size
                files[fname] = max(files.get(fname, 0), msz)

        cmd_templ = "dd oflag=direct " + \
                    "if=/dev/zero of={0} bs={1} count={2}"

        if self.use_sudo:
            cmd_templ = "sudo " + cmd_templ

        ssize = 0
        stime = time.time()

        for fname, curr_sz in files.items():
            cmd = cmd_templ.format(fname, 1024 ** 2, curr_sz)
            ssize += curr_sz
            self.run_over_ssh(cmd, timeout=curr_sz)

        ddtime = time.time() - stime
        if ddtime > 1E-3:
            fill_bw = int(ssize / ddtime)
            mess = "Initiall dd fill bw is {0} MiBps for this vm"
            logger.info(mess.format(fill_bw))
            self.coordinate(('init_bw', fill_bw))

    def install_utils(self, max_retry=3, timeout=5):
        need_install = []
        for bin_name, package in (('fio', 'fio'), ('screen', 'screen')):
            try:
                self.run_over_ssh('which ' + bin_name, nolog=True)
            except OSError:
                need_install.append(package)

        if len(need_install) == 0:
            return

        cmd = "sudo apt-get -y install " + " ".join(need_install)

        for i in range(max_retry):
            try:
                self.run_over_ssh(cmd)
                break
            except OSError as err:
                time.sleep(timeout)
        else:
            raise OSError("Can't install - " + str(err))

    def pre_run(self):
        try:
            cmd = 'mkdir -p "{0}"'.format(self.remote_dir)
            if self.use_sudo:
                cmd = "sudo " + cmd
                cmd += " ; sudo chown {0} {1}".format(self.node.get_user(),
                                                      self.remote_dir)

            self.run_over_ssh(cmd)
        except Exception as exc:
            msg = "Failed to create folder {0} on remote {1}. Error: {2!s}"
            msg = msg.format(self.remote_dir, self.node.get_conn_id(), exc)
            logger.exception(msg)
            raise StopTestError(msg, exc)

        self.install_utils()

        if self.options.get('prefill_files', True):
            self.prefill_test_files()
        elif self.is_primary:
            logger.warning("Prefilling of test files is disabled")

    def run(self, barrier):
        try:
            if len(self.fio_configs) > 1 and self.is_primary:

                exec_time = 0
                for test_slice in self.fio_configs:
                    exec_time += sum(map(execution_time, test_slice))

                # +10% - is a rough estimation for additional operations
                # like sftp, etc
                exec_time = int(exec_time * 1.1)

                exec_time_s = sec_to_str(exec_time)
                now_dt = datetime.datetime.now()
                end_dt = now_dt + datetime.timedelta(0, exec_time)
                msg = "Entire test should takes aroud: {0} and finished at {1}"
                logger.info(msg.format(exec_time_s,
                                       end_dt.strftime("%H:%M:%S")))

            for pos, fio_cfg_slice in enumerate(self.fio_configs):
                fio_cfg_slice = list(fio_cfg_slice)
                names = [i.name for i in fio_cfg_slice]
                msgs = []
                already_processed = set()
                for name in names:
                    if name not in already_processed:
                        already_processed.add(name)

                        if 1 == names.count(name):
                            msgs.append(name)
                        else:
                            frmt = "{0} * {1}"
                            msgs.append(frmt.format(name,
                                                    names.count(name)))

                if self.is_primary:
                    logger.info("Will run tests: " + ", ".join(msgs))

                nolog = (pos != 0) or not self.is_primary
                out_err, interval = self.do_run(barrier, fio_cfg_slice,
                                                nolog=nolog)

                try:
                    full_raw_res = json.loads(out_err)

                    res = {"bw": [], "iops": [], "lat": [],
                           "clat": [], "slat": []}

                    for raw_result in full_raw_res['jobs']:
                        load_data = raw_result['mixed']

                        res["bw"].append(load_data["bw"])
                        res["iops"].append(load_data["iops"])
                        res["lat"].append(load_data["lat"]["mean"])
                        res["clat"].append(load_data["clat"]["mean"])
                        res["slat"].append(load_data["slat"]["mean"])

                    first = fio_cfg_slice[0]
                    p1 = first.vals.copy()
                    p1.pop('ramp_time', 0)

                    for nxt in fio_cfg_slice[1:]:
                        assert nxt.name == first.name
                        p2 = nxt.vals
                        p2.pop('_ramp_time', 0)

                        assert p1 == p2

                    tres = IOTestResults(first,
                                         self.config_params, res,
                                         full_raw_res, interval,
                                         vm_count=self.total_nodes_count)
                    self.on_result_cb(tres)
                except (OSError, StopTestError):
                    raise
                except Exception as exc:
                    msg_templ = "Error during postprocessing results: {0!s}"
                    raise RuntimeError(msg_templ.format(exc))

        finally:
            barrier.exit()

    def do_run(self, barrier, cfg_slice, nolog=False):
        # return open("/tmp/lit-sunshine/io/results.json").read(), (1, 2)
        conn_id = self.node.get_conn_id()

        cmd_templ = "fio --output-format=json --output={1} " + \
                    "--alloc-size=262144 {0}"

        if self.options.get("use_sudo", True):
            cmd_templ = "sudo " + cmd_templ

        task_fc = "\n\n".join(map(str, cfg_slice))
        with self.node.connection.open_sftp() as sftp:
            save_to_remote(sftp, self.task_file, task_fc)

        cmd = cmd_templ.format(self.task_file, self.results_file)

        exec_time = sum(map(execution_time, cfg_slice))
        exec_time_str = sec_to_str(exec_time)

        timeout = int(exec_time + max(300, exec_time))
        soft_tout = exec_time
        barrier.wait()

        if self.is_primary:
            templ = "Test should takes about {0}." + \
                    " Should finish at {1}," + \
                    " will wait at most till {2}"
            now_dt = datetime.datetime.now()
            end_dt = now_dt + datetime.timedelta(0, exec_time)
            wait_till = now_dt + datetime.timedelta(0, timeout)

            logger.info(templ.format(exec_time_str,
                                     end_dt.strftime("%H:%M:%S"),
                                     wait_till.strftime("%H:%M:%S")))

        task = BGSSHTask(self.node, self.options.get("use_sudo", True))
        begin = time.time()
        task.start(cmd)
        task.wait(soft_tout, timeout)
        end = time.time()

        if not nolog:
            logger.debug("Test on node {0} is finished".format(conn_id))

        with self.node.connection.open_sftp() as sftp:
            return read_from_remote(sftp, self.results_file), (begin, end)

    @classmethod
    def merge_results(cls, results):
        merged = results[0]
        for block in results[1:]:
            assert block["__meta__"] == merged["__meta__"]
            merged['res'].extend(block['res'])
        return merged

    @classmethod
    def format_for_console(cls, data, dinfo):
        return format_results_for_console(dinfo)
