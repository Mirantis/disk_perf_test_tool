import abc
import time
import socket
import random
import os.path
import logging
import datetime

from paramiko import SSHException, SFTPError

from wally.utils import (ssize_to_b, open_for_append_or_create,
                         sec_to_str, StopTestError)

from wally.ssh_utils import (copy_paths, run_over_ssh,
                             save_to_remote,
                             # delete_file,
                             connect, read_from_remote, Local)

from . import postgres
from .io import agent as io_agent
from .io import formatter as io_formatter
from .io.results_loader import parse_output


logger = logging.getLogger("wally")


class IPerfTest(object):
    def __init__(self, options, is_primary, on_result_cb, test_uuid, node,
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

        if 'run_script' in self.options:
            self.run_script = self.options['run_script']
            self.prepare_script = self.options['prepare_script']

    def get_remote_for_script(self, script):
        return os.path.join(self.tmp_dir, script.rpartition('/')[2])

    def copy_script(self, src):
        remote_path = self.get_remote_for_script(src)
        copy_paths(self.node.connection, {src: remote_path})
        return remote_path

    def pre_run(self):
        remote_script = self.copy_script(self.node.connection,
                                         self.pre_run_script)
        cmd = remote_script
        self.run_over_ssh(cmd)

    def run(self, barrier):
        remote_script = self.copy_script(self.node.connection, self.run_script)
        cmd_opts = ' '.join(["%s %s" % (key, val) for key, val
                             in self.options.items()])
        cmd = remote_script + ' ' + cmd_opts
        out_err = self.run_over_ssh(cmd)
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


class PgBenchTest(TwoScriptTest):
    root = os.path.dirname(postgres.__file__)
    prepare_script = os.path.join(root, "prepare.sh")
    run_script = os.path.join(root, "run.sh")


class IOPerfTest(IPerfTest):
    tcp_conn_timeout = 30
    max_pig_timeout = 5
    soft_runcycle = 5 * 60

    def __init__(self, *dt, **mp):
        IPerfTest.__init__(self, *dt, **mp)
        self.config_fname = self.options['cfg']
        self.alive_check_interval = self.options.get('alive_check_interval')
        self.config_params = self.options.get('params', {})
        self.tool = self.options.get('tool', 'fio')
        self.raw_cfg = open(self.config_fname).read()
        self.configs = list(io_agent.parse_all_in_1(self.raw_cfg,
                                                    self.config_params))

        cmd_log = os.path.join(self.log_directory, "task_compiled.cfg")
        raw_res = os.path.join(self.log_directory, "raw_results.txt")

        self.io_py_remote = self.join_remote("agent.py")
        self.log_fl = self.join_remote("log.txt")
        self.pid_file = self.join_remote("pid")
        self.task_file = self.join_remote("task.cfg")
        self.use_sudo = self.options.get("use_sudo", True)
        self.test_logging = self.options.get("test_logging", False)

        fio_command_file = open_for_append_or_create(cmd_log)

        if self.test_logging:
            soft_runcycle = self.soft_runcycle
        else:
            soft_runcycle = None

        self.fio_configs = io_agent.parse_and_slice_all_in_1(
                        self.raw_cfg,
                        self.config_params,
                        soft_runcycle=soft_runcycle)

        self.fio_configs = list(self.fio_configs)
        splitter = "\n\n" + "-" * 60 + "\n\n"

        cfg = splitter.join(
                map(io_agent.fio_config_to_str,
                    self.fio_configs))

        fio_command_file.write(cfg)
        self.fio_raw_results_file = open_for_append_or_create(raw_res)

    def __str__(self):
        return "{0}({1})".format(self.__class__.__name__,
                                 self.node.get_conn_id())

    def cleanup(self):
        # delete_file(conn, self.io_py_remote)
        # Need to remove tempo files, used for testing
        pass

    def prefill_test_files(self):
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
            logger.error(msg)
            raise StopTestError(msg, exc)

        self.install_utils()

        local_fname = os.path.splitext(io_agent.__file__)[0] + ".py"
        files_to_copy = {local_fname: self.io_py_remote}
        copy_paths(self.node.connection, files_to_copy)

        if self.options.get('prefill_files', True):
            self.prefill_test_files()
        elif self.is_primary:
            logger.warning("Prefilling of test files is disabled")

    def check_process_is_running(self, sftp, pid):
        try:
            sftp.stat("/proc/{0}".format(pid))
            return True
        except (OSError, IOError, NameError):
            return False

    def kill_remote_process(self, conn, pid, soft=True):
        try:
            if soft:
                cmd = "kill {0}"
            else:
                cmd = "kill -9 {0}"

            if self.use_sudo:
                cmd = "sudo " + cmd

            self.run_over_ssh(cmd.format(pid))
            return True
        except OSError:
            return False

    def get_test_status(self, die_timeout=3):
        is_connected = None
        is_running = None
        pid = None
        err = None

        try:
            conn = connect(self.node.conn_url,
                           conn_timeout=self.tcp_conn_timeout)
            with conn:
                with conn.open_sftp() as sftp:
                    try:
                        pid = read_from_remote(sftp, self.pid_file)
                        is_running = True
                    except (NameError, IOError) as exc:
                        pid = None
                        is_running = False

                    if is_running:
                        if not self.check_process_is_running(sftp, pid):
                            sftp.remove(self.pid_file)
                            is_running = False

            is_connected = True

        except (socket.error, SSHException, EOFError, SFTPError) as exc:
            err = str(exc)
            is_connected = False

        return is_connected, is_running, pid, err

    def wait_till_finished(self, soft_timeout, timeout):
        conn_id = self.node.get_conn_id()
        end_of_wait_time = timeout + time.time()
        soft_end_of_wait_time = soft_timeout + time.time()

        # time_till_check = random.randint(30, 90)
        time_till_check = 5
        pid = None
        is_running = False
        pid_get_timeout = self.max_pig_timeout + time.time()
        curr_connected = True

        while end_of_wait_time > time.time():
            time.sleep(time_till_check)

            is_connected, is_running, npid, err = self.get_test_status()

            if is_connected and not is_running:
                if pid is None:
                    if time.time() > pid_get_timeout:
                        msg = ("On node {0} pid file doesn't " +
                               "appears in time")
                        logger.error(msg.format(conn_id))
                        raise StopTestError("Start timeout")
                else:
                    # execution finished
                    break

            if npid is not None:
                pid = npid

            if is_connected and pid is not None and is_running:
                if time.time() < soft_end_of_wait_time:
                    time.sleep(soft_end_of_wait_time - time.time())

            if is_connected and not curr_connected:
                msg = "Connection with {0} is restored"
                logger.debug(msg.format(conn_id))
            elif not is_connected and curr_connected:
                msg = "Lost connection with " + conn_id + ". Error: " + err
                logger.debug(msg)

            curr_connected = is_connected

    def run(self, barrier):
        try:
            if len(self.fio_configs) > 1 and self.is_primary:

                exec_time = 0
                for test in self.fio_configs:
                    exec_time += io_agent.calculate_execution_time(test)

                exec_time_s = sec_to_str(exec_time)
                now_dt = datetime.datetime.now()
                end_dt = now_dt + datetime.timedelta(0, exec_time)
                msg = "Entire test should takes aroud: {0} and finished at {1}"
                logger.info(msg.format(exec_time_s,
                                       end_dt.strftime("%H:%M:%S")))

            for pos, fio_cfg_slice in enumerate(self.fio_configs):
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
                out_err = self.do_run(barrier, fio_cfg_slice, nolog=nolog)

                try:
                    for data in parse_output(out_err):
                        data['__meta__']['raw_cfg'] = self.raw_cfg
                        self.on_result_cb(data)
                except (OSError, StopTestError):
                    raise
                except Exception as exc:
                    msg_templ = "Error during postprocessing results: {0!s}"
                    raise RuntimeError(msg_templ.format(exc))

        finally:
            barrier.exit()

    def do_run(self, barrier, cfg, nolog=False):
        conn_id = self.node.get_conn_id()

        cmd_templ = "screen -S {screen_name} -d -m " + \
                    "env python2 {0} -p {pid_file} -o {results_file} " + \
                    "--type {1} {2} --json {3}"

        if self.options.get("use_sudo", True):
            cmd_templ = "sudo " + cmd_templ

        params = " ".join("{0}={1}".format(k, v)
                          for k, v in self.config_params.items())

        if "" != params:
            params = "--params " + params

        with self.node.connection.open_sftp() as sftp:
            save_to_remote(sftp, self.task_file,
                           io_agent.fio_config_to_str(cfg))

        screen_name = self.test_uuid
        cmd = cmd_templ.format(self.io_py_remote,
                               self.tool,
                               params,
                               self.task_file,
                               pid_file=self.pid_file,
                               results_file=self.log_fl,
                               screen_name=screen_name)

        exec_time = io_agent.calculate_execution_time(cfg)
        exec_time_str = sec_to_str(exec_time)

        timeout = int(exec_time + max(300, exec_time))
        barrier.wait()
        self.run_over_ssh(cmd, nolog=nolog)

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

            if not nolog:
                msg = "Tests started in screen {1} on each testnode"
                logger.debug(msg.format(conn_id, screen_name))

        # TODO: add monitoring socket
        if self.node.connection is not Local:
            self.node.connection.close()

        self.wait_till_finished(timeout)
        if not nolog:
            logger.debug("Test on node {0} is finished".format(conn_id))

        if self.node.connection is not Local:
            conn_timeout = self.tcp_conn_timeout * 3
            self.node.connection = connect(self.node.conn_url,
                                           conn_timeout=conn_timeout)

        with self.node.connection.open_sftp() as sftp:
            return read_from_remote(sftp, self.log_fl)

    def merge_results(self, results):
        if len(results) == 0:
            return None

        merged_result = results[0]
        merged_data = merged_result['res']
        mergable_fields = ['bw', 'clat', 'iops', 'lat', 'slat']

        for res in results[1:]:
            assert res['__meta__'] == merged_result['__meta__']
            data = res['res']

            for testname, test_data in data.items():
                if testname not in merged_data:
                    merged_data[testname] = test_data
                    continue

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
