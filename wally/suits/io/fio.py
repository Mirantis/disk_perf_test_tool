import re
import time
import json
import os.path
import logging
import datetime
import functools
import subprocess
import collections

import yaml
import paramiko
import texttable
from paramiko.ssh_exception import SSHException
from concurrent.futures import ThreadPoolExecutor

from wally.pretty_yaml import dumps
from wally.statistic import round_3_digit, data_property
from wally.utils import ssize2b, sec_to_str, StopTestError, Barrier, get_os
from wally.ssh_utils import (save_to_remote, read_from_remote, BGSSHTask, reconnect)

from .fio_task_parser import (execution_time, fio_cfg_compile,
                              get_test_summary, get_test_sync_mode)
from ..itest import TimeSeriesValue, PerfTest, TestResults, run_on_node

logger = logging.getLogger("wally")


# Results folder structure
# results/
#     {loadtype}_{num}/
#         config.yaml
#         ......


class NoData(object):
    pass


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


def load_fio_log_file(fname):
    with open(fname) as fd:
        it = [ln.split(',')[:2] for ln in fd]
    vals = [(float(off) / 1000, float(val.strip())) for off, val in it]
    return TimeSeriesValue(vals)


def load_test_results(cls, folder, run_num):
    res = {}
    params = None

    fn = os.path.join(folder, str(run_num) + '_params.yaml')
    params = yaml.load(open(fn).read())

    conn_ids = set()
    for fname in os.listdir(folder):
        rr = r"{0}_(?P<conn_id>.*?)_(?P<type>[^_.]*)\.\d+\.log$".format(run_num)
        rm = re.match(rr, fname)
        if rm is None:
            continue

        conn_id_s = rm.group('conn_id')
        conn_id = conn_id_s.replace('_', ':')
        ftype = rm.group('type')

        if ftype not in ('iops', 'bw', 'lat'):
            continue

        try:
            ts = load_fio_log_file(os.path.join(folder, fname))
            if ftype in res:
                assert conn_id not in res[ftype]

            res.setdefault(ftype, {})[conn_id] = ts
        except AssertionError:
            pass

        conn_ids.add(conn_id)

    raw_res = {}
    for conn_id in conn_ids:
        fn = os.path.join(folder, "{0}_{1}_rawres.json".format(run_num, conn_id_s))

        # remove message hack
        fc = "{" + open(fn).read().split('{', 1)[1]
        raw_res[conn_id] = json.loads(fc)

    return cls(params, res, raw_res)


class Attrmapper(object):
    def __init__(self, dct):
        self.__dct = dct

    def __getattr__(self, name):
        try:
            return self.__dct[name]
        except KeyError:
            raise AttributeError(name)


class DiskPerfInfo(object):
    def __init__(self, name, summary, params, testnodes_count):
        self.name = name
        self.bw = None
        self.iops = None
        self.lat = None
        self.lat_50 = None
        self.lat_95 = None

        self.raw_bw = []
        self.raw_iops = []
        self.raw_lat = []

        self.params = params
        self.testnodes_count = testnodes_count
        self.summary = summary
        self.p = Attrmapper(self.params['vals'])

        self.sync_mode = get_test_sync_mode(self.params['vals'])
        self.concurence = self.params['vals'].get('numjobs', 1)


def get_lat_perc_50_95(lat_mks):
    curr_perc = 0
    perc_50 = None
    perc_95 = None
    pkey = None
    for key, val in sorted(lat_mks.items()):
        if curr_perc + val >= 50 and perc_50 is None:
            if pkey is None or val < 1.:
                perc_50 = key
            else:
                perc_50 = (50. - curr_perc) / val * (key - pkey) + pkey

        if curr_perc + val >= 95:
            if pkey is None or val < 1.:
                perc_95 = key
            else:
                perc_95 = (95. - curr_perc) / val * (key - pkey) + pkey
            break

        pkey = key
        curr_perc += val

    return perc_50 / 1000., perc_95 / 1000.


def prepare(ramp_time, data, avg_interval):
    if data is None:
        return data

    res = {}
    for key, ts_data in data.items():
        if ramp_time > 0:
            ts_data = ts_data.skip(ramp_time)

        res[key] = ts_data.derived(avg_interval)
    return res


class IOTestResult(TestResults):
    """
    Fio run results
    config: TestConfig
    fio_task: FioJobSection
    ts_results: {str: TimeSeriesValue}
    raw_result: ????
    run_interval:(float, float) - test tun time, used for sensors
    """
    def __init__(self, config, fio_task, ts_results, raw_result, run_interval):

        self.name = fio_task.name.split("_")[0]
        self.fio_task = fio_task

        ramp_time = fio_task.vals.get('ramp_time', 0)

        self.bw = prepare(ramp_time, ts_results.get('bw'), 1.0)
        self.lat = prepare(ramp_time, ts_results.get('lat'), 1.0)
        self.iops = prepare(ramp_time, ts_results.get('iops'), 1.0)
        # self.slat = drop_warmup(res.get('clat', None), self.params)
        # self.clat = drop_warmup(res.get('slat', None), self.params)

        res = {"bw": self.bw, "lat": self.lat, "iops": self.iops}

        self.sensors_data = None
        self._pinfo = None
        TestResults.__init__(self, config, res, raw_result, run_interval)

    def summary(self):
        return get_test_summary(self.fio_task) + "vm" \
               + str(len(self.config.nodes))

    def get_yamable(self):
        return self.summary()

    @property
    def disk_perf_info(self):
        if self._pinfo is not None:
            return self._pinfo

        lat_mks = collections.defaultdict(lambda: 0)
        num_res = 0

        for _, result in self.raw_result.items():
            num_res += len(result['jobs'])
            for job_info in result['jobs']:
                for k, v in job_info['latency_ms'].items():
                    if isinstance(k, basestring) and k.startswith('>='):
                        lat_mks[int(k[2:]) * 1000] += v
                    else:
                        lat_mks[int(k) * 1000] += v

                for k, v in job_info['latency_us'].items():
                    lat_mks[int(k)] += v

        for k, v in lat_mks.items():
            lat_mks[k] = float(v) / num_res

        testnodes_count = len(self.fio_raw_res)

        pinfo = DiskPerfInfo(self.name,
                             self.summary(),
                             self.params,
                             testnodes_count)

        pinfo.raw_bw = [res.vals() for res in self.bw.values()]
        pinfo.raw_iops = [res.vals() for res in self.iops.values()]
        pinfo.raw_lat = [res.vals() for res in self.lat.values()]

        pinfo.bw = data_property(map(sum, zip(*pinfo.raw_bw)))
        pinfo.iops = data_property(map(sum, zip(*pinfo.raw_iops)))
        pinfo.lat = data_property(sum(pinfo.raw_lat, []))
        pinfo.lat_50, pinfo.lat_95 = get_lat_perc_50_95(lat_mks)

        self._pinfo = pinfo

        return pinfo


class IOPerfTest(PerfTest):
    tcp_conn_timeout = 30
    max_pig_timeout = 5
    soft_runcycle = 5 * 60

    def __init__(self, config):
        PerfTest.__init__(self, config)

        get = self.config.params.get
        do_get = self.config.params.__getitem__

        self.config_fname = do_get('cfg')

        if '/' not in self.config_fname and '.' not in self.config_fname:
            cfgs_dir = os.path.dirname(__file__)
            self.config_fname = os.path.join(cfgs_dir,
                                             self.config_fname + '.cfg')

        self.alive_check_interval = get('alive_check_interval')
        self.use_system_fio = get('use_system_fio', False)

        self.config_params = get('params', {}).copy()

        self.io_py_remote = self.join_remote("agent.py")
        self.results_file = self.join_remote("results.json")
        self.pid_file = self.join_remote("pid")
        self.task_file = self.join_remote("task.cfg")
        self.sh_file = self.join_remote("cmd.sh")
        self.err_out_file = self.join_remote("fio_err_out")
        self.exit_code_file = self.join_remote("exit_code")

        self.use_sudo = get("use_sudo", True)
        self.test_logging = get("test_logging", False)

        self.raw_cfg = open(self.config_fname).read()
        self.fio_configs = fio_cfg_compile(self.raw_cfg,
                                           self.config_fname,
                                           self.config_params,
                                           split_on_names=self.test_logging)
        self.fio_configs = list(self.fio_configs)

    @classmethod
    def load(cls, folder):
        for fname in os.listdir(folder):
            if re.match("\d+_params.yaml$", fname):
                num = int(fname.split('_')[0])
                yield load_test_results(IOTestResult, folder, num)

    def cleanup(self):
        # delete_file(conn, self.io_py_remote)
        # Need to remove tempo files, used for testing
        pass

    def prefill_test_files(self, files, rossh):
        cmd_templ = "fio --name=xxx --filename={0} --direct=1" + \
                    " --bs=4m --size={1}m --rw=write"

        if self.use_sudo:
            cmd_templ = "sudo " + cmd_templ

        ssize = 0
        stime = time.time()

        for fname, curr_sz in files.items():
            cmd = cmd_templ.format(fname, curr_sz)
            ssize += curr_sz

            rossh(cmd, timeout=curr_sz)

        ddtime = time.time() - stime
        if ddtime > 1E-3:
            fill_bw = int(ssize / ddtime)
            mess = "Initiall dd fill bw is {0} MiBps for this vm"
            logger.info(mess.format(fill_bw))
        return fill_bw

    def install_utils(self, rossh, max_retry=3, timeout=5):
        need_install = []
        packs = [('screen', 'screen')]

        if self.use_system_fio:
            packs.append(('fio', 'fio'))
        else:
            # define OS and x32/x64
            # copy appropriate fio
            # add fio deps
            pass

        for bin_name, package in packs:
            if bin_name is None:
                need_install.append(package)
                continue

            try:
                rossh('which ' + bin_name, nolog=True)
            except OSError:
                need_install.append(package)

        if len(need_install) == 0:
            return

        if 'redhat' == get_os(rossh):
            cmd = "sudo yum -y install " + " ".join(need_install)
        else:
            cmd = "sudo apt-get -y install " + " ".join(need_install)

        for _ in range(max_retry):
            try:
                rossh(cmd)
                break
            except OSError as err:
                time.sleep(timeout)
        else:
            raise OSError("Can't install - " + str(err))

    def pre_run(self):
        prefill = False
        prefill = self.config.options.get('prefill_files', True)

        if prefill:
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
        else:
            files = None
            logger.warning("Prefilling of test files is disabled")

        with ThreadPoolExecutor(len(self.config.nodes)) as pool:
            fc = functools.partial(self.pre_run_th, files=files)
            list(pool.map(fc, self.config.nodes))

    def pre_run_th(self, node, files):
        # fill files with pseudo-random data
        rossh = run_on_node(node)

        try:
            cmd = 'mkdir -p "{0}"'.format(self.config.remote_dir)
            if self.use_sudo:
                cmd = "sudo " + cmd
                cmd += " ; sudo chown {0} {1}".format(self.node.get_user(),
                                                      self.config.remote_dir)

            rossh(cmd)
        except Exception as exc:
            msg = "Failed to create folder {0} on remote {1}. Error: {2!s}"
            msg = msg.format(self.config.remote_dir, self.node.get_conn_id(), exc)
            logger.exception(msg)
            raise StopTestError(msg, exc)

        if files is not None:
            self.prefill_test_files(rossh, files)

        self.install_utils(rossh)

    def run(self):
        if len(self.fio_configs) > 1:
            # +10% - is a rough estimation for additional operations
            # like sftp, etc
            exec_time = int(sum(map(execution_time, self.fio_configs)) * 1.1)
            exec_time_s = sec_to_str(exec_time)
            now_dt = datetime.datetime.now()
            end_dt = now_dt + datetime.timedelta(0, exec_time)
            msg = "Entire test should takes aroud: {0} and finished at {1}"
            logger.info(msg.format(exec_time_s,
                                   end_dt.strftime("%H:%M:%S")))

        tname = os.path.basename(self.config_fname)
        if tname.endswith('.cfg'):
            tname = tname[:-4]

        barrier = Barrier(len(self.config.nodes))
        results = []

        with ThreadPoolExecutor(len(self.config.nodes)) as pool:
            for pos, fio_cfg in enumerate(self.fio_configs):
                logger.info("Will run {0} test".format(fio_cfg.name))

                templ = "Test should takes about {0}." + \
                        " Should finish at {1}," + \
                        " will wait at most till {2}"
                exec_time = execution_time(fio_cfg)
                exec_time_str = sec_to_str(exec_time)
                timeout = int(exec_time + max(300, exec_time))

                now_dt = datetime.datetime.now()
                end_dt = now_dt + datetime.timedelta(0, exec_time)
                wait_till = now_dt + datetime.timedelta(0, timeout)

                logger.info(templ.format(exec_time_str,
                                         end_dt.strftime("%H:%M:%S"),
                                         wait_till.strftime("%H:%M:%S")))

                func = functools.partial(self.do_run,
                                         barrier=barrier,
                                         fio_cfg=fio_cfg,
                                         pos=pos)

                max_retr = 3
                for idx in range(max_retr):
                    try:
                        intervals = list(pool.map(func, self.config.nodes))
                        break
                    except (EnvironmentError, SSHException) as exc:
                        logger.exception("During fio run")
                        if idx == max_retr - 1:
                            raise StopTestError("Fio failed", exc)

                    logger.info("Sleeping 30s and retrying")
                    time.sleep(30)

                fname = "{0}_task.fio".format(pos)
                with open(os.path.join(self.config.log_directory, fname), "w") as fd:
                    fd.write(str(fio_cfg))

                params = {'vm_count': len(self.config.nodes)}
                params['name'] = fio_cfg.name
                params['vals'] = dict(fio_cfg.vals.items())
                params['intervals'] = intervals
                params['nodes'] = [node.get_conn_id() for node in self.config.nodes]

                fname = "{0}_params.yaml".format(pos)
                with open(os.path.join(self.config.log_directory, fname), "w") as fd:
                    fd.write(dumps(params))

                res = load_test_results(self.config.log_directory, pos)
                results.append(res)
        return results

    def do_run(self, node, barrier, fio_cfg, pos, nolog=False):
        exec_folder = os.path.dirname(self.task_file)
        bash_file = "#!/bin/bash\n" + \
                    "cd {exec_folder}\n" + \
                    "fio --output-format=json --output={out_file} " + \
                    "--alloc-size=262144 {job_file} " + \
                    " >{err_out_file} 2>&1 \n" + \
                    "echo $? >{res_code_file}\n"

        bash_file = bash_file.format(out_file=self.results_file,
                                     job_file=self.task_file,
                                     err_out_file=self.err_out_file,
                                     res_code_file=self.exit_code_file,
                                     exec_folder=exec_folder)

        run_on_node(node)("cd {0} ; rm -rf *".format(exec_folder), nolog=True)

        with node.connection.open_sftp() as sftp:
            print ">>>>", self.task_file
            save_to_remote(sftp, self.task_file, str(fio_cfg))
            save_to_remote(sftp, self.sh_file, bash_file)

        exec_time = execution_time(fio_cfg)

        timeout = int(exec_time + max(300, exec_time))
        soft_tout = exec_time

        begin = time.time()

        if self.config.options.get("use_sudo", True):
            sudo = "sudo "
        else:
            sudo = ""

        fnames_before = run_on_node(node)("ls -1 " + exec_folder, nolog=True)

        barrier.wait()

        task = BGSSHTask(node, self.config.options.get("use_sudo", True))
        task.start(sudo + "bash " + self.sh_file)

        while True:
            try:
                task.wait(soft_tout, timeout)
                break
            except paramiko.SSHException:
                pass

            try:
                node.connection.close()
            except:
                pass

            reconnect(node.connection, node.conn_url)

        end = time.time()
        rossh = run_on_node(node)
        fnames_after = rossh("ls -1 " + exec_folder, nolog=True)

        conn_id = node.get_conn_id().replace(":", "_")
        if not nolog:
            logger.debug("Test on node {0} is finished".format(conn_id))

        log_files_pref = []
        if 'write_lat_log' in fio_cfg.vals:
            fname = fio_cfg.vals['write_lat_log']
            log_files_pref.append(fname + '_clat')
            log_files_pref.append(fname + '_lat')
            log_files_pref.append(fname + '_slat')

        if 'write_iops_log' in fio_cfg.vals:
            fname = fio_cfg.vals['write_iops_log']
            log_files_pref.append(fname + '_iops')

        if 'write_bw_log' in fio_cfg.vals:
            fname = fio_cfg.vals['write_bw_log']
            log_files_pref.append(fname + '_bw')

        files = collections.defaultdict(lambda: [])
        all_files = [os.path.basename(self.results_file)]
        new_files = set(fnames_after.split()) - set(fnames_before.split())
        for fname in new_files:
            if fname.endswith('.log') and fname.split('.')[0] in log_files_pref:
                name, _ = os.path.splitext(fname)
                if fname.count('.') == 1:
                    tp = name.split("_")[-1]
                    cnt = 0
                else:
                    tp_cnt = name.split("_")[-1]
                    tp, cnt = tp_cnt.split('.')
                files[tp].append((int(cnt), fname))
                all_files.append(fname)

        arch_name = self.join_remote('wally_result.tar.gz')
        tmp_dir = os.path.join(self.config.log_directory, 'tmp_' + conn_id)
        os.mkdir(tmp_dir)
        loc_arch_name = os.path.join(tmp_dir, 'wally_result.{0}.tar.gz'.format(conn_id))
        file_full_names = " ".join(all_files)

        try:
            os.unlink(loc_arch_name)
        except:
            pass

        with node.connection.open_sftp() as sftp:
            exit_code = read_from_remote(sftp, self.exit_code_file)
            err_out = read_from_remote(sftp, self.err_out_file)
            exit_code = exit_code.strip()

            if exit_code != '0':
                msg = "fio exit with code {0}: {1}".format(exit_code, err_out)
                logger.critical(msg.strip())
                raise StopTestError("fio failed")

            rossh("rm -f {0}".format(arch_name), nolog=True)
            cmd = "cd {0} ; tar zcvf {1} {2}".format(exec_folder, arch_name, file_full_names)
            rossh(cmd, nolog=True)
            sftp.get(arch_name, loc_arch_name)

        cmd = "cd {0} ; tar xvzf {1} >/dev/null".format(tmp_dir, loc_arch_name)
        subprocess.check_call(cmd, shell=True)
        os.unlink(loc_arch_name)

        for ftype, fls in files.items():
            for idx, fname in fls:
                cname = os.path.join(tmp_dir, fname)
                loc_fname = "{0}_{1}_{2}.{3}.log".format(pos, conn_id, ftype, idx)
                loc_path = os.path.join(self.config.log_directory, loc_fname)
                os.rename(cname, loc_path)

        cname = os.path.join(tmp_dir,
                             os.path.basename(self.results_file))
        loc_fname = "{0}_{1}_rawres.json".format(pos, conn_id)
        loc_path = os.path.join(self.config.log_directory, loc_fname)
        os.rename(cname, loc_path)

        os.rmdir(tmp_dir)
        return begin, end

    @classmethod
    def format_for_console(cls, data, dinfo):
        """
        create a table with io performance report
        for console
        """

        def getconc(data):
            th_count = data.params['vals'].get('numjobs')

            if th_count is None:
                th_count = data.params['vals'].get('concurence', 1)
            return th_count

        def key_func(data):
            p = data.params['vals']

            th_count = getconc(data)

            return (data.name.rsplit("_", 1)[0],
                    p['rw'],
                    get_test_sync_mode(data.params),
                    ssize2b(p['blocksize']),
                    int(th_count) * data.testnodes_count)

        tab = texttable.Texttable(max_width=120)
        tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
        tab.set_cols_align(["l", "l", "r", "r", "r", "r", "r", "r", "r"])

        items = sorted(dinfo.values(), key=key_func)

        prev_k = None
        header = ["Name", "Description", "iops\ncum", "KiBps\ncum",
                  "Cnf\n95%", "Dev%", "iops\nper vm", "KiBps\nper vm", "lat\nms"]

        for data in items:
            curr_k = key_func(data)[:4]

            if prev_k is not None:
                if prev_k != curr_k:
                    tab.add_row(
                        ["-------", "-----------", "-----", "------",
                         "---", "----", "------", "---", "-----"])

            prev_k = curr_k

            test_dinfo = dinfo[(data.name, data.summary)]

            iops, _ = test_dinfo.iops.rounded_average_conf()

            bw, bw_conf = test_dinfo.bw.rounded_average_conf()
            _, bw_dev = test_dinfo.bw.rounded_average_dev()
            conf_perc = int(round(bw_conf * 100 / bw))
            dev_perc = int(round(bw_dev * 100 / bw))

            lat, _ = test_dinfo.lat.rounded_average_conf()
            lat = round_3_digit(int(lat) // 1000)

            iops_per_vm = round_3_digit(iops / data.testnodes_count)
            bw_per_vm = round_3_digit(bw / data.testnodes_count)

            iops = round_3_digit(iops)
            bw = round_3_digit(bw)

            params = (data.name.rsplit('_', 1)[0],
                      data.summary, int(iops), int(bw), str(conf_perc),
                      str(dev_perc),
                      int(iops_per_vm), int(bw_per_vm), lat)
            tab.add_row(params)

        tab.header(header)

        return tab.draw()
