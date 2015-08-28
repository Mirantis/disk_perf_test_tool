import re
import time
import json
import stat
import random
import shutil
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
from concurrent.futures import ThreadPoolExecutor, wait

import wally
from wally.pretty_yaml import dumps
from wally.statistic import round_3_digit, data_property, average
from wally.utils import ssize2b, sec_to_str, StopTestError, Barrier, get_os
from wally.ssh_utils import (save_to_remote, read_from_remote, BGSSHTask, reconnect)

from .fio_task_parser import (execution_time, fio_cfg_compile,
                              get_test_summary, get_test_summary_tuple,
                              get_test_sync_mode, FioJobSection)

from ..itest import (TimeSeriesValue, PerfTest, TestResults,
                     run_on_node, TestConfig, MeasurementMatrix)

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

    vals = [(float(off) / 1000,  # convert us to ms
             float(val.strip()) + 0.5)  # add 0.5 to compemsate average value
                                        # as fio trimm all values in log to integer
            for off, val in it]

    return TimeSeriesValue(vals)


READ_IOPS_DISCSTAT_POS = 3
WRITE_IOPS_DISCSTAT_POS = 7


def load_sys_log_file(ftype, fname):
    assert ftype == 'iops'
    pval = None
    with open(fname) as fd:
        iops = []
        for ln in fd:
            params = ln.split()
            cval = int(params[WRITE_IOPS_DISCSTAT_POS]) + \
                int(params[READ_IOPS_DISCSTAT_POS])
            if pval is not None:
                iops.append(cval - pval)
            pval = cval

    vals = [(idx * 1000, val) for idx, val in enumerate(iops)]
    return TimeSeriesValue(vals)


def load_test_results(folder, run_num):
    res = {}
    params = None

    fn = os.path.join(folder, str(run_num) + '_params.yaml')
    params = yaml.load(open(fn).read())

    conn_ids_set = set()
    rr = r"{0}_(?P<conn_id>.*?)_(?P<type>[^_.]*)\.\d+\.log$".format(run_num)
    for fname in os.listdir(folder):
        rm = re.match(rr, fname)
        if rm is None:
            continue

        conn_id_s = rm.group('conn_id')
        conn_id = conn_id_s.replace('_', ':')
        ftype = rm.group('type')

        if ftype not in ('iops', 'bw', 'lat'):
            continue

        ts = load_fio_log_file(os.path.join(folder, fname))
        res.setdefault(ftype, {}).setdefault(conn_id, []).append(ts)

        conn_ids_set.add(conn_id)

    rr = r"{0}_(?P<conn_id>.*?)_(?P<type>[^_.]*)\.sys\.log$".format(run_num)
    for fname in os.listdir(folder):
        rm = re.match(rr, fname)
        if rm is None:
            continue

        conn_id_s = rm.group('conn_id')
        conn_id = conn_id_s.replace('_', ':')
        ftype = rm.group('type')

        if ftype not in ('iops', 'bw', 'lat'):
            continue

        ts = load_sys_log_file(ftype, os.path.join(folder, fname))
        res.setdefault(ftype + ":sys", {}).setdefault(conn_id, []).append(ts)

        conn_ids_set.add(conn_id)

    mm_res = {}

    if len(res) == 0:
        raise ValueError("No data was found")

    for key, data in res.items():
        conn_ids = sorted(conn_ids_set)
        matr = [data[conn_id] for conn_id in conn_ids]

        mm_res[key] = MeasurementMatrix(matr, conn_ids)

    raw_res = {}
    for conn_id in conn_ids:
        fn = os.path.join(folder, "{0}_{1}_rawres.json".format(run_num, conn_id_s))

        # remove message hack
        fc = "{" + open(fn).read().split('{', 1)[1]
        raw_res[conn_id] = json.loads(fc)

    fio_task = FioJobSection(params['name'])
    fio_task.vals.update(params['vals'])

    config = TestConfig('io', params, None, params['nodes'], folder, None)
    return FioRunResult(config, fio_task, mm_res, raw_res, params['intervals'], run_num)


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
        self.lat_avg = None

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

    # for k, v in sorted(lat_mks.items()):
    #     if k / 1000 > 0:
    #         print "{0:>4}".format(k / 1000), v

    # print perc_50 / 1000., perc_95 / 1000.
    # exit(1)
    return perc_50 / 1000., perc_95 / 1000.


class IOTestResults(object):
    def __init__(self, suite_name, fio_results, log_directory):
        self.suite_name = suite_name
        self.fio_results = fio_results
        self.log_directory = log_directory

    def __iter__(self):
        return iter(self.fio_results)

    def __len__(self):
        return len(self.fio_results)

    def get_yamable(self):
        items = [(fio_res.summary(), fio_res.idx) for fio_res in self]
        return {self.suite_name: [self.log_directory] + items}


class FioRunResult(TestResults):
    """
    Fio run results
    config: TestConfig
    fio_task: FioJobSection
    ts_results: {str: MeasurementMatrix[TimeSeriesValue]}
    raw_result: ????
    run_interval:(float, float) - test tun time, used for sensors
    """
    def __init__(self, config, fio_task, ts_results, raw_result, run_interval, idx):

        self.name = fio_task.name.rsplit("_", 1)[0]
        self.fio_task = fio_task
        self.idx = idx

        self.bw = ts_results['bw']
        self.lat = ts_results['lat']
        self.iops = ts_results['iops']

        if 'iops:sys' in ts_results:
            self.iops_sys = ts_results['iops:sys']
        else:
            self.iops_sys = None

        res = {"bw": self.bw,
               "lat": self.lat,
               "iops": self.iops,
               "iops:sys": self.iops_sys}

        self.sensors_data = None
        self._pinfo = None
        TestResults.__init__(self, config, res, raw_result, run_interval)

    def get_params_from_fio_report(self):
        nodes = self.bw.connections_ids

        iops = [self.raw_result[node]['jobs'][0]['mixed']['iops'] for node in nodes]
        total_ios = [self.raw_result[node]['jobs'][0]['mixed']['total_ios'] for node in nodes]
        runtime = [self.raw_result[node]['jobs'][0]['mixed']['runtime'] / 1000 for node in nodes]
        flt_iops = [float(ios) / rtime for ios, rtime in zip(total_ios, runtime)]

        bw = [self.raw_result[node]['jobs'][0]['mixed']['bw'] for node in nodes]
        total_bytes = [self.raw_result[node]['jobs'][0]['mixed']['io_bytes'] for node in nodes]
        flt_bw = [float(tbytes) / rtime for tbytes, rtime in zip(total_bytes, runtime)]

        return {'iops': iops,
                'flt_iops': flt_iops,
                'bw': bw,
                'flt_bw': flt_bw}

    def summary(self):
        return get_test_summary(self.fio_task, len(self.config.nodes))

    def summary_tpl(self):
        return get_test_summary_tuple(self.fio_task, len(self.config.nodes))

    def get_lat_perc_50_95_multy(self):
        lat_mks = collections.defaultdict(lambda: 0)
        num_res = 0

        for result in self.raw_result.values():
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
        return get_lat_perc_50_95(lat_mks)

    def disk_perf_info(self, avg_interval=2.0):

        if self._pinfo is not None:
            return self._pinfo

        testnodes_count = len(self.config.nodes)

        pinfo = DiskPerfInfo(self.name,
                             self.summary(),
                             self.params,
                             testnodes_count)

        def prepare(data, drop=1):
            if data is None:
                return data

            res = []
            for ts_data in data:
                if ts_data.average_interval() < avg_interval:
                    ts_data = ts_data.derived(avg_interval)

                # drop last value on bounds
                # as they may contains ranges without activities
                assert len(ts_data.values) >= drop + 1, str(drop) + " " + str(ts_data.values)

                if drop > 0:
                    res.append(ts_data.values[:-drop])
                else:
                    res.append(ts_data.values)

            return res

        def agg_data(matr):
            arr = sum(matr, [])
            min_len = min(map(len, arr))
            res = []
            for idx in range(min_len):
                res.append(sum(dt[idx] for dt in arr))
            return res

        pinfo.raw_lat = map(prepare, self.lat.per_vm())
        num_th = sum(map(len, pinfo.raw_lat))
        lat_avg = [val / num_th for val in agg_data(pinfo.raw_lat)]
        pinfo.lat_avg = data_property(lat_avg).average / 1000  # us to ms

        pinfo.lat_50, pinfo.lat_95 = self.get_lat_perc_50_95_multy()
        pinfo.lat = pinfo.lat_50

        pinfo.raw_bw = map(prepare, self.bw.per_vm())
        pinfo.raw_iops = map(prepare, self.iops.per_vm())

        if self.iops_sys is not None:
            pinfo.raw_iops_sys = map(prepare, self.iops_sys.per_vm())
            pinfo.iops_sys = data_property(agg_data(pinfo.raw_iops_sys))
        else:
            pinfo.raw_iops_sys = None
            pinfo.iops_sys = None

        fparams = self.get_params_from_fio_report()
        fio_report_bw = sum(fparams['flt_bw'])
        fio_report_iops = sum(fparams['flt_iops'])

        agg_bw = agg_data(pinfo.raw_bw)
        agg_iops = agg_data(pinfo.raw_iops)

        log_bw_avg = average(agg_bw)
        log_iops_avg = average(agg_iops)

        # update values to match average from fio report
        coef_iops = fio_report_iops / float(log_iops_avg)
        coef_bw = fio_report_bw / float(log_bw_avg)

        bw_log = data_property([val * coef_bw for val in agg_bw])
        iops_log = data_property([val * coef_iops for val in agg_iops])

        bw_report = data_property([fio_report_bw])
        iops_report = data_property([fio_report_iops])

        # When IOPS/BW per thread is too low
        # data from logs is rounded to match
        iops_per_th = sum(sum(pinfo.raw_iops, []), [])
        if average(iops_per_th) > 10:
            pinfo.iops = iops_log
            pinfo.iops2 = iops_report
        else:
            pinfo.iops = iops_report
            pinfo.iops2 = iops_log

        bw_per_th = sum(sum(pinfo.raw_bw, []), [])
        if average(bw_per_th) > 10:
            pinfo.bw = bw_log
            pinfo.bw2 = bw_report
        else:
            pinfo.bw = bw_report
            pinfo.bw2 = bw_log

        self._pinfo = pinfo

        return pinfo


class IOPerfTest(PerfTest):
    tcp_conn_timeout = 30
    max_pig_timeout = 5
    soft_runcycle = 5 * 60
    retry_time = 30

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

        if get('prefill_files') is not None:
            logger.warning("prefill_files option is depricated. Use force_prefill instead")

        self.force_prefill = get('force_prefill', False)
        self.config_params = get('params', {}).copy()

        self.io_py_remote = self.join_remote("agent.py")
        self.results_file = self.join_remote("results.json")
        self.pid_file = self.join_remote("pid")
        self.task_file = self.join_remote("task.cfg")
        self.sh_file = self.join_remote("cmd.sh")
        self.err_out_file = self.join_remote("fio_err_out")
        self.io_log_file = self.join_remote("io_log.txt")
        self.exit_code_file = self.join_remote("exit_code")

        self.max_latency = get("max_lat", None)
        self.min_bw_per_thread = get("min_bw", None)

        self.use_sudo = get("use_sudo", True)

        self.raw_cfg = open(self.config_fname).read()
        self.fio_configs = None

    @classmethod
    def load(cls, suite_name, folder):
        res = []
        for fname in os.listdir(folder):
            if re.match("\d+_params.yaml$", fname):
                num = int(fname.split('_')[0])
                res.append(load_test_results(folder, num))
        return IOTestResults(suite_name, res, folder)

    def cleanup(self):
        # delete_file(conn, self.io_py_remote)
        # Need to remove tempo files, used for testing
        pass

    # size is megabytes
    def check_prefill_required(self, rossh, fname, size, num_blocks=16):
        try:
            with rossh.connection.open_sftp() as sftp:
                fstats = sftp.stat(fname)

            if stat.S_ISREG(fstats.st_mode) and fstats.st_size < size * 1024 ** 2:
                return True
        except EnvironmentError:
            return True

        cmd = 'python -c "' + \
              "import sys;" + \
              "fd = open('{0}', 'rb');" + \
              "fd.seek({1});" + \
              "data = fd.read(1024); " + \
              "sys.stdout.write(data + ' ' * ( 1024 - len(data)))\" | md5sum"

        if self.use_sudo:
            cmd = "sudo " + cmd

        zero_md5 = '0f343b0931126a20f133d67c2b018a3b'
        bsize = size * (1024 ** 2)
        offsets = [random.randrange(bsize - 1024) for _ in range(num_blocks)]
        offsets.append(bsize - 1024)
        offsets.append(0)

        for offset in offsets:
            data = rossh(cmd.format(fname, offset), nolog=True)

            md = ""
            for line in data.split("\n"):
                if "unable to resolve" not in line:
                    md = line.split()[0].strip()
                    break

            if len(md) != 32:
                logger.error("File data check is failed - " + data)
                return True

            if zero_md5 == md:
                return True

        return False

    def prefill_test_files(self, rossh, files, force=False):
        if self.use_system_fio:
            cmd_templ = "fio "
        else:
            cmd_templ = "{0}/fio ".format(self.config.remote_dir)

        if self.use_sudo:
            cmd_templ = "sudo " + cmd_templ

        cmd_templ += "--name=xxx --filename={0} --direct=1" + \
                     " --bs=4m --size={1}m --rw=write"

        ssize = 0

        if force:
            logger.info("File prefilling is forced")

        ddtime = 0
        for fname, curr_sz in files.items():
            if not force:
                if not self.check_prefill_required(rossh, fname, curr_sz):
                    logger.debug("prefill is skipped")
                    continue

            logger.info("Prefilling file {0}".format(fname))
            cmd = cmd_templ.format(fname, curr_sz)
            ssize += curr_sz

            stime = time.time()
            rossh(cmd, timeout=curr_sz)
            ddtime += time.time() - stime

        if ddtime > 1.0:
            fill_bw = int(ssize / ddtime)
            mess = "Initiall fio fill bw is {0} MiBps for this vm"
            logger.info(mess.format(fill_bw))

    def install_utils(self, node, rossh, max_retry=3, timeout=5):
        need_install = []
        packs = [('screen', 'screen')]
        os_info = get_os(rossh)

        if self.use_system_fio:
            packs.append(('fio', 'fio'))
        else:
            packs.append(('bzip2', 'bzip2'))

        for bin_name, package in packs:
            if bin_name is None:
                need_install.append(package)
                continue

            try:
                rossh('which ' + bin_name, nolog=True)
            except OSError:
                need_install.append(package)

        if len(need_install) != 0:
            if 'redhat' == os_info.distro:
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

        if not self.use_system_fio:
            fio_dir = os.path.dirname(os.path.dirname(wally.__file__))
            fio_dir = os.path.join(os.getcwd(), fio_dir)
            fio_dir = os.path.join(fio_dir, 'fio_binaries')
            fname = 'fio_{0.release}_{0.arch}.bz2'.format(os_info)
            fio_path = os.path.join(fio_dir, fname)

            if not os.path.exists(fio_path):
                raise RuntimeError("No prebuild fio available for {0}".format(os_info))

            bz_dest = self.join_remote('fio.bz2')
            with node.connection.open_sftp() as sftp:
                sftp.put(fio_path, bz_dest)

            rossh("bzip2 --decompress " + bz_dest, nolog=True)
            rossh("chmod a+x " + self.join_remote("fio"), nolog=True)

    def pre_run(self):
        if 'FILESIZE' not in self.config_params:
            # need to detect file size
            pass

        self.fio_configs = fio_cfg_compile(self.raw_cfg,
                                           self.config_fname,
                                           self.config_params)
        self.fio_configs = list(self.fio_configs)

        files = {}
        for section in self.fio_configs:
            sz = ssize2b(section.vals['size'])
            msz = sz / (1024 ** 2)

            if sz % (1024 ** 2) != 0:
                msz += 1

            fname = section.vals['filename']

            # if already has other test with the same file name
            # take largest size
            files[fname] = max(files.get(fname, 0), msz)

        with ThreadPoolExecutor(len(self.config.nodes)) as pool:
            fc = functools.partial(self.pre_run_th,
                                   files=files,
                                   force=self.force_prefill)
            list(pool.map(fc, self.config.nodes))

    def pre_run_th(self, node, files, force):
        try:
            # fill files with pseudo-random data
            rossh = run_on_node(node)
            rossh.connection = node.connection

            try:
                cmd = 'mkdir -p "{0}"'.format(self.config.remote_dir)
                if self.use_sudo:
                    cmd = "sudo " + cmd
                    cmd += " ; sudo chown {0} {1}".format(node.get_user(),
                                                          self.config.remote_dir)
                rossh(cmd, nolog=True)

                assert self.config.remote_dir != "" and self.config.remote_dir != "/"
                rossh("rm -rf {0}/*".format(self.config.remote_dir), nolog=True)

            except Exception as exc:
                msg = "Failed to create folder {0} on remote {1}. Error: {2!s}"
                msg = msg.format(self.config.remote_dir, node.get_conn_id(), exc)
                logger.exception(msg)
                raise StopTestError(msg, exc)

            self.install_utils(node, rossh)
            self.prefill_test_files(rossh, files, force)
        except:
            logger.exception("XXXX")
            raise

    def show_test_execution_time(self):
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

    def run(self):
        logger.debug("Run preparation")
        self.pre_run()
        self.show_test_execution_time()

        tname = os.path.basename(self.config_fname)
        if tname.endswith('.cfg'):
            tname = tname[:-4]

        barrier = Barrier(len(self.config.nodes))
        results = []

        # set of Operation_Mode_BlockSize str's
        # which should not be tested anymore, as
        # they already too slow with previous thread count
        lat_bw_limit_reached = set()

        with ThreadPoolExecutor(len(self.config.nodes)) as pool:
            for pos, fio_cfg in enumerate(self.fio_configs):
                test_descr = get_test_summary(fio_cfg.vals).split("th")[0]
                if test_descr in lat_bw_limit_reached:
                    continue
                else:
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
                        if None not in intervals:
                            break
                    except (EnvironmentError, SSHException) as exc:
                        logger.exception("During fio run")
                        if idx == max_retr - 1:
                            raise StopTestError("Fio failed", exc)

                    logger.info("Reconnectiongm, sleeping %ss and retrying", self.retry_time)

                    wait(pool.submit(node.connection.close)
                         for node in self.config.nodes)

                    time.sleep(self.retry_time)

                    wait(pool.submit(reconnect, node.connection, node.conn_url)
                         for node in self.config.nodes)

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

                if self.max_latency is not None:
                    lat_50, _ = res.get_lat_perc_50_95_multy()

                    # conver us to ms
                    if self.max_latency < lat_50:
                        logger.info(("Will skip all subsequent tests of {0} " +
                                     "due to lat/bw limits").format(fio_cfg.name))
                        lat_bw_limit_reached.add(test_descr)

                test_res = res.get_params_from_fio_report()
                if self.min_bw_per_thread is not None:
                    if self.min_bw_per_thread > average(test_res['bw']):
                        lat_bw_limit_reached.add(test_descr)

        return IOTestResults(self.config.params['cfg'],
                             results, self.config.log_directory)

    def do_run(self, node, barrier, fio_cfg, pos, nolog=False):
        if self.use_sudo:
            sudo = "sudo "
        else:
            sudo = ""

        bash_file = """
#!/bin/bash

function get_dev() {{
    if [ -b "$1" ] ; then
        echo $1
    else
        echo $(df "$1" | tail -1 | awk '{{print $1}}')
    fi
}}

function log_io_activiti(){{
    local dest="$1"
    local dev=$(get_dev "$2")
    local sleep_time="$3"
    dev=$(basename "$dev")

    echo $dev

    for (( ; ; )) ; do
        grep -E "\\b$dev\\b" /proc/diskstats >> "$dest"
        sleep $sleep_time
    done
}}

sync
cd {exec_folder}

log_io_activiti {io_log_file} {test_file} 1 &
local pid="$!"

{fio_path}fio --output-format=json --output={out_file} --alloc-size=262144 {job_file} >{err_out_file} 2>&1
echo $? >{res_code_file}
kill -9 $pid

"""

        exec_folder = self.config.remote_dir

        if self.use_system_fio:
            fio_path = ""
        else:
            if not exec_folder.endswith("/"):
                fio_path = exec_folder + "/"
            else:
                fio_path = exec_folder

        bash_file = bash_file.format(out_file=self.results_file,
                                     job_file=self.task_file,
                                     err_out_file=self.err_out_file,
                                     res_code_file=self.exit_code_file,
                                     exec_folder=exec_folder,
                                     fio_path=fio_path,
                                     test_file=self.config_params['FILENAME'],
                                     io_log_file=self.io_log_file).strip()

        with node.connection.open_sftp() as sftp:
            save_to_remote(sftp, self.task_file, str(fio_cfg))
            save_to_remote(sftp, self.sh_file, bash_file)

        exec_time = execution_time(fio_cfg)

        timeout = int(exec_time + max(300, exec_time))
        soft_tout = exec_time

        begin = time.time()

        fnames_before = run_on_node(node)("ls -1 " + exec_folder, nolog=True)

        barrier.wait()

        task = BGSSHTask(node, self.use_sudo)
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
            elif fname == os.path.basename(self.io_log_file):
                files['iops'].append(('sys', fname))
                all_files.append(fname)

        arch_name = self.join_remote('wally_result.tar.gz')
        tmp_dir = os.path.join(self.config.log_directory, 'tmp_' + conn_id)

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        os.mkdir(tmp_dir)
        loc_arch_name = os.path.join(tmp_dir, 'wally_result.{0}.tar.gz'.format(conn_id))
        file_full_names = " ".join(all_files)

        try:
            os.unlink(loc_arch_name)
        except:
            pass

        with node.connection.open_sftp() as sftp:
            try:
                exit_code = read_from_remote(sftp, self.exit_code_file)
            except IOError:
                logger.error("No exit code file found on %s. Looks like process failed to start",
                             conn_id)
                return None

            err_out = read_from_remote(sftp, self.err_out_file)
            exit_code = exit_code.strip()

            if exit_code != '0':
                msg = "fio exit with code {0}: {1}".format(exit_code, err_out)
                logger.critical(msg.strip())
                raise StopTestError("fio failed")

            rossh("rm -f {0}".format(arch_name), nolog=True)
            pack_files_cmd = "cd {0} ; tar zcvf {1} {2}".format(exec_folder, arch_name, file_full_names)
            rossh(pack_files_cmd, nolog=True)
            sftp.get(arch_name, loc_arch_name)

        unpack_files_cmd = "cd {0} ; tar xvzf {1} >/dev/null".format(tmp_dir, loc_arch_name)
        subprocess.check_call(unpack_files_cmd, shell=True)
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

        remove_remote_res_files_cmd = "cd {0} ; rm -f {1} {2}".format(exec_folder,
                                                                      arch_name,
                                                                      file_full_names)
        rossh(remove_remote_res_files_cmd, nolog=True)
        return begin, end

    @classmethod
    def prepare_data(cls, results):
        """
        create a table with io performance report
        for console
        """

        def key_func(data):
            tpl = data.summary_tpl()
            return (data.name,
                    tpl.oper,
                    tpl.mode,
                    ssize2b(tpl.bsize),
                    int(tpl.th_count) * int(tpl.vm_count))
        res = []

        for item in sorted(results, key=key_func):
            test_dinfo = item.disk_perf_info()
            testnodes_count = len(item.config.nodes)

            iops, _ = test_dinfo.iops.rounded_average_conf()

            if test_dinfo.iops_sys is not None:
                iops_sys, iops_sys_conf = test_dinfo.iops_sys.rounded_average_conf()
                _, iops_sys_dev = test_dinfo.iops_sys.rounded_average_dev()
                iops_sys_per_vm = round_3_digit(iops_sys / testnodes_count)
                iops_sys = round_3_digit(iops_sys)
            else:
                iops_sys = None
                iops_sys_per_vm = None
                iops_sys_dev = None
                iops_sys_conf = None

            bw, bw_conf = test_dinfo.bw.rounded_average_conf()
            _, bw_dev = test_dinfo.bw.rounded_average_dev()
            conf_perc = int(round(bw_conf * 100 / bw))
            dev_perc = int(round(bw_dev * 100 / bw))

            lat_50 = round_3_digit(int(test_dinfo.lat_50))
            lat_95 = round_3_digit(int(test_dinfo.lat_95))
            lat_avg = round_3_digit(int(test_dinfo.lat_avg))

            iops_per_vm = round_3_digit(iops / testnodes_count)
            bw_per_vm = round_3_digit(bw / testnodes_count)

            iops = round_3_digit(iops)
            bw = round_3_digit(bw)

            summ = "{0.oper}{0.mode} {0.bsize:>4} {0.th_count:>3}th {0.vm_count:>2}vm".format(item.summary_tpl())

            res.append({"name": key_func(item)[0],
                        "key": key_func(item)[:4],
                        "summ": summ,
                        "iops": int(iops),
                        "bw": int(bw),
                        "conf": str(conf_perc),
                        "dev": str(dev_perc),
                        "iops_per_vm": int(iops_per_vm),
                        "bw_per_vm": int(bw_per_vm),
                        "lat_50": lat_50,
                        "lat_95": lat_95,
                        "lat_avg": lat_avg,

                        "iops_sys": iops_sys,
                        "iops_sys_per_vm": iops_sys_per_vm,
                        "sys_conf": iops_sys_conf,
                        "sys_dev": iops_sys_dev})

        return res

    Field = collections.namedtuple("Field", ("header", "attr", "allign", "size"))
    fiels_and_header = [
        Field("Name",           "name",        "l",  7),
        Field("Description",    "summ",        "l", 19),
        Field("IOPS\ncum",      "iops",        "r",  3),
        # Field("IOPS_sys\ncum",  "iops_sys",    "r",  3),
        Field("KiBps\ncum",     "bw",          "r",  6),
        Field("Cnf %\n95%",     "conf",        "r",  3),
        Field("Dev%",           "dev",         "r",  3),
        Field("iops\n/vm",      "iops_per_vm", "r",  3),
        Field("KiBps\n/vm",     "bw_per_vm",   "r",  6),
        Field("lat ms\nmedian", "lat_50",      "r",  3),
        Field("lat ms\n95%",    "lat_95",      "r",  3),
        Field("lat\navg",       "lat_avg",     "r",  3),
    ]

    fiels_and_header_dct = dict((item.attr, item) for item in fiels_and_header)

    @classmethod
    def format_for_console(cls, results):
        """
        create a table with io performance report
        for console
        """

        tab = texttable.Texttable(max_width=120)
        tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
        tab.set_cols_align([f.allign for f in cls.fiels_and_header])
        sep = ["-" * f.size for f in cls.fiels_and_header]
        tab.header([f.header for f in cls.fiels_and_header])
        prev_k = None
        for item in cls.prepare_data(results):
            if prev_k is not None:
                if prev_k != item["key"]:
                    tab.add_row(sep)

            prev_k = item["key"]
            tab.add_row([item[f.attr] for f in cls.fiels_and_header])

        return tab.draw()

    @classmethod
    def format_diff_for_console(cls, list_of_results):
        """
        create a table with io performance report
        for console
        """

        tab = texttable.Texttable(max_width=200)
        tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)

        header = [
            cls.fiels_and_header_dct["name"].header,
            cls.fiels_and_header_dct["summ"].header,
        ]
        allign = ["l", "l"]

        header.append("IOPS ~ Cnf% ~ Dev%")
        allign.extend(["r"] * len(list_of_results))
        header.extend(
            "IOPS_{0} %".format(i + 2) for i in range(len(list_of_results[1:]))
        )

        header.append("BW")
        allign.extend(["r"] * len(list_of_results))
        header.extend(
            "BW_{0} %".format(i + 2) for i in range(len(list_of_results[1:]))
        )

        header.append("LAT")
        allign.extend(["r"] * len(list_of_results))
        header.extend(
            "LAT_{0}".format(i + 2) for i in range(len(list_of_results[1:]))
        )

        tab.header(header)
        sep = ["-" * 3] * len(header)
        processed_results = map(cls.prepare_data, list_of_results)

        key2results = []
        for res in processed_results:
            key2results.append(dict(
                ((item["name"], item["summ"]), item) for item in res
            ))

        prev_k = None
        iops_frmt = "{0[iops]} ~ {0[conf]:>2} ~ {0[dev]:>2}"
        for item in processed_results[0]:
            if prev_k is not None:
                if prev_k != item["key"]:
                    tab.add_row(sep)

            prev_k = item["key"]

            key = (item['name'], item['summ'])
            line = list(key)
            base = key2results[0][key]

            line.append(iops_frmt.format(base))

            for test_results in key2results[1:]:
                val = test_results.get(key)
                if val is None:
                    line.append("-")
                elif base['iops'] == 0:
                    line.append("Nan")
                else:
                    prc_val = {'dev': val['dev'], 'conf': val['conf']}
                    prc_val['iops'] = int(100 * val['iops'] / base['iops'])
                    line.append(iops_frmt.format(prc_val))

            line.append(base['bw'])

            for test_results in key2results[1:]:
                val = test_results.get(key)
                if val is None:
                    line.append("-")
                elif base['bw'] == 0:
                    line.append("Nan")
                else:
                    line.append(int(100 * val['bw'] / base['bw']))

            for test_results in key2results:
                val = test_results.get(key)
                if val is None:
                    line.append("-")
                else:
                    line.append("{0[lat_50]} - {0[lat_95]}".format(val))

            tab.add_row(line)

        tab.set_cols_align(allign)
        return tab.draw()
