import array
import os.path
import logging
from typing import cast, Any, Tuple, List

import wally

from ...utils import StopTestError, ssize2b, b2ssize
from ...node_interfaces import IRPCNode
from ...node_utils import get_os
from ..itest import ThreadedTest
from ...result_classes import TimeSeries, DataSource
from ..job import JobConfig
from .fio_task_parser import execution_time, fio_cfg_compile, FioJobConfig, FioParams, get_log_files
from . import rpc_plugin
from .fio_hist import expected_lat_bins


logger = logging.getLogger("wally")


class FioTest(ThreadedTest):
    soft_runcycle = 5 * 60
    retry_time = 30
    configs_dir = os.path.dirname(__file__)  # type: str
    name = 'fio'
    job_config_cls = FioJobConfig

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        get = self.suite.params.get

        self.remote_task_file = self.join_remote("task.fio")
        self.remote_output_file = self.join_remote("fio_result.json")
        self.use_system_fio = get('use_system_fio', False)  # type: bool
        self.use_sudo = get("use_sudo", True)  # type: bool
        self.force_prefill = get('force_prefill', False)  # type: bool

        self.load_profile_name = self.suite.params['load']  # type: str

        if os.path.isfile(self.load_profile_name):
            self.load_profile_path = self.load_profile_name   # type: str
        else:
            self.load_profile_path = os.path.join(self.configs_dir, self.load_profile_name+ '.cfg')

        self.load_profile = open(self.load_profile_path, 'rt').read()  # type: str

        if self.use_system_fio:
            self.fio_path = "fio"    # type: str
        else:
            self.fio_path = os.path.join(self.suite.remote_dir, "fio")

        self.load_params = self.suite.params['params']
        self.file_name = self.load_params['FILENAME']

        if 'FILESIZE' not in self.load_params:
            logger.debug("Getting test file sizes on all nodes")
            try:
                sizes = {node.conn.fs.file_stat(self.file_name)[b'size']
                         for node in self.suite.nodes}
            except Exception:
                logger.exception("FILESIZE is not set in config file and fail to detect it." +
                                 "Set FILESIZE or fix error and rerun test")
                raise StopTestError()

            if len(sizes) != 1:
                logger.error("IO target file %r has different sizes on test nodes - %r",
                             self.file_name, sizes)
                raise StopTestError()

            self.file_size = list(sizes)[0]
            logger.info("Detected test file size is %sB", b2ssize(self.file_size))
            self.load_params['FILESIZE'] = self.file_size
        else:
            self.file_size = ssize2b(self.load_params['FILESIZE'])

        self.job_configs = list(fio_cfg_compile(self.load_profile, self.load_profile_path,
                                                cast(FioParams, self.load_params)))

        if len(self.job_configs) == 0:
            logger.error("Empty fio config provided")
            raise StopTestError()

        self.exec_folder = self.suite.remote_dir

    def config_node(self, node: IRPCNode) -> None:
        plugin_code = open(rpc_plugin.__file__.rsplit(".", 1)[0] + ".py", "rb").read()  # type: bytes
        node.upload_plugin("fio", plugin_code)

        try:
            node.conn.fs.rmtree(self.suite.remote_dir)
        except Exception:
            pass

        try:
            node.conn.fs.makedirs(self.suite.remote_dir)
        except Exception:
            msg = "Failed to recreate folder {} on remote {}.".format(self.suite.remote_dir, node)
            logger.exception(msg)
            raise StopTestError()

        # TODO: check this during config validation
        if self.file_size % (4 * (1024 ** 2)) != 0:
            logger.error("Test file size must be proportional to 4MiB")
            raise StopTestError()

        self.install_utils(node)

        mb = int(self.file_size / 1024 ** 2)
        logger.info("Filling test file %s on node %s with %sMiB of random data", self.file_name, node.info, mb)
        is_prefilled, fill_bw = node.conn.fio.fill_file(self.file_name, mb,
                                                        force=self.force_prefill,
                                                        fio_path=self.fio_path)

        if not is_prefilled:
            logger.info("Test file on node %s is already prefilled", node.info)
        elif fill_bw is not None:
            logger.info("Initial fio fill bw is %s MiBps for %s", fill_bw, node.info)

    def install_utils(self, node: IRPCNode) -> None:
        os_info = get_os(node)
        if self.use_system_fio:
            if os_info.distro != 'ubuntu':
                logger.error("Only ubuntu supported on test VM")
                raise StopTestError()
            node.conn.fio.install('fio', binary='fio')
        else:
            node.conn.fio.install('bzip2', binary='bzip2')
            fio_dir = os.path.dirname(os.path.dirname(wally.__file__))  # type: str
            fio_dir = os.path.join(os.getcwd(), fio_dir)
            fio_dir = os.path.join(fio_dir, 'fio_binaries')
            fname = 'fio_{0.release}_{0.arch}.bz2'.format(os_info)
            fio_path = os.path.join(fio_dir, fname)  # type: str

            if not os.path.exists(fio_path):
                logger.error("No prebuild fio binary available for {0}".format(os_info))
                raise StopTestError()

            bz_dest = self.join_remote('fio.bz2')  # type: str
            node.copy_file(fio_path, bz_dest, compress=False)
            node.run("bzip2 --decompress {} ; chmod a+x {}".format(bz_dest, self.join_remote("fio")))

    def get_expected_runtime(self, job_config: JobConfig) -> int:
        return execution_time(cast(FioJobConfig, job_config))

    def prepare_iteration(self, node: IRPCNode, job: JobConfig) -> None:
        node.put_to_file(self.remote_task_file, str(job).encode("utf8"))

    # TODO: get a link to substorage as a parameter
    def run_iteration(self, node: IRPCNode, job: JobConfig) -> List[TimeSeries]:
        exec_time = execution_time(cast(FioJobConfig, job))


        fio_cmd_templ = "cd {exec_folder}; " + \
                        "{fio_path} --output-format=json --output={out_file} --alloc-size=262144 {job_file}"

        cmd = fio_cmd_templ.format(exec_folder=self.exec_folder,
                                   fio_path=self.fio_path,
                                   out_file=self.remote_output_file,
                                   job_file=self.remote_task_file)
        must_be_empty = node.run(cmd, timeout=exec_time + max(300, exec_time), check_timeout=1).strip()

        if must_be_empty:
            logger.error("Unexpected fio output: %r", must_be_empty)

        # put fio output into storage
        fio_out = node.get_file_content(self.remote_output_file)

        path = DataSource(suite_id=self.suite.storage_id,
                          job_id=job.storage_id,
                          node_id=node.node_id,
                          dev='fio',
                          sensor='stdout',
                          tag='json')

        self.storage.put_extra(fio_out, path)
        node.conn.fs.unlink(self.remote_output_file)

        files = [name for name in node.conn.fs.listdir(self.exec_folder)]
        result = []
        for name, file_path in get_log_files(cast(FioJobConfig, job)):
            log_files = [fname for fname in files if fname.startswith(file_path)]
            if len(log_files) != 1:
                logger.error("Found %s files, match log pattern %s(%s) - %s",
                             len(log_files), file_path, name, ",".join(log_files[10:]))
                raise StopTestError()

            fname = os.path.join(self.exec_folder, log_files[0])
            raw_result = node.get_file_content(fname)  # type: bytes
            node.conn.fs.unlink(fname)

            try:
                log_data = raw_result.decode("utf8").split("\n")
            except UnicodeEncodeError:
                logger.exception("Error during parse %s fio log file - can't decode usint UTF8", name)
                raise StopTestError()

            parsed = array.array('L' if name == 'lat' else 'Q')
            times = array.array('Q')

            for idx, line in enumerate(log_data):
                line = line.strip()
                if line:
                    try:
                        time_ms_s, val_s, _, *rest = line.split(",")
                        times.append(int(time_ms_s.strip()))

                        if name == 'lat':
                            vals = [int(i.strip()) for i in rest]

                            if len(vals) != expected_lat_bins:
                                logger.error("Expect {} bins in latency histogram, but found {} at time {}"
                                             .format(expected_lat_bins, len(vals), time_ms_s))
                                raise StopTestError()

                            parsed.extend(vals)
                        else:
                            parsed.append(int(val_s.strip()))
                    except ValueError:
                        logger.exception("Error during parse %s fio log file in line %s: %r", name, idx, line)
                        raise StopTestError()

            result.append(TimeSeries(name=name,
                                     raw=raw_result,
                                     second_axis_size=expected_lat_bins if name == 'lat' else 1,
                                     data=parsed,
                                     times=times,
                                     source=path(sensor=name, tag=None)))
        return result

    def format_for_console(self, data: Any) -> str:
        raise NotImplementedError()
