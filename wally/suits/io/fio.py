import array
import os.path
import logging
from typing import cast, Any

import wally

from ...utils import StopTestError, get_os, ssize2b
from ...node_interfaces import IRPCNode
from ..itest import ThreadedTest, IterationConfig, NodeTestResults
from ...result_classes import TimeSerie
from .fio_task_parser import execution_time, fio_cfg_compile, FioJobSection, FioParams, get_log_files, get_test_summary
from . import rpc_plugin
from .fio_hist import expected_lat_bins


logger = logging.getLogger("wally")


class IOPerfTest(ThreadedTest):
    soft_runcycle = 5 * 60
    retry_time = 30
    configs_dir = os.path.dirname(__file__)  # type: str
    name = 'fio'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        get = self.config.params.get

        self.remote_task_file = self.join_remote("task.fio")
        self.remote_output_file = self.join_remote("fio_result.json")
        self.use_system_fio = get('use_system_fio', False)  # type: bool
        self.use_sudo = get("use_sudo", True)  # type: bool
        self.force_prefill = get('force_prefill', False)  # type: bool

        self.load_profile_name = self.config.params['load']  # type: str
        self.name = "io." + self.load_profile_name

        if os.path.isfile(self.load_profile_name):
            self.load_profile_path = self.load_profile_name   # type: str
        else:
            self.load_profile_path = os.path.join(self.configs_dir, self.load_profile_name+ '.cfg')

        self.load_profile = open(self.load_profile_path, 'rt').read()  # type: str

        if self.use_system_fio:
            self.fio_path = "fio"    # type: str
        else:
            self.fio_path = os.path.join(self.config.remote_dir, "fio")

        self.load_params = self.config.params['params']
        self.file_name = self.load_params['FILENAME']

        if 'FILESIZE' not in self.load_params:
            logger.debug("Getting test file sizes on all nodes")
            try:
                sizes = {node.conn.fs.file_stat(self.file_name)['size']
                         for node in self.config.nodes}
            except Exception:
                logger.exception("FILESIZE is not set in config file and fail to detect it." +
                                 "Set FILESIZE or fix error and rerun test")
                raise StopTestError()

            if len(sizes) != 1:
                logger.error("IO target file %r has different sizes on test nodes - %r",
                             self.file_name, sizes)
                raise StopTestError()

            self.file_size = list(sizes)[0]
            logger.info("Detected test file size is %s", self.file_size)
            self.load_params['FILESIZE'] = self.file_size
        else:
            self.file_size = ssize2b(self.load_params['FILESIZE'])

        self.fio_configs = list(fio_cfg_compile(self.load_profile, self.load_profile_path,
                                                cast(FioParams, self.load_params)))

        if len(self.fio_configs) == 0:
            logger.error("Empty fio config provided")
            raise StopTestError()

        self.iterations_configs = self.fio_configs  # type: ignore
        self.exec_folder = self.config.remote_dir

    def config_node(self, node: IRPCNode) -> None:
        plugin_code = open(rpc_plugin.__file__.rsplit(".", 1)[0] + ".py", "rb").read()  # type: bytes
        node.upload_plugin("fio", plugin_code)

        try:
            node.conn.fs.rmtree(self.config.remote_dir)
        except Exception:
            pass

        try:
            node.conn.fs.makedirs(self.config.remote_dir)
        except Exception:
            msg = "Failed to recreate folder {} on remote {}.".format(self.config.remote_dir, node)
            logger.exception(msg)
            raise StopTestError()

        self.install_utils(node)

        mb = int(self.file_size / 1024 ** 2)
        logger.info("Filling test file %s with %sMiB of random data", self.file_name, mb)
        fill_bw = node.conn.fio.fill_file(self.file_name, mb, force=self.force_prefill, fio_path=self.fio_path)
        if fill_bw is not None:
            logger.info("Initial fio fill bw is {} MiBps for {}".format(fill_bw, node))

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
            node.copy_file(fio_path, bz_dest)
            node.run("bzip2 --decompress {} ; chmod a+x {}".format(bz_dest, self.join_remote("fio")))

    def get_expected_runtime(self, iteration_info: IterationConfig) -> int:
        return execution_time(cast(FioJobSection, iteration_info))

    def prepare_iteration(self, node: IRPCNode, iter_config: IterationConfig) -> None:
        node.put_to_file(self.remote_task_file, str(iter_config).encode("utf8"))

    # TODO: get a link to substorage as a parameter
    def run_iteration(self, node: IRPCNode, iter_config: IterationConfig, stor_prefix: str) -> NodeTestResults:
        f_iter_config = cast(FioJobSection, iter_config)
        exec_time = execution_time(f_iter_config)

        fio_cmd_templ = "cd {exec_folder}; " + \
                        "{fio_path} --output-format=json --output={out_file} --alloc-size=262144 {job_file}"

        cmd = fio_cmd_templ.format(exec_folder=self.exec_folder,
                                   fio_path=self.fio_path,
                                   out_file=self.remote_output_file,
                                   job_file=self.remote_task_file)
        must_be_empty = node.run(cmd, timeout=exec_time + max(300, exec_time), check_timeout=1).strip()

        if must_be_empty:
            logger.error("Unexpected fio output: %r", must_be_empty)

        res = NodeTestResults(self.__class__.__name__, node.info.node_id(), get_test_summary(f_iter_config))

        res.extra_logs['fio'] = node.get_file_content(self.remote_output_file)
        self.store_data(res.extra_logs['fio'], "raw", stor_prefix, "fio_raw")
        node.conn.fs.unlink(self.remote_output_file)

        files = [name for name in node.conn.fs.listdir(self.exec_folder)]

        expected_time_delta = 1000  # 1000ms == 1s
        max_time_diff = 50  # 50ms - 5%

        for name, path in get_log_files(f_iter_config):
            log_files = [fname for fname in files if fname.startswith(path)]
            if len(log_files) != 1:
                logger.error("Found %s files, match log pattern %s(%s) - %s",
                             len(log_files), path, name, ",".join(log_files[10:]))
                raise StopTestError()

            fname = os.path.join(self.exec_folder, log_files[0])
            raw_result = node.get_file_content(fname)  # type: bytes
            self.store_data(raw_result, "raw", stor_prefix, "{}_raw".format(name))
            node.conn.fs.unlink(fname)

            try:
                log_data = raw_result.decode("utf8").split("\n")
            except UnicodeEncodeError:
                logger.exception("Error during parse %s fio log file - can't decode usint UTF8", name)
                raise StopTestError()

            parsed = array.array('L' if name == 'lat' else 'Q')
            prev_ts = None
            load_start_at = None

            # TODO: need to adjust vals for timedelta
            for idx, line in enumerate(log_data):
                line = line.strip()
                if line:
                    try:
                        time_ms_s, val_s, _, *rest = line.split(",")
                        time_ms = int(time_ms_s.strip())
                        if prev_ts and abs(time_ms - prev_ts - expected_time_delta) > max_time_diff:
                            logger.warning("Too large gap in {} log at {} - {}ms"
                                           .format(time_ms, name, time_ms - prev_ts))
                        else:
                            prev_ts = time_ms - expected_time_delta
                            load_start_at = time_ms
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
                    prev_ts += expected_time_delta

            res.series[name] = TimeSerie(name=name,
                                         raw=raw_result,
                                         second_axis_size=expected_lat_bins if name == 'lat' else 1,
                                         start_at=load_start_at,
                                         step=expected_time_delta,
                                         data=parsed)

            self.store_data(parsed, "array", stor_prefix, "{}_data".format(name))
            self.store_data(res.series[name].meta(), "yaml", stor_prefix, "{}_meta".format(name))

        return res

    def format_for_console(self, data: Any) -> str:
        raise NotImplementedError()
