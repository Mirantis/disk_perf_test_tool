import os.path
import logging
from typing import cast

import wally

from ...utils import StopTestError, get_os, ssize2b
from ...node_interfaces import IRPCNode
from ..itest import ThreadedTest, IterationConfig, RunTestRes
from .fio_task_parser import execution_time, fio_cfg_compile, FioJobSection, FioParams, get_log_files
from . import rpc_plugin

logger = logging.getLogger("wally")


class IOPerfTest(ThreadedTest):
    soft_runcycle = 5 * 60
    retry_time = 30
    configs_dir = os.path.dirname(__file__)  # type: str

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
        plugin_code = open(rpc_plugin.__file__.rsplit(".", 1)[0] + ".py", "rb").read()
        node.upload_plugin(code=plugin_code, name="fio")

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

        fio_config = "\n".join(map(str, self.iterations_configs))
        node.put_to_file(self.remote_task_file, fio_config.encode("utf8"))

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

    def do_test(self, node: IRPCNode, iter_config: IterationConfig) -> RunTestRes:
        exec_time = execution_time(cast(FioJobSection, iter_config))
        fio_cmd_templ = "cd {exec_folder}; " + \
                        "{fio_path} --output-format=json --output={out_file} --alloc-size=262144 {job_file}"

        bw_log, iops_log, lat_hist_log = get_log_files(iter_config)

        cmd = fio_cmd_templ.format(exec_folder=self.exec_folder,
                                   fio_path=self.fio_path,
                                   out_file=self.remote_output_file,
                                   job_file=self.remote_task_file)
        raw_res = node.run(cmd, timeout=exec_time + max(300, exec_time))
        
        return

        # TODO(koder): fix next error
        # raise NotImplementedError("Need to extract time from test result")
        # return raw_res, (0, 0)

