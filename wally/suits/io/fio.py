import os.path
import logging
from typing import Dict, List, Union, cast

import wally

from ...utils import ssize2b, StopTestError, get_os
from ...node_interfaces import IRPCNode
from ..itest import ThreadedTest, IterationConfig, RunTestRes
from .fio_task_parser import execution_time, fio_cfg_compile, FioJobSection, FioParams


logger = logging.getLogger("wally")


class IOPerfTest(ThreadedTest):
    soft_runcycle = 5 * 60
    retry_time = 30
    configs_dir = os.path.dirname(__file__)  # type: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        get = self.config.params.get

        self.load_profile_name = self.config.params['load']  # type: str
        self.name = "io." + self.load_profile_name

        if os.path.isfile(self.load_profile_name):
            self.load_profile_path = os.path.join(self.configs_dir, self.load_profile_name+ '.cfg')  # type: str
        else:
            self.load_profile_path = self.load_profile_name

        self.load_profile = open(self.load_profile_path, 'rt').read()  # type: str

        self.use_system_fio = get('use_system_fio', False)  # type: bool

        if self.use_system_fio:
            self.fio_path = "fio"    # type: str
        else:
            self.fio_path = os.path.join(self.config.remote_dir, "fio")

        self.force_prefill = get('force_prefill', False)  # type: bool

        if 'FILESIZE' not in self.config.params:
            raise NotImplementedError("File size detection is not implemented")

        # self.max_latency = get("max_lat")  # type: Optional[int]
        # self.min_bw_per_thread = get("min_bw")   # type: Optional[int]

        self.use_sudo = get("use_sudo", True)  # type: bool

        self.fio_configs = list(fio_cfg_compile(self.load_profile,
                                                self.load_profile_path,
                                                cast(FioParams, self.config.params)))

        if len(self.fio_configs) == 0:
            logger.exception("Empty fio config provided")
            raise StopTestError("Empty fio config provided")

        self.iterations_configs = self.fio_configs  # type: ignore
        self.files_sizes = self.get_file_sizes()

        self.exec_folder = self.config.remote_dir
        self.fio_path = "" if self.use_system_fio else self.exec_folder

    def get_file_sizes(self) -> Dict[str, int]:
        files_sizes = {}  # type: Dict[str, int]

        for section in self.fio_configs:
            sz = ssize2b(section.vals['size'])
            msz = sz // (1024 ** 2) + (1 if sz % (1024 ** 2) != 0 else 0)
            fname = section.vals['filename']  # type: str

            # if already has other test with the same file name
            # take largest size
            files_sizes[fname] = max(files_sizes.get(fname, 0), msz)

        return files_sizes

    def config_node(self, node: IRPCNode) -> None:
        try:
            node.conn.rmdir(self.config.remote_dir, recursive=True, ignore_missing=True)
            node.conn.mkdir(self.config.remote_dir)
        except Exception as exc:
            msg = "Failed to create folder {} on remote {}.".format(self.config.remote_dir, node, exc)
            logger.exception(msg)
            raise StopTestError(msg) from exc

        self.install_utils(node)
        logger.info("Prefilling test files with random data")
        fill_bw = node.conn.prefill_test_files(self.files_sizes, force=self.force_prefill, fio_path=self.fio_path)
        if fill_bw is not None:
            logger.info("Initial fio fill bw is {} MiBps for {}".format(fill_bw, node.info.node_id()))

    def install_utils(self, node: IRPCNode) -> None:
        if self.use_system_fio:
            node.conn.install('fio', binary='fio')

        if not self.use_system_fio:
            os_info = get_os(node)
            fio_dir = os.path.dirname(os.path.dirname(wally.__file__))  # type: str
            fio_dir = os.path.join(os.getcwd(), fio_dir)
            fio_dir = os.path.join(fio_dir, 'fio_binaries')
            fname = 'fio_{0.release}_{0.arch}.bz2'.format(os_info)
            fio_path = os.path.join(fio_dir, fname)  # type: str

            if not os.path.exists(fio_path):
                raise RuntimeError("No prebuild fio binary available for {0}".format(os_info))

            bz_dest = self.join_remote('fio.bz2')  # type: str
            node.copy_file(fio_path, bz_dest)
            node.run("bzip2 --decompress {}" + bz_dest)
            node.run("chmod a+x " + self.join_remote("fio"))

    def get_expected_runtime(self, iteration_info: IterationConfig) -> int:
        return execution_time(cast(FioJobSection, iteration_info))

    def do_test(self, node: IRPCNode, iter_config: IterationConfig) -> RunTestRes:
        exec_time = execution_time(cast(FioJobSection, iter_config))
        raw_res = node.conn.fio.run_fio(self.fio_path,
                                        self.exec_folder,
                                        str(cast(FioJobSection, iter_config)),
                                        exec_time + max(300, exec_time))
        # TODO(koder): fix next error
        raise NotImplementedError("Need to extract time from test result")
        return raw_res, (0, 0)

