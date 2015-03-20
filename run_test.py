import os
import sys
import json
import time
import pprint
import logging
import os.path
import argparse
from nodes import discover

import ssh_runner
import io_scenario
from config import cfg_dict
from utils import log_error
from rest_api import add_test
from itest import IOPerfTest,PgBenchTest
from formatters import get_formatter

logger = logging.getLogger("io-perf-tool")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
formatter = logging.Formatter(log_format,
                              "%H:%M:%S")
ch.setFormatter(formatter)


tool_type_mapper = {
    "iozone": IOPerfTest,
    "fio": IOPerfTest,
    "pgbench": PgBenchTest,
}


def run_io_test(tool,
                script_args,
                test_runner,
                keep_temp_files=False):

    files_dir = os.path.dirname(io_scenario.__file__)

    path = 'iozone' if 'iozone' == tool else 'fio'
    src_testtool_path = os.path.join(files_dir, path)

    obj_cls = tool_type_mapper[tool]
    obj = obj_cls(script_args,
                  src_testtool_path,
                  None,
                  keep_temp_files)

    return test_runner(obj)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run disk io performance test")

    parser.add_argument("tool_type", help="test tool type",
                        choices=['iozone', 'fio', 'pgbench', 'two_scripts'])

    parser.add_argument("-l", dest='extra_logs',
                        action='store_true', default=False,
                        help="print some extra log info")

    parser.add_argument("-o", "--test-opts", dest='opts',
                        help="cmd line options for test")

    parser.add_argument("-f", "--test-opts-file", dest='opts_file',
                        type=argparse.FileType('r'), default=None,
                        help="file with cmd line options for test")

    parser.add_argument("--max-preparation-time", default=300,
                        type=int, dest="max_preparation_time")

    parser.add_argument("-b", "--build-info", default=None,
                        dest="build_name")

    parser.add_argument("-d", "--data-server-url", default=None,
                        dest="data_server_url")

    parser.add_argument("-n", "--lab-name", default=None,
                        dest="lab_name")

    parser.add_argument("--create-vms-opts", default=None,
                        help="Creating vm's before run ssh runner",
                        dest="create_vms_opts")

    parser.add_argument("-k", "--keep", default=False,
                        help="keep temporary files",
                        dest="keep_temp_files", action='store_true')

    choices = ["local", "ssh"]

    parser.add_argument("--runner", required=True,
                        choices=choices, help="runner type")

    parser.add_argument("--runner-extra-opts", default=None,
                        dest="runner_opts", help="runner extra options")

    return parser.parse_args(argv)


def format_result(res, formatter):
    data = "\n{0}\n".format("=" * 80)
    data += pprint.pformat(res) + "\n"
    data += "{0}\n".format("=" * 80)
    templ = "{0}\n\n====> {1}\n\n{2}\n\n"
    return templ.format(data, formatter(res), "=" * 80)


def deploy_and_start_sensors(sensors_conf, nodes):
    pass


def main(argv):
    logging_conf = cfg_dict.get('logging')
    if logging_conf:
        if logging_conf.get('extra_logs'):
            logger.setLevel(logging.DEBUG)
            ch.setLevel(logging.DEBUG)

    # Discover nodes
    nodes_to_run = discover.discover(cfg_dict.get('cluster'))

    tests = cfg_dict.get("tests", [])

    # Deploy and start sensors
    deploy_and_start_sensors(cfg_dict.get('sensors'), nodes_to_run)

    for test_name, opts in tests.items():
        cmd_line = " ".join(opts['opts'])
        logger.debug("Run test with {0!r} params".format(cmd_line))
        latest_start_time = 300 + time.time()
        uris = [node.connection_url for node in nodes_to_run]
        runner = ssh_runner.get_ssh_runner(uris,
                                           latest_start_time,
                                           opts.get('keep_temp_files'))
        res = run_io_test(test_name,
                          opts['opts'],
                          runner,
                          opts.get('keep_temp_files'))
        logger.debug(format_result(res, get_formatter(test_name)))

    if cfg_dict.get('data_server_url'):
        result = json.loads(get_formatter(opts.tool_type)(res))
        result['name'] = opts.build_name
        add_test(opts.build_name, result, opts.data_server_url)

    return 0


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
