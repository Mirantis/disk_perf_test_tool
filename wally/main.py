import os
import sys
import time
import signal
import logging
import argparse
import functools
from typing import List, Tuple, Any, Callable, IO, cast, Optional
from yaml import load as _yaml_load


YLoader = Callable[[IO], Any]
yaml_load = None  # type: YLoader


try:
    from yaml import CLoader
    yaml_load = cast(YLoader,  functools.partial(_yaml_load, Loader=CLoader))
except ImportError:
    yaml_load = cast(YLoader,  _yaml_load)


import texttable

try:
    import faulthandler
except ImportError:
    faulthandler = None


from . import utils, run_test, pretty_yaml
from .storage import make_storage, Storage
from .config import Config
from .logger import setup_loggers
from .stage import log_stage, StageType
from .test_run_class import TestRun


logger = logging.getLogger("wally")


def list_results(path: str) -> List[Tuple[str, str, str, str]]:
    results = []  # type: List[Tuple[float, str, str, str, str]]

    for dir_name in os.listdir(path):
        full_path = os.path.join(path, dir_name)

        try:
            stor = make_storage(full_path, existing=True)
        except Exception as exc:
            logger.warning("Can't load folder {}. Error {}".format(full_path, exc))

        comment = cast(str, stor['info/comment'])
        run_uuid = cast(str, stor['info/run_uuid'])
        run_time = cast(float, stor['info/run_time'])
        test_types = ""
        results.append((run_time,
                        run_uuid,
                        test_types,
                        time.ctime(run_time),
                        '-' if comment is None else comment))

    results.sort()
    return [i[1:] for i in results]


def log_nodes_statistic_stage(ctx: TestRun) -> None:
    utils.log_nodes_statistic(ctx.nodes)


def parse_args(argv):
    descr = "Disk io performance test suite"
    parser = argparse.ArgumentParser(prog='wally', description=descr)
    parser.add_argument("-l", '--log-level', help="print some extra log info")
    parser.add_argument("-s", '--settings-dir', default=None,
                        help="Folder to store key/settings/history files")

    subparsers = parser.add_subparsers(dest='subparser_name')

    # ---------------------------------------------------------------------
    report_parser = subparsers.add_parser('ls', help='list all results')
    report_parser.add_argument("result_storage", help="Folder with test results")

    # ---------------------------------------------------------------------
    compare_help = 'compare two results'
    report_parser = subparsers.add_parser('compare', help=compare_help)
    report_parser.add_argument("data_path1", help="First folder with test results")
    report_parser.add_argument("data_path2", help="Second folder with test results")

    # ---------------------------------------------------------------------
    report_help = 'run report on previously obtained results'
    report_parser = subparsers.add_parser('report', help=report_help)
    report_parser.add_argument("data_dir", help="folder with rest results")

    # ---------------------------------------------------------------------
    test_parser = subparsers.add_parser('test', help='run tests')
    test_parser.add_argument('--build-description', type=str, default="Build info")
    test_parser.add_argument('--build-id', type=str, default="id")
    test_parser.add_argument('--build-type', type=str, default="GA")
    test_parser.add_argument('-n', '--no-tests', action='store_true', help="Don't run tests")
    test_parser.add_argument('--load-report', action='store_true')
    test_parser.add_argument("-k", '--keep-vm', action='store_true', help="Don't remove test vm's")
    test_parser.add_argument("-d", '--dont-discover-nodes', action='store_true',
                             help="Don't connect/discover fuel nodes")
    test_parser.add_argument('--no-report', action='store_true', help="Skip report stages")
    test_parser.add_argument('-r', '--resume', default=None, help="Resume previously stopped test, stored in DIR",
                             metavar="DIR")
    test_parser.add_argument('--result-dir', default=None, help="Save results to DIR", metavart="DIR")
    test_parser.add_argument("comment", help="Test information")
    test_parser.add_argument("config_file", help="Yaml config file", nargs='?', default=None)

    # ---------------------------------------------------------------------

    return parser.parse_args(argv[1:])


def get_config_path(config: Config, opts_value: Optional[str]) -> str:
    if opts_value is None and 'settings_dir' not in config:
        val = "~/.wally"
    elif opts_value is not None:
        val = opts_value
    else:
        val = config.settings_dir

    return os.path.abspath(os.path.expanduser(val))


def main(argv: List[str]) -> int:
    if faulthandler is not None:
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    opts = parse_args(argv)

    stages = []  # type: List[StageType]
    report_stages = []  # type: List[StageType]

    # stop mypy from telling that config & storage might be undeclared
    config = None  # type: Config
    storage = None  # type: Storage

    if opts.subparser_name == 'test':
        if opts.resume:
            storage = make_storage(opts.resume, existing=True)
            config = storage.load('config', Config)
        else:
            file_name = os.path.abspath(opts.config_file)
            with open(file_name) as fd:
                config = Config(yaml_load(fd.read()))  # type: ignore

            config.storage_url, config.run_uuid = utils.get_uniq_path_uuid(config.results_dir)
            config.comment = opts.comment
            config.keep_vm = opts.keep_vm
            config.no_tests = opts.no_tests
            config.dont_discover_nodes = opts.dont_discover_nodes
            config.build_id = opts.build_id
            config.build_description = opts.build_description
            config.build_type = opts.build_type
            config.settings_dir = get_config_path(config, opts.settings_dir)

            storage = make_storage(config.storage_url)

            storage['config'] = config  # type: ignore

        stages.extend([
            run_test.clouds_connect_stage,
            run_test.discover_stage,
            run_test.reuse_vms_stage,
            log_nodes_statistic_stage,
            run_test.save_nodes_stage,
            run_test.connect_stage])

        if config.get("collect_info", True):
            stages.append(run_test.collect_info_stage)

        stages.extend([
            run_test.run_tests_stage,
        ])

    elif opts.subparser_name == 'ls':
        tab = texttable.Texttable(max_width=200)
        tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
        tab.set_cols_align(["l", "l", "l", "l"])
        tab.header(["Name", "Tests", "Run at", "Comment"])
        tab.add_rows(list_results(opts.result_storage))
        print(tab.draw())
        return 0

    elif opts.subparser_name == 'report':
        storage = make_storage(opts.data_dir, existing=True)
        config.settings_dir = get_config_path(config, opts.settings_dir)

    elif opts.subparser_name == 'compare':
        # x = run_test.load_data_from_path(opts.data_path1)
        # y = run_test.load_data_from_path(opts.data_path2)
        # print(run_test.IOPerfTest.format_diff_for_console(
        #     [x['io'][0], y['io'][0]]))
        return 0

    if not getattr(opts, "no_report", False):
        report_stages.append(run_test.console_report_stage)
        report_stages.append(run_test.html_report_stage)

    # log level is not a part of config
    if opts.log_level is not None:
        str_level = opts.log_level
    else:
        str_level = config.get('logging/log_level', 'INFO')

    setup_loggers(getattr(logging, str_level), log_fd=storage.get_stream('log'))
    logger.info("All info would be stored into %r", config.storage_url)

    ctx = TestRun(config, storage)

    for stage in stages:
        ok = False
        with log_stage(stage):
            stage(ctx)
            ok = True
        if not ok:
            break

    exc, cls, tb = sys.exc_info()
    for stage in ctx.clear_calls_stack[::-1]:
        with log_stage(stage):
            stage(ctx)

    logger.debug("Start utils.cleanup")
    for clean_func, args, kwargs in utils.iter_clean_func():
        with log_stage(clean_func):
            clean_func(*args, **kwargs)

    if exc is None:
        for report_stage in report_stages:
            with log_stage(report_stage):
                report_stage(ctx)

    logger.info("All info is stored into %r", config.storage_url)

    if exc is None:
        logger.info("Tests finished successfully")
        return 0
    else:
        logger.error("Tests are failed. See error details in log above")
        return 1
