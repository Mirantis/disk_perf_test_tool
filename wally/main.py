import os
import time
import signal
import pprint
import getpass
import logging
import argparse
import functools
import contextlib
from typing import List, Tuple, Any, Callable, IO, cast, Optional, Iterator
from yaml import load as _yaml_load

YLoader = Callable[[IO], Any]
yaml_load = None  # type: YLoader

try:
    from yaml import CLoader
    yaml_load = cast(YLoader,  functools.partial(_yaml_load, Loader=CLoader))
except ImportError:
    yaml_load = cast(YLoader,  _yaml_load)

from cephlib.texttable import Texttable
from cephlib.istorage import IStorage

try:
    import faulthandler
except ImportError:
    faulthandler = None

from cephlib.common import setup_logging
from cephlib.storage import make_storage
from cephlib.wally_storage import WallyDB
from cephlib.ssh import set_ssh_key_passwd
from cephlib.node import log_nodes_statistic
from cephlib.node_impl import get_rpc_server_code

from . import utils, report_profiles, report
from .config import Config
from .stage import Stage
from .test_run_class import TestRun
from .result_storage import WallyStorage

# stages
from .ceph import DiscoverCephStage, CollectCephInfoStage
from .openstack import DiscoverOSStage
from .fuel import DiscoverFuelStage
from .run_test import (CollectInfoStage, ExplicitNodesStage, SaveNodesStage,
                       RunTestsStage, ConnectStage, SleepStage, PrepareNodes,
                       LoadStoredNodesStage)

from .report import HtmlReportStage
from .sensors import StartSensorsStage, CollectSensorsStage
from .console_report import ConsoleReportStage


logger = logging.getLogger("wally")


@contextlib.contextmanager
def log_stage(stage: Stage, cleanup: bool = False) -> Iterator[None]:
    logger.info("Start " + stage.name() + ("::cleanup" if cleanup else ""))
    try:
        yield
    except utils.StopTestError as exc:
        raise
    except Exception:
        logger.exception("During %s", stage.name() + ("::cleanup" if cleanup else ""))
        raise


def list_results(path: str, limit: int = None) -> List[Tuple[str, str, str, str, str]]:
    dirs = []
    for dir_name in os.listdir(path):
        full_path = os.path.join(path, dir_name)
        dirs.append((os.stat(full_path).st_ctime, full_path))

    dirs.sort()
    results = []  # type: List[Tuple[str, str, str, str, str]]
    for _, full_path in dirs[::-1]:
        try:
            stor = make_storage(full_path, existing=True)
        except Exception as exc:
            logger.warning("Can't load folder {}. Error {}".format(full_path, exc))

        try:
            try:
                cfg = stor.load(Config, WallyDB.config)
            except KeyError:
                cfg = stor.load(Config, "config")
        except Exception as exc:
            print("Fail to load {}. {}".format(os.path.basename(full_path), exc))
            continue

        if WallyDB.run_interval in stor:
            run_time = stor.get(WallyDB.run_interval)[0]
        else:
            run_time = os.stat(full_path).st_ctime

        ftime = time.strftime("%d %b %H:%M", time.localtime(run_time))

        test_types = []
        for suite_cfg in cfg.get('tests', []):
            for suite_name, params in suite_cfg.items():
                if suite_name == 'fio':
                    test_types.append("{}.{}".format(suite_name, params['load']))
                else:
                    test_types.append(suite_name)
        results.append((cfg.run_uuid,
                        ",".join(test_types),
                        ftime,
                        '-' if cfg.comment is None else cfg.comment,
                        '-'))

        if limit and len(results) >= limit:
            break

    return results


def log_nodes_statistic_stage(ctx: TestRun) -> None:
    log_nodes_statistic(ctx.nodes)


def parse_args(argv):
    descr = "Disk io performance test suite"
    parser = argparse.ArgumentParser(prog='wally', description=descr)
    parser.add_argument("-l", '--log-level', help="print some extra log info")
    parser.add_argument("--ssh-key-passwd", default=None, help="Pass ssh key password")
    parser.add_argument("--ssh-key-passwd-kbd", action="store_true", help="Enter ssh key password interactively")
    parser.add_argument("--profile", action="store_true", help="Profile execution")
    parser.add_argument("-s", '--settings-dir', default=None,
                        help="Folder to store key/settings/history files")

    subparsers = parser.add_subparsers(dest='subparser_name')

    # ---------------------------------------------------------------------
    report_parser = subparsers.add_parser('ls', help='list all results')
    report_parser.add_argument("-l", "--limit", metavar='LIMIT', help="Show only LIMIT last results",
                               default=None, type=int)
    report_parser.add_argument("result_storage", help="Folder with test results")

    # ---------------------------------------------------------------------
    compare_help = 'compare two results'
    report_parser = subparsers.add_parser('compare', help=compare_help)
    report_parser.add_argument("data_path1", help="First folder with test results")
    report_parser.add_argument("data_path2", help="Second folder with test results")

    # ---------------------------------------------------------------------
    report_help = 'run report on previously obtained results'
    report_parser = subparsers.add_parser('report', help=report_help)
    report_parser.add_argument('-R', '--reporters', help="Comma-separated list of reportes - html,txt",
                               default='html,txt')
    report_parser.add_argument('-f', '--format', help="Images format, default is " + report_profiles.default_format,
                               choices=('svg', 'png'), default=report_profiles.default_format)
    report_parser.add_argument("data_dir", help="folder with rest results")

    # ---------------------------------------------------------------------
    ipython_help = 'run ipython in prepared environment'
    ipython_parser = subparsers.add_parser('ipython', help=ipython_help)
    ipython_parser.add_argument("storage_dir", help="Storage path")
    # ---------------------------------------------------------------------
    jupyter_help = 'run ipython in prepared environment'
    jupyter_parser = subparsers.add_parser('jupyter', help=jupyter_help)
    jupyter_parser.add_argument("storage_dir", help="Storage path")

    # ---------------------------------------------------------------------
    test_parser = subparsers.add_parser('test', help='run tests')
    test_parser.add_argument("-d", '--dont-discover-nodes', action='store_true', help="Don't discover nodes")
    test_parser.add_argument('-D', '--dont-collect', action='store_true', help="Don't collect cluster info")
    test_parser.add_argument("-k", '--keep-vm', action='store_true', help="Don't remove test vm's")
    test_parser.add_argument('-L', '--load-report', action='store_true', help="Create cluster load report")
    test_parser.add_argument('-n', '--no-tests', action='store_true', help="Don't run tests")
    test_parser.add_argument('-N', '--no-report', action='store_true', help="Skip report stages")
    test_parser.add_argument('-r', '--result-dir', default=None, help="Save results to DIR", metavar="DIR")
    test_parser.add_argument('-R', '--reporters', help="Comma-separated list of reportes - html,txt",
                             default='html,txt')
    test_parser.add_argument('--build-description', type=str, default="Build info")
    test_parser.add_argument('--build-id', type=str, default="id")
    test_parser.add_argument('--build-type', type=str, default="GA")
    test_parser.add_argument("comment", help="Test information")
    test_parser.add_argument("config_file", help="Yaml config file")

    # ---------------------------------------------------------------------
    test_parser = subparsers.add_parser('resume', help='resume tests')
    test_parser.add_argument("storage_dir", help="Path to test directory")

    # ---------------------------------------------------------------------
    test_parser = subparsers.add_parser('db', help='Exec command on DB')
    test_parser.add_argument("cmd", choices=("show",), help="Command to execute")
    test_parser.add_argument("params", nargs='*', help="Command params")
    test_parser.add_argument("storage_dir", help="Storage path")

    return parser.parse_args(argv[1:])


def get_config_path(config: Config, opts_value: Optional[str]) -> str:
    if opts_value is None and 'settings_dir' not in config:
        val = "~/.wally"
    elif opts_value is not None:
        val = opts_value
    else:
        val = config.settings_dir

    return os.path.abspath(os.path.expanduser(val))


def find_cfg_file(name: str, included_from: str = None) -> str:
    paths = [".", os.path.expanduser('~/.wally')]
    if included_from is not None:
        paths.append(os.path.dirname(included_from))

    search_paths = set(os.path.abspath(path) for path in paths if os.path.isdir(path))

    for folder in search_paths:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(name)


def load_config(path: str) -> Config:
    path = os.path.abspath(path)
    cfg_dict = yaml_load(open(path).read())

    while 'include' in cfg_dict:
        inc = cfg_dict.pop('include')
        if isinstance(inc, str):
            inc = [inc]

        for fname in inc:
            inc_path = find_cfg_file(fname, path)
            inc_dict = yaml_load(open(inc_path).read())
            inc_dict.update(cfg_dict)
            cfg_dict = inc_dict

    return Config(cfg_dict)


def get_run_stages() -> List[Stage]:
    return [DiscoverCephStage(),
            CollectCephInfoStage(),
            DiscoverOSStage(),
            DiscoverFuelStage(),
            ExplicitNodesStage(),
            StartSensorsStage(),
            RunTestsStage(),
            CollectSensorsStage(),
            ConnectStage(),
            SleepStage(),
            PrepareNodes()]


def main(argv: List[str]) -> int:
    if faulthandler is not None:
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    opts = parse_args(argv)
    stages = []  # type: List[Stage]

    # stop mypy from telling that config & storage might be undeclared
    config = None  # type: Config
    storage = None  # type: IStorage

    if opts.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
    else:
        pr = None

    if opts.subparser_name == 'test':
        config = load_config(opts.config_file)
        config.storage_url, config.run_uuid = utils.get_uniq_path_uuid(config.results_storage)
        config.comment = opts.comment
        config.keep_vm = opts.keep_vm
        config.no_tests = opts.no_tests
        config.dont_discover_nodes = opts.dont_discover_nodes
        config.build_id = opts.build_id
        config.build_description = opts.build_description
        config.build_type = opts.build_type
        config.settings_dir = get_config_path(config, opts.settings_dir)
        config.discover = set(name for name in config.get('discover', '').split(",") if name)

        storage = make_storage(config.storage_url)
        storage.put(config, WallyDB.config)

        stages.extend(get_run_stages())
        stages.append(SaveNodesStage())

        if not opts.dont_collect:
            stages.append(CollectInfoStage())

        argv2 = argv[:]
        if '--ssh-key-passwd' in argv2:
            # don't save ssh key password to storage
            argv2[argv2.index("--ssh-key-passwd") + 1] = "<removed from output>"
        storage.put(argv2, 'cli')

    elif opts.subparser_name == 'resume':
        opts.resumed = True
        storage = make_storage(opts.storage_dir, existing=True)
        config = storage.load(Config, WallyDB.config)
        stages.extend(get_run_stages())
        stages.append(LoadStoredNodesStage())
        prev_opts = storage.get('cli')  # type: List[str]

        if '--ssh-key-passwd' in prev_opts and opts.ssh_key_passwd:
            prev_opts[prev_opts.index("--ssh-key-passwd") + 1] = opts.ssh_key_passwd

        restored_opts = parse_args(prev_opts)
        opts.__dict__.update(restored_opts.__dict__)
        opts.subparser_name = 'resume'

    elif opts.subparser_name == 'ls':
        tab = Texttable(max_width=200)
        tab.set_cols_align(["l", "l", "l", "l", 'c'])
        tab.set_deco(Texttable.VLINES | Texttable.BORDER | Texttable.HEADER)
        tab.header(["Name", "Tests", "Started at", "Comment", "Result"])
        tab.add_rows(list_results(opts.result_storage, opts.limit), header=False)
        print(tab.draw())
        return 0

    elif opts.subparser_name == 'report':
        if getattr(opts, "no_report", False):
            print(" --no-report option can't be used with 'report' cmd")
            return 1
        storage = make_storage(opts.data_dir, existing=True)
        config = storage.load(Config, WallyDB.config)
        report_profiles.default_format = opts.format
        report.default_format = opts.format
        stages.append(LoadStoredNodesStage())
        stages.append(SaveNodesStage())
    elif opts.subparser_name == 'compare':
        # x = run_test.load_data_from_path(opts.data_path1)
        # y = run_test.load_data_from_path(opts.data_path2)
        # print(run_test.IOPerfTest.format_diff_for_console(
        #     [x['io'][0], y['io'][0]]))
        return 0

    elif opts.subparser_name == 'db':
        storage = make_storage(opts.storage_dir, existing=True)
        if opts.cmd == 'show':
            if len(opts.params) != 1:
                print("'show' command requires parameter - key to show")
                return 1
            pprint.pprint(storage.get(opts.params[0]))
        else:
            print("Unknown/not_implemented command {!r}".format(opts.cmd))
            return 1
        return 0
    elif opts.subparser_name == 'ipython':
        storage = make_storage(opts.storage_dir, existing=True)
        rstorage = WallyStorage(storage=storage)

        import IPython
        IPython.embed()

        return 0
    else:
        print("Subparser {!r} is not supported".format(opts.subparser_name))
        return 1

    start_time = int(time.time())

    report_stages = []  # type: List[Stage]
    if not getattr(opts, "no_report", False):
        reporters = opts.reporters.split(",")
        assert len(set(reporters)) == len(reporters)
        assert set(reporters).issubset({'txt', 'html'})
        if 'txt' in reporters:
            report_stages.append(ConsoleReportStage())
        if 'html' in reporters:
            report_stages.append(HtmlReportStage())

    log_config_obj = config.raw().get('logging')
    assert isinstance(log_config_obj, dict) or log_config_obj is None, "Broken 'logging' option in config"
    setup_logging(log_config_obj=log_config_obj, log_level=opts.log_level, log_file=storage.get_fname('log'))

    logger.info("All info would be stored into %r", config.storage_url)

    ctx = TestRun(config, storage, WallyStorage(storage))
    ctx.rpc_code, ctx.default_rpc_plugins = get_rpc_server_code()

    if 'dev_roles' in ctx.config:
        ctx.devs_locator = ctx.config.dev_roles

    if opts.ssh_key_passwd is not None:
        set_ssh_key_passwd(opts.ssh_key_passwd)
    elif opts.ssh_key_passwd_kbd:
        set_ssh_key_passwd(getpass.getpass("Ssh key password: ").strip())

    stages.sort(key=lambda x: x.priority)

    # TODO: run only stages, which have config
    failed = False
    cleanup_stages = []

    for stage in stages:
        if stage.config_block is not None:
            if stage.config_block not in ctx.config:
                logger.debug("Skip stage %r, as config has no required block %r", stage.name(), stage.config_block)
                continue

        cleanup_stages.append(stage)
        try:
            with log_stage(stage):
                stage.run(ctx)
        except (Exception, KeyboardInterrupt):
            failed = True
            break
        ctx.storage.sync()
    ctx.storage.sync()

    logger.debug("Start cleanup")
    cleanup_failed = False
    for stage in cleanup_stages[::-1]:
        try:
            with log_stage(stage, cleanup=True):
                stage.cleanup(ctx)
        except:
            cleanup_failed = True
        ctx.storage.sync()

    if not failed:
        for report_stage in report_stages:
            with log_stage(report_stage):
                try:
                    report_stage.run(ctx)
                except utils.StopTestError:
                    logger.error("Report stage %s requested stop execution", report_stage.name())
                    failed = True
                    break

    ctx.storage.sync()

    logger.info("All info is stored into %r", config.storage_url)
    end_time = int(time.time())
    storage.put([start_time, end_time], WallyDB.run_interval)

    if failed or cleanup_failed:
        if opts.subparser_name == 'report':
            logger.error("Report generation failed. See error details in log above")
        else:
            logger.error("Tests are failed. See error details in log above")
        code = 1
    else:
        if opts.subparser_name == 'report':
            logger.info("Report successfully generated")
        else:
            logger.info("Tests finished successfully")
        code = 0

    if opts.profile:
        assert pr is not None
        pr.disable()
        import pstats
        pstats.Stats(pr).sort_stats('tottime').print_stats(30)

    if opts.subparser_name == 'test':
        storage.put(code, WallyDB.res_code)

    storage.sync()
    return code
