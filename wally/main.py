import os
import sys
import time
import signal
import logging
import argparse
import functools
import contextlib

from yaml import load as _yaml_load

try:
    from yaml import CLoader
    yaml_load = functools.partial(_yaml_load, Loader=CLoader)
except ImportError:
    yaml_load = _yaml_load


import texttable

try:
    import faulthandler
except ImportError:
    faulthandler = None


from wally.timeseries import SensorDatastore
from wally import utils, run_test, pretty_yaml
from wally.config import (load_config, setup_loggers,
                          get_test_files, save_run_params, load_run_params)


logger = logging.getLogger("wally")


class Context(object):
    def __init__(self):
        self.build_meta = {}
        self.nodes = []
        self.clear_calls_stack = []
        self.openstack_nodes_ids = []
        self.sensors_mon_q = None
        self.hw_info = []
        self.fuel_openstack_creds = None


def get_stage_name(func):
    nm = get_func_name(func)
    if nm.endswith("stage"):
        return nm
    else:
        return nm + " stage"


def get_test_names(raw_res):
    res = []
    for tp, data in raw_res:
        if not isinstance(data, list):
            raise ValueError()

        keys = []
        for dt in data:
            if not isinstance(dt, dict):
                raise ValueError()

            keys.append(",".join(dt.keys()))

        res.append(tp + "(" + ",".join(keys) + ")")
    return res


def list_results(path):
    results = []

    for dname in os.listdir(path):
        try:
            files_cfg = get_test_files(os.path.join(path, dname))

            if not os.path.isfile(files_cfg['raw_results']):
                continue

            mt = os.path.getmtime(files_cfg['raw_results'])
            res_mtime = time.ctime(mt)

            raw_res = yaml_load(open(files_cfg['raw_results']).read())
            test_names = ",".join(sorted(get_test_names(raw_res)))

            params = load_run_params(files_cfg['run_params_file'])

            comm = params.get('comment')
            results.append((mt, dname, test_names, res_mtime,
                           '-' if comm is None else comm))
        except ValueError:
            pass

    tab = texttable.Texttable(max_width=200)
    tab.set_deco(tab.HEADER | tab.VLINES | tab.BORDER)
    tab.set_cols_align(["l", "l", "l", "l"])
    results.sort()

    for data in results[::-1]:
        tab.add_row(data[1:])

    tab.header(["Name", "Tests", "etime", "Comment"])

    print(tab.draw())


def get_func_name(obj):
    if hasattr(obj, '__name__'):
        return obj.__name__
    if hasattr(obj, 'func_name'):
        return obj.func_name
    return obj.func.func_name


@contextlib.contextmanager
def log_stage(func):
    msg_templ = "Exception during {0}: {1!s}"
    msg_templ_no_exc = "During {0}"

    logger.info("Start " + get_stage_name(func))

    try:
        yield
    except utils.StopTestError as exc:
        logger.error(msg_templ.format(
            get_func_name(func), exc))
    except Exception:
        logger.exception(msg_templ_no_exc.format(
            get_func_name(func)))


def make_storage_dir_struct(cfg):
    utils.mkdirs_if_unxists(cfg.results_dir)
    utils.mkdirs_if_unxists(cfg.sensor_storage)
    utils.mkdirs_if_unxists(cfg.hwinfo_directory)
    utils.mkdirs_if_unxists(cfg.results_storage)


def log_nodes_statistic_stage(_, ctx):
    utils.log_nodes_statistic(ctx.nodes)


def parse_args(argv):
    descr = "Disk io performance test suite"
    parser = argparse.ArgumentParser(prog='wally', description=descr)
    parser.add_argument("-l", '--log-level',
                        help="print some extra log info")

    subparsers = parser.add_subparsers(dest='subparser_name')

    # ---------------------------------------------------------------------
    compare_help = 'list all results'
    report_parser = subparsers.add_parser('ls', help=compare_help)
    report_parser.add_argument("result_storage", help="Folder with test results")

    # ---------------------------------------------------------------------
    compare_help = 'compare two results'
    report_parser = subparsers.add_parser('compare', help=compare_help)
    report_parser.add_argument("data_path1", help="First folder with test results")
    report_parser.add_argument("data_path2", help="Second folder with test results")

    # ---------------------------------------------------------------------
    report_help = 'run report on previously obtained results'
    report_parser = subparsers.add_parser('report', help=report_help)
    report_parser.add_argument('--load_report',  action='store_true')
    report_parser.add_argument("data_dir", help="folder with rest results")

    # ---------------------------------------------------------------------
    test_parser = subparsers.add_parser('test', help='run tests')
    test_parser.add_argument('--build-description',
                             type=str, default="Build info")
    test_parser.add_argument('--build-id', type=str, default="id")
    test_parser.add_argument('--build-type', type=str, default="GA")
    test_parser.add_argument('-n', '--no-tests', action='store_true',
                             help="Don't run tests", default=False)
    test_parser.add_argument('--load_report',  action='store_true')
    test_parser.add_argument("-k", '--keep-vm', action='store_true',
                             help="Don't remove test vm's", default=False)
    test_parser.add_argument("-d", '--dont-discover-nodes', action='store_true',
                             help="Don't connect/discover fuel nodes",
                             default=False)
    test_parser.add_argument('--no-report', action='store_true',
                             help="Skip report stages", default=False)
    test_parser.add_argument("comment", help="Test information")
    test_parser.add_argument("config_file", help="Yaml config file")

    # ---------------------------------------------------------------------

    return parser.parse_args(argv[1:])


def main(argv):
    if faulthandler is not None:
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    opts = parse_args(argv)
    stages = []
    report_stages = []

    ctx = Context()
    ctx.results = {}
    ctx.sensors_data = SensorDatastore()

    if opts.subparser_name == 'test':
        cfg = load_config(opts.config_file)
        make_storage_dir_struct(cfg)
        cfg.comment = opts.comment
        save_run_params(cfg)

        with open(cfg.saved_config_file, 'w') as fd:
            fd.write(pretty_yaml.dumps(cfg.__dict__))

        stages = [
            run_test.discover_stage
        ]

        stages.extend([
            run_test.reuse_vms_stage,
            log_nodes_statistic_stage,
            run_test.save_nodes_stage,
            run_test.connect_stage])

        if cfg.settings.get('collect_info', True):
            stages.append(run_test.collect_hw_info_stage)

        stages.extend([
            # deploy_sensors_stage,
            run_test.run_tests_stage,
            run_test.store_raw_results_stage,
            # gather_sensors_stage
        ])

        cfg.keep_vm = opts.keep_vm
        cfg.no_tests = opts.no_tests
        cfg.dont_discover_nodes = opts.dont_discover_nodes

        ctx.build_meta['build_id'] = opts.build_id
        ctx.build_meta['build_descrption'] = opts.build_description
        ctx.build_meta['build_type'] = opts.build_type

    elif opts.subparser_name == 'ls':
        list_results(opts.result_storage)
        return 0

    elif opts.subparser_name == 'report':
        cfg = load_config(get_test_files(opts.data_dir)['saved_config_file'])
        stages.append(run_test.load_data_from(opts.data_dir))
        opts.no_report = False
        # load build meta

    elif opts.subparser_name == 'compare':
        x = run_test.load_data_from_path(opts.data_path1)
        y = run_test.load_data_from_path(opts.data_path2)
        print(run_test.IOPerfTest.format_diff_for_console(
            [x['io'][0], y['io'][0]]))
        return 0

    if not opts.no_report:
        report_stages.append(run_test.console_report_stage)
        if opts.load_report:
            report_stages.append(run_test.test_load_report_stage)
        report_stages.append(run_test.html_report_stage)

    if opts.log_level is not None:
        str_level = opts.log_level
    else:
        str_level = cfg.settings.get('log_level', 'INFO')

    setup_loggers(getattr(logging, str_level), cfg.log_file)
    logger.info("All info would be stored into " + cfg.results_dir)

    for stage in stages:
        ok = False
        with log_stage(stage):
            stage(cfg, ctx)
            ok = True
        if not ok:
            break

    exc, cls, tb = sys.exc_info()
    for stage in ctx.clear_calls_stack[::-1]:
        with log_stage(stage):
            stage(cfg, ctx)

    logger.debug("Start utils.cleanup")
    for clean_func, args, kwargs in utils.iter_clean_func():
        with log_stage(clean_func):
            clean_func(*args, **kwargs)

    if exc is None:
        for report_stage in report_stages:
            with log_stage(report_stage):
                report_stage(cfg, ctx)

    logger.info("All info stored into " + cfg.results_dir)

    if exc is None:
        logger.info("Tests finished successfully")
        return 0
    else:
        logger.error("Tests are failed. See detailed error above")
        return 1
