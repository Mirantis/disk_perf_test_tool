from __future__ import print_function

import os
import sys
import time
import Queue
import pprint
import logging
import argparse
import functools
import threading
import subprocess
import contextlib
import collections

import yaml
from concurrent.futures import ThreadPoolExecutor

from wally import pretty_yaml
from wally.hw_info import get_hw_info
from wally.discover import discover, Node
from wally.timeseries import SensorDatastore
from wally import utils, report, ssh_utils, start_vms
from wally.suits.itest import IOPerfTest, PgBenchTest, MysqlTest
from wally.sensors_utils import deploy_sensors_stage
from wally.config import cfg_dict, load_config, setup_loggers


try:
    from wally import webui
except ImportError:
    webui = None


logger = logging.getLogger("wally")


def format_result(res, formatter):
    data = "\n{0}\n".format("=" * 80)
    data += pprint.pformat(res) + "\n"
    data += "{0}\n".format("=" * 80)
    templ = "{0}\n\n====> {1}\n\n{2}\n\n"
    return templ.format(data, formatter(res), "=" * 80)


class Context(object):
    def __init__(self):
        self.build_meta = {}
        self.nodes = []
        self.clear_calls_stack = []
        self.openstack_nodes_ids = []
        self.sensors_mon_q = None
        self.hw_info = []


def connect_one(node, vm=False):
    if node.conn_url == 'local':
        node.connection = ssh_utils.connect(node.conn_url)
        return

    try:
        ssh_pref = "ssh://"
        if node.conn_url.startswith(ssh_pref):
            url = node.conn_url[len(ssh_pref):]

            if vm:
                conn_timeout = 240
            else:
                conn_timeout = 30

            node.connection = ssh_utils.connect(url,
                                                conn_timeout=conn_timeout)
        else:
            raise ValueError("Unknown url type {0}".format(node.conn_url))
    except Exception as exc:
        # logger.exception("During connect to " + node.get_conn_id())
        msg = "During connect to {0}: {1!s}".format(node.get_conn_id(),
                                                    exc)
        logger.error(msg)
        node.connection = None


def connect_all(nodes, vm=False):
    logger.info("Connecting to nodes")
    with ThreadPoolExecutor(32) as pool:
        connect_one_f = functools.partial(connect_one, vm=vm)
        list(pool.map(connect_one_f, nodes))


def collect_hw_info_stage(cfg, ctx):
    if os.path.exists(cfg['hwreport_fname']):
        msg = "{0} already exists. Skip hw info"
        logger.info(msg.format(cfg['hwreport_fname']))
        return

    with ThreadPoolExecutor(32) as pool:
        connections = (node.connection for node in ctx.nodes)
        ctx.hw_info.extend(pool.map(get_hw_info, connections))

    with open(cfg['hwreport_fname'], 'w') as hwfd:
        for node, info in zip(ctx.nodes, ctx.hw_info):
            hwfd.write("-" * 60 + "\n")
            hwfd.write("Roles : " + ", ".join(node.roles) + "\n")
            hwfd.write(str(info) + "\n")
            hwfd.write("-" * 60 + "\n\n")

            if info.hostname is not None:
                fname = os.path.join(
                    cfg_dict['hwinfo_directory'],
                    info.hostname + "_lshw.xml")

                with open(fname, "w") as fd:
                    fd.write(info.raw)
    logger.info("Hardware report stored in " + cfg['hwreport_fname'])
    logger.debug("Raw hardware info in " + cfg['hwinfo_directory'] + " folder")


def test_thread(test, node, barrier, res_q):
    exc = None
    try:
        logger.debug("Run preparation for {0}".format(node.get_conn_id()))
        test.pre_run()
        logger.debug("Run test for {0}".format(node.get_conn_id()))
        test.run(barrier)
    except utils.StopTestError as exc:
        pass
    except Exception as exc:
        msg = "In test {0} for node {1}"
        msg = msg.format(test, node.get_conn_id())
        logger.exception(msg)
        exc = utils.StopTestError(msg, exc)

    try:
        test.cleanup()
    except utils.StopTestError as exc1:
        if exc is None:
            exc = exc1
    except:
        msg = "Duringf cleanup - in test {0} for node {1}"
        logger.exception(msg.format(test, node))

    if exc is not None:
        res_q.put(exc)


def run_tests(cfg, test_block, nodes):
    tool_type_mapper = {
        "io": IOPerfTest,
        "pgbench": PgBenchTest,
        "mysql": MysqlTest,
    }

    test_nodes = [node for node in nodes
                  if 'testnode' in node.roles]
    if len(test_nodes) == 0:
        logger.error("No test nodes found")
        return

    test_number_per_type = {}
    res_q = Queue.Queue()

    for name, params in test_block.items():
        logger.info("Starting {0} tests".format(name))
        test_num = test_number_per_type.get(name, 0)
        test_number_per_type[name] = test_num + 1
        threads = []
        barrier = utils.Barrier(len(test_nodes))
        coord_q = Queue.Queue()
        test_cls = tool_type_mapper[name]
        rem_folder = cfg['default_test_local_folder'].format(name=name)

        for idx, node in enumerate(test_nodes):
            msg = "Starting {0} test on {1} node"
            logger.debug(msg.format(name, node.conn_url))

            dr = os.path.join(
                    cfg_dict['test_log_directory'],
                    "{0}_{1}_{2}".format(name, test_num, node.get_ip())
                )

            if not os.path.exists(dr):
                os.makedirs(dr)

            test = test_cls(options=params,
                            is_primary=(idx == 0),
                            on_result_cb=res_q.put,
                            test_uuid=cfg['run_uuid'],
                            node=node,
                            remote_dir=rem_folder,
                            log_directory=dr,
                            coordination_queue=coord_q)
            th = threading.Thread(None, test_thread, None,
                                  (test, node, barrier, res_q))
            threads.append(th)
            th.daemon = True
            th.start()

        th = threading.Thread(None, test_cls.coordination_th, None,
                              (coord_q, barrier, len(threads)))
        threads.append(th)
        th.daemon = True
        th.start()

        def gather_results(res_q, results):
            while not res_q.empty():
                val = res_q.get()

                if isinstance(val, utils.StopTestError):
                    raise val

                if isinstance(val, Exception):
                    msg = "Exception during test execution: {0!s}"
                    raise ValueError(msg.format(val))

                results.append(val)

        results = []

        # MAX_WAIT_TIME = 10
        # end_time = time.time() + MAX_WAIT_TIME

        # while time.time() < end_time:
        while True:
            for th in threads:
                th.join(1)
                gather_results(res_q, results)
                # if time.time() > end_time:
                #     break

            if all(not th.is_alive() for th in threads):
                break

        # if any(th.is_alive() for th in threads):
        #     logger.warning("Some test threads still running")

        gather_results(res_q, results)
        result = test_cls.merge_results(results)
        result['__test_meta__'] = {'testnodes_count': len(test_nodes)}
        yield name, result


def log_nodes_statistic(_, ctx):
    nodes = ctx.nodes
    logger.info("Found {0} nodes total".format(len(nodes)))
    per_role = collections.defaultdict(lambda: 0)
    for node in nodes:
        for role in node.roles:
            per_role[role] += 1

    for role, count in sorted(per_role.items()):
        logger.debug("Found {0} nodes with role {1}".format(count, role))


def connect_stage(cfg, ctx):
    ctx.clear_calls_stack.append(disconnect_stage)
    connect_all(ctx.nodes)

    all_ok = True

    for node in ctx.nodes:
        if node.connection is None:
            if 'testnode' in node.roles:
                msg = "Can't connect to testnode {0}"
                msg = msg.format(node.get_conn_id())
                logger.error(msg)
                raise utils.StopTestError(msg)
            else:
                msg = "Node {0} would be excluded - can't connect"
                logger.warning(msg.format(node.get_conn_id()))
                all_ok = False

    if all_ok:
        logger.info("All nodes connected successfully")

    ctx.nodes = [node for node in ctx.nodes
                 if node.connection is not None]


def discover_stage(cfg, ctx):
    if cfg.get('discover') is not None:
        discover_objs = [i.strip() for i in cfg['discover'].strip().split(",")]

        nodes = discover(ctx,
                         discover_objs,
                         cfg['clouds'],
                         cfg['var_dir'],
                         not cfg['dont_discover_nodes'])

        ctx.nodes.extend(nodes)

    for url, roles in cfg.get('explicit_nodes', {}).items():
        ctx.nodes.append(Node(url, roles.split(",")))


def save_nodes_stage(cfg, ctx):
    cluster = {}
    for node in ctx.nodes:
        roles = node.roles[:]
        if 'testnode' in roles:
            roles.remove('testnode')

        if len(roles) != 0:
            cluster[node.conn_url] = roles

    with open(cfg['nodes_report_file'], "w") as fd:
        fd.write(pretty_yaml.dumps(cluster))


def reuse_vms_stage(vm_name_pattern, conn_pattern):
    def reuse_vms(cfg, ctx):
        try:
            msg = "Looking for vm with name like {0}".format(vm_name_pattern)
            logger.debug(msg)

            os_creds = get_OS_credentials(cfg, ctx, "clouds")
            conn = start_vms.nova_connect(**os_creds)
            for ip, vm_id in start_vms.find_vms(conn, vm_name_pattern):
                node = Node(conn_pattern.format(ip=ip), ['testnode'])
                node.os_vm_id = vm_id
                ctx.nodes.append(node)
        except utils.StopTestError:
            raise
        except Exception as exc:
            msg = "Vm like {0} lookup failed".format(vm_name_pattern)
            logger.exception(msg)
            raise utils.StopTestError(msg, exc)

    return reuse_vms


def get_OS_credentials(cfg, ctx, creds_type):
    creds = None

    if creds_type == 'clouds':
        logger.info("Using OS credentials from 'cloud' section")
        if 'openstack' in cfg['clouds']:
            os_cfg = cfg['clouds']['openstack']

            tenant = os_cfg['OS_TENANT_NAME'].strip()
            user = os_cfg['OS_USERNAME'].strip()
            passwd = os_cfg['OS_PASSWORD'].strip()
            auth_url = os_cfg['OS_AUTH_URL'].strip()

        elif 'fuel' in cfg['clouds'] and \
             'openstack_env' in cfg['clouds']['fuel']:
            creds = ctx.fuel_openstack_creds

    elif creds_type == 'ENV':
        logger.info("Using OS credentials from shell environment")
        user, passwd, tenant, auth_url = start_vms.ostack_get_creds()
    elif os.path.isfile(creds_type):
        logger.info("Using OS credentials from " + creds_type)
        fc = open(creds_type).read()

        echo = 'echo "$OS_TENANT_NAME:$OS_USERNAME:$OS_PASSWORD@$OS_AUTH_URL"'

        try:
            data = utils.run_locally(['/bin/bash'], input=fc + "\n" + echo)
        except subprocess.CalledProcessError as exc:
            msg = "Failed to get creads from openrc file: " + data
            logger.exception(msg)
            raise utils.StopTestError(msg, exc)

        try:
            user, tenant, passwd_auth_url = data.split(':', 2)
            passwd, auth_url = passwd_auth_url.rsplit("@", 1)
            assert (auth_url.startswith("https://") or
                    auth_url.startswith("http://"))
        except Exception as exc:
            msg = "Failed to get creads from openrc file: " + data
            logger.exception(msg)
            raise utils.StopTestError(msg, exc)

    else:
        msg = "Creds {0!r} isn't supported".format(creds_type)
        logger.error(msg)
        raise utils.StopTestError(msg, None)

    if creds is None:
        creds = {'name': user,
                 'passwd': passwd,
                 'tenant': tenant,
                 'auth_url': auth_url}

    msg = "OS_CREDS: user={name} tenant={tenant} auth_url={auth_url}"
    logger.debug(msg.format(**creds))
    return creds


@contextlib.contextmanager
def create_vms_ctx(ctx, cfg, config, already_has_count=0):
    params = cfg['vm_configs'][config['cfg_name']].copy()
    os_nodes_ids = []

    os_creds_type = config['creds']
    os_creds = get_OS_credentials(cfg, ctx, os_creds_type)
    start_vms.nova_connect(**os_creds)

    logger.info("Preparing openstack")
    params.update(config)
    params['keypair_file_private'] = params['keypair_name'] + ".pem"
    params['group_name'] = cfg_dict['run_uuid']

    if not config.get('skip_preparation', False):
        start_vms.prepare_os_subpr(params=params, **os_creds)

    new_nodes = []
    try:
        for new_node, node_id in start_vms.launch_vms(params,
                                                      already_has_count):
            new_node.roles.append('testnode')
            ctx.nodes.append(new_node)
            os_nodes_ids.append(node_id)
            new_nodes.append(new_node)

        store_nodes_in_log(cfg, os_nodes_ids)
        ctx.openstack_nodes_ids = os_nodes_ids

        yield new_nodes

    finally:
        if not cfg['keep_vm']:
            shut_down_vms_stage(cfg, ctx)


def run_tests_stage(cfg, ctx):
    ctx.results = []

    if 'tests' not in cfg:
        return

    for group in cfg['tests']:

        assert len(group.items()) == 1
        key, config = group.items()[0]

        if 'start_test_nodes' == key:
            if 'openstack' not in config:
                msg = "No openstack block in config - can't spawn vm's"
                logger.error(msg)
                raise utils.StopTestError(msg)

            num_test_nodes = sum(1 for node in ctx.nodes
                                 if 'testnode' in node.roles)

            vm_ctx = create_vms_ctx(ctx, cfg, config['openstack'],
                                    num_test_nodes)
            with vm_ctx as new_nodes:
                connect_all(new_nodes, True)

                for node in new_nodes:
                    if node.connection is None:
                        msg = "Failed to connect to vm {0}"
                        raise RuntimeError(msg.format(node.get_conn_id()))

                deploy_sensors_stage(cfg_dict,
                                     ctx,
                                     nodes=new_nodes,
                                     undeploy=False)

                if not cfg['no_tests']:
                    for test_group in config.get('tests', []):
                        test_res = run_tests(cfg, test_group, ctx.nodes)
                        ctx.results.extend(test_res)
        else:
            if not cfg['no_tests']:
                test_res = run_tests(cfg, group, ctx.nodes)
                ctx.results.extend(test_res)


def shut_down_vms_stage(cfg, ctx):
    vm_ids_fname = cfg_dict['vm_ids_fname']
    if ctx.openstack_nodes_ids is None:
        nodes_ids = open(vm_ids_fname).read().split()
    else:
        nodes_ids = ctx.openstack_nodes_ids

    if len(nodes_ids) != 0:
        logger.info("Removing nodes")
        start_vms.clear_nodes(nodes_ids)
        logger.info("Nodes has been removed")

    if os.path.exists(vm_ids_fname):
        os.remove(vm_ids_fname)


def store_nodes_in_log(cfg, nodes_ids):
    with open(cfg['vm_ids_fname'], 'w') as fd:
        fd.write("\n".join(nodes_ids))


def clear_enviroment(cfg, ctx):
    if os.path.exists(cfg_dict['vm_ids_fname']):
        shut_down_vms_stage(cfg, ctx)


def disconnect_stage(cfg, ctx):
    ssh_utils.close_all_sessions()

    for node in ctx.nodes:
        if node.connection is not None:
            node.connection.close()


def store_raw_results_stage(cfg, ctx):

    raw_results = os.path.join(cfg_dict['var_dir'], 'raw_results.yaml')

    if os.path.exists(raw_results):
        cont = yaml.load(open(raw_results).read())
    else:
        cont = []

    cont.extend(utils.yamable(ctx.results))
    raw_data = pretty_yaml.dumps(cont)

    with open(raw_results, "w") as fd:
        fd.write(raw_data)


def console_report_stage(cfg, ctx):
    for tp, data in ctx.results:
        if 'io' == tp and data is not None:
            dinfo = report.process_disk_info(data)
            print("\n")
            print(IOPerfTest.format_for_console(data, dinfo))
            print("\n")
        if tp in ['mysql', 'pgbench'] and data is not None:
            print("\n")
            print(MysqlTest.format_for_console(data))
            print("\n")


def html_report_stage(cfg, ctx):
    html_rep_fname = cfg['html_report_file']
    found = False
    for tp, data in ctx.results:
        if 'io' == tp and data is not None:
            if found:
                logger.error("Making reports for more than one " +
                             "io block isn't supported! All " +
                             "report, except first are skipped")
                continue
            found = True
            dinfo = report.process_disk_info(data)
            report.make_io_report(dinfo, data, html_rep_fname,
                                  cfg['charts_img_path'],
                                  lab_info=ctx.hw_info)

            text_rep_fname = cfg_dict['text_report_file']
            with open(text_rep_fname, "w") as fd:
                fd.write(IOPerfTest.format_for_console(data, dinfo))
                fd.write("\n")
                fd.flush()

            logger.info("Text report were stored in " + text_rep_fname)


def complete_log_nodes_statistic(cfg, ctx):
    nodes = ctx.nodes
    for node in nodes:
        logger.debug(str(node))


def load_data_from(var_dir):
    def load_data_from_file(cfg, ctx):
        raw_results = os.path.join(var_dir, 'raw_results.yaml')
        ctx.results = yaml.load(open(raw_results).read())
    return load_data_from_file


def start_web_ui(cfg, ctx):
    if webui is None:
        logger.error("Can't start webui. Install cherrypy module")
        ctx.web_thread = None
    else:
        th = threading.Thread(None, webui.web_main_thread, "webui", (None,))
        th.daemon = True
        th.start()
        ctx.web_thread = th


def stop_web_ui(cfg, ctx):
    webui.web_main_stop()
    time.sleep(1)


def parse_args(argv):
    descr = "Disk io performance test suite"
    parser = argparse.ArgumentParser(prog='wally', description=descr)

    parser.add_argument("-l", dest='extra_logs',
                        action='store_true', default=False,
                        help="print some extra log info")
    parser.add_argument("-b", '--build_description',
                        type=str, default="Build info")
    parser.add_argument("-i", '--build_id', type=str, default="id")
    parser.add_argument("-t", '--build_type', type=str, default="GA")
    parser.add_argument("-u", '--username', type=str, default="admin")
    parser.add_argument("-n", '--no-tests', action='store_true',
                        help="Don't run tests", default=False)
    parser.add_argument("-p", '--post-process-only', metavar="VAR_DIR",
                        help="Only process data from previour run")
    parser.add_argument("-k", '--keep-vm', action='store_true',
                        help="Don't remove test vm's", default=False)
    parser.add_argument("-d", '--dont-discover-nodes', action='store_true',
                        help="Don't connect/discover fuel nodes",
                        default=False)
    parser.add_argument("-r", '--no-html-report', action='store_true',
                        help="Skip html report", default=False)
    parser.add_argument("--params", metavar="testname.paramname",
                        help="Test params", default=[])
    parser.add_argument("--reuse-vms", default=[], nargs='*')
    parser.add_argument("config_file")

    return parser.parse_args(argv[1:])


def main(argv):
    opts = parse_args(argv)
    load_config(opts.config_file, opts.post_process_only)

    if opts.post_process_only is not None:
        stages = [
            load_data_from(opts.post_process_only)
        ]
    else:
        stages = [
            discover_stage
        ]

        for reuse_param in opts.reuse_vms:
            pref, ssh_templ = reuse_param.split(',', 1)
            stages.append(reuse_vms_stage(pref, ssh_templ))

        stages.extend([
            log_nodes_statistic,
            save_nodes_stage,
            connect_stage])

        if cfg_dict.get('collect_info', True):
            stages.append(collect_hw_info_stage)

        stages.extend([
            deploy_sensors_stage,
            run_tests_stage,
            store_raw_results_stage
        ])

    report_stages = [
        console_report_stage,
    ]

    if not opts.no_html_report:
        report_stages.append(html_report_stage)

    if cfg_dict.get('logging', {}).get("extra_logs", False) or opts.extra_logs:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    setup_loggers(level, cfg_dict['log_file'])

    logger.info("All info would be stored into {0}".format(
        cfg_dict['var_dir']))

    ctx = Context()
    ctx.build_meta['build_id'] = opts.build_id
    ctx.build_meta['build_descrption'] = opts.build_description
    ctx.build_meta['build_type'] = opts.build_type
    ctx.build_meta['username'] = opts.username
    ctx.sensors_data = SensorDatastore()

    cfg_dict['keep_vm'] = opts.keep_vm
    cfg_dict['no_tests'] = opts.no_tests
    cfg_dict['dont_discover_nodes'] = opts.dont_discover_nodes

    if cfg_dict.get('run_web_ui', False):
        start_web_ui(cfg_dict, ctx)

    msg_templ = "Exception during {0.__name__}: {1!s}"
    msg_templ_no_exc = "During {0.__name__}"

    try:
        for stage in stages:
            logger.info("Start {0.__name__} stage".format(stage))
            stage(cfg_dict, ctx)
    except utils.StopTestError as exc:
        logger.error(msg_templ.format(stage, exc))
    except Exception:
        logger.exception(msg_templ_no_exc.format(stage))
    finally:
        exc, cls, tb = sys.exc_info()
        for stage in ctx.clear_calls_stack[::-1]:
            try:
                logger.info("Start {0.__name__} stage".format(stage))
                stage(cfg_dict, ctx)
            except utils.StopTestError as cleanup_exc:
                logger.error(msg_templ.format(stage, cleanup_exc))
            except Exception:
                logger.exception(msg_templ_no_exc.format(stage))

        logger.debug("Start utils.cleanup")
        for clean_func, args, kwargs in utils.iter_clean_func():
            try:
                clean_func(*args, **kwargs)
            except utils.StopTestError as cleanup_exc:
                logger.error(msg_templ.format(stage, cleanup_exc))
            except Exception:
                logger.exception(msg_templ_no_exc.format(stage))

    if exc is None:
        for report_stage in report_stages:
            report_stage(cfg_dict, ctx)

    logger.info("All info stored in {0} folder".format(cfg_dict['var_dir']))

    if cfg_dict.get('run_web_ui', False):
        stop_web_ui(cfg_dict, ctx)

    if exc is None:
        logger.info("Tests finished successfully")
        return 0
    else:
        logger.error("Tests are failed. See detailed error above")
        return 1
