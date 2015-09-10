import os
import re
import time
import logging
import functools
import contextlib
import collections

from yaml import load as _yaml_load

try:
    from yaml import CLoader
    yaml_load = functools.partial(_yaml_load, Loader=CLoader)
except ImportError:
    yaml_load = _yaml_load

from concurrent.futures import ThreadPoolExecutor

from wally.hw_info import get_hw_info
from wally.config import get_test_files
from wally.discover import discover, Node
from wally import pretty_yaml, utils, report, ssh_utils, start_vms
from wally.sensors_utils import with_sensors_util, sensors_info_util

from wally.suits.mysql import MysqlTest
from wally.suits.itest import TestConfig
from wally.suits.io.fio import IOPerfTest
from wally.suits.postgres import PgBenchTest


TOOL_TYPE_MAPPER = {
    "io": IOPerfTest,
    "pgbench": PgBenchTest,
    "mysql": MysqlTest,
}


logger = logging.getLogger("wally")


def connect_all(nodes, spawned_node=False):
    """
    Connect to all nodes, log errors
    nodes:[Node] - list of nodes
    spawned_node:bool - whenever nodes is newly spawned VM
    """

    logger.info("Connecting to nodes")

    conn_timeout = 240 if spawned_node else 30

    def connect_ext(conn_url):
        try:
            return ssh_utils.connect(conn_url, conn_timeout=conn_timeout)
        except Exception as exc:
            logger.error("During connect to {0}: {1!s}".format(conn_url, exc))
            return None

    urls = []
    ssh_pref = "ssh://"

    for node in nodes:
        if node.conn_url == 'local':
            urls.append(node.conn_url)
        elif node.conn_url.startswith(ssh_pref):
            urls.append(node.conn_url[len(ssh_pref):])
        else:
            msg = "Unknown url type {0}".format(node.conn_url)
            logger.error(msg)
            raise utils.StopTestError(msg)

    with ThreadPoolExecutor(32) as pool:
        for node, conn in zip(nodes, pool.map(connect_ext, urls)):
            node.connection = conn

    failed_testnodes = []
    failed_nodes = []

    for node in nodes:
        if node.connection is None:
            if 'testnode' in node.roles:
                failed_testnodes.append(node.get_conn_id())
            else:
                failed_nodes.append(node.get_conn_id())

    if failed_nodes != []:
        msg = "Node(s) {0} would be excluded - can't connect"
        logger.warning(msg.format(",".join(failed_nodes)))

    if failed_testnodes != []:
        msg = "Can't connect to testnode(s) " + ",".join(failed_testnodes)
        logger.error(msg)
        raise utils.StopTestError(msg)

    if len(failed_nodes) == 0:
        logger.info("All nodes connected successfully")


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
                    cfg.hwinfo_directory,
                    info.hostname + "_lshw.xml")

                with open(fname, "w") as fd:
                    fd.write(info.raw)
    logger.info("Hardware report stored in " + cfg['hwreport_fname'])
    logger.debug("Raw hardware info in " + cfg['hwinfo_directory'] + " folder")


@contextlib.contextmanager
def suspend_vm_nodes_ctx(unused_nodes):
    pausable_nodes_ids = [node.os_vm_id for node in unused_nodes
                          if node.os_vm_id is not None]
    non_pausable = len(unused_nodes) - len(pausable_nodes_ids)

    if 0 != non_pausable:
        logger.warning("Can't pause {0} nodes".format(
                       non_pausable))

    if len(pausable_nodes_ids) != 0:
        logger.debug("Try to pause {0} unused nodes".format(
                     len(pausable_nodes_ids)))
        start_vms.pause(pausable_nodes_ids)

    try:
        yield pausable_nodes_ids
    finally:
        if len(pausable_nodes_ids) != 0:
            logger.debug("Unpausing {0} nodes".format(
                         len(pausable_nodes_ids)))
            start_vms.unpause(pausable_nodes_ids)


def generate_result_dir_name(results, name, params):
    # make a directory for results
    all_tests_dirs = os.listdir(results)

    if 'name' in params:
        dir_name = "{0}_{1}".format(name, params['name'])
    else:
        for idx in range(len(all_tests_dirs) + 1):
            dir_name = "{0}_{1}".format(name, idx)
            if dir_name not in all_tests_dirs:
                break
        else:
            raise utils.StopTestError("Can't select directory for test results")

    return os.path.join(results, dir_name)


def run_tests(cfg, test_block, nodes):
    """
    Run test from test block
    """
    test_nodes = [node for node in nodes if 'testnode' in node.roles]
    not_test_nodes = [node for node in nodes if 'testnode' not in node.roles]

    if len(test_nodes) == 0:
        logger.error("No test nodes found")
        return

    for name, params in test_block.items():
        results = []

        # iterate over all node counts
        limit = params.get('node_limit', len(test_nodes))
        if isinstance(limit, (int, long)):
            vm_limits = [limit]
        else:
            list_or_tpl = isinstance(limit, (tuple, list))
            all_ints = list_or_tpl and all(isinstance(climit, (int, long))
                                           for climit in limit)
            if not all_ints:
                msg = "'node_limit' parameter ion config should" + \
                      "be either int or list if integers, not {0!r}".format(limit)
                raise ValueError(msg)
            vm_limits = limit

        for vm_count in vm_limits:
            # select test nodes
            if vm_count == 'all':
                curr_test_nodes = test_nodes
                unused_nodes = []
            else:
                curr_test_nodes = test_nodes[:vm_count]
                unused_nodes = test_nodes[vm_count:]

            if 0 == len(curr_test_nodes):
                continue

            results_path = generate_result_dir_name(cfg.results_storage, name, params)
            utils.mkdirs_if_unxists(results_path)

            # suspend all unused virtual nodes
            if cfg.settings.get('suspend_unused_vms', True):
                suspend_ctx = suspend_vm_nodes_ctx(unused_nodes)
            else:
                suspend_ctx = utils.empty_ctx()

            with suspend_ctx:
                resumable_nodes_ids = [node.os_vm_id for node in curr_test_nodes
                                       if node.os_vm_id is not None]

                if len(resumable_nodes_ids) != 0:
                    logger.debug("Check and unpause {0} nodes".format(
                                 len(resumable_nodes_ids)))
                    start_vms.unpause(resumable_nodes_ids)

                sens_nodes = curr_test_nodes + not_test_nodes
                with sensors_info_util(cfg, sens_nodes) as sensor_data:
                    test_cls = TOOL_TYPE_MAPPER[name]

                    remote_dir = cfg.default_test_local_folder.format(name=name)

                    test_cfg = TestConfig(test_cls.__name__,
                                          params=params,
                                          test_uuid=cfg.run_uuid,
                                          nodes=test_nodes,
                                          log_directory=results_path,
                                          remote_dir=remote_dir)

                    t_start = time.time()
                    res = test_cls(test_cfg).run()
                    t_end = time.time()

            # save sensor data
            if sensor_data is not None:
                fname = "{0}_{1}.csv".format(int(t_start), int(t_end))
                fpath = os.path.join(cfg.sensor_storage, fname)

                with open(fpath, "w") as fd:
                    fd.write("\n\n".join(sensor_data))

            results.append(res)

        yield name, results


def connect_stage(cfg, ctx):
    ctx.clear_calls_stack.append(disconnect_stage)
    connect_all(ctx.nodes)
    ctx.nodes = [node for node in ctx.nodes if node.connection is not None]


def discover_stage(cfg, ctx):
    """
    discover clusters and nodes stage
    """
    if cfg.get('discover') is not None:
        discover_objs = [i.strip() for i in cfg.discover.strip().split(",")]

        nodes = discover(ctx,
                         discover_objs,
                         cfg.clouds,
                         cfg.results_storage,
                         not cfg.dont_discover_nodes)

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

    with open(cfg.nodes_report_file, "w") as fd:
        fd.write(pretty_yaml.dumps(cluster))


def reuse_vms_stage(cfg, ctx):
    vms_patterns = cfg.get('clouds', {}).get('openstack', {}).get('vms', [])
    private_key_path = get_vm_keypair(cfg)['keypair_file_private']

    for creds in vms_patterns:
        user_name, vm_name_pattern = creds.split("@", 1)
        msg = "Vm like {0} lookup failed".format(vm_name_pattern)

        with utils.log_error(msg):
            msg = "Looking for vm with name like {0}".format(vm_name_pattern)
            logger.debug(msg)

            if not start_vms.is_connected():
                os_creds = get_OS_credentials(cfg, ctx)
            else:
                os_creds = None

            conn = start_vms.nova_connect(os_creds)
            for ip, vm_id in start_vms.find_vms(conn, vm_name_pattern):
                conn_url = "ssh://{user}@{ip}::{key}".format(user=user_name,
                                                             ip=ip,
                                                             key=private_key_path)
                node = Node(conn_url, ['testnode'])
                node.os_vm_id = vm_id
                ctx.nodes.append(node)


def get_OS_credentials(cfg, ctx):
    creds = None
    os_creds = None
    force_insecure = False

    if 'openstack' in cfg.clouds:
        os_cfg = cfg.clouds['openstack']
        if 'OPENRC' in os_cfg:
            logger.info("Using OS credentials from " + os_cfg['OPENRC'])
            creds_tuple = utils.get_creds_openrc(os_cfg['OPENRC'])
            os_creds = start_vms.OSCreds(*creds_tuple)
        elif 'ENV' in os_cfg:
            logger.info("Using OS credentials from shell environment")
            os_creds = start_vms.ostack_get_creds()
        elif 'OS_TENANT_NAME' in os_cfg:
            logger.info("Using predefined credentials")
            os_creds = start_vms.OSCreds(os_cfg['OS_USERNAME'].strip(),
                                         os_cfg['OS_PASSWORD'].strip(),
                                         os_cfg['OS_TENANT_NAME'].strip(),
                                         os_cfg['OS_AUTH_URL'].strip(),
                                         os_cfg.get('OS_INSECURE', False))

        elif 'OS_INSECURE' in os_cfg:
            force_insecure = os_cfg.get('OS_INSECURE', False)

    if os_creds is None and 'fuel' in cfg.clouds and \
       'openstack_env' in cfg.clouds['fuel'] and \
       ctx.fuel_openstack_creds is not None:
        logger.info("Using fuel creds")
        creds = start_vms.OSCreds(**ctx.fuel_openstack_creds)
    elif os_creds is None:
        logger.error("Can't found OS credentials")
        raise utils.StopTestError("Can't found OS credentials", None)

    if creds is None:
        creds = os_creds

    if force_insecure and not creds.insecure:
        creds = start_vms.OSCreds(creds.name,
                                  creds.passwd,
                                  creds.tenant,
                                  creds.auth_url,
                                  True)

    logger.debug(("OS_CREDS: user={0.name} tenant={0.tenant}" +
                  "auth_url={0.auth_url} insecure={0.insecure}").format(creds))

    return creds


def get_vm_keypair(cfg):
    res = {}
    for field, ext in (('keypair_file_private', 'pem'),
                       ('keypair_file_public', 'pub')):
        fpath = cfg.vm_configs.get(field)

        if fpath is None:
            fpath = cfg.vm_configs['keypair_name'] + "." + ext

        if os.path.isabs(fpath):
            res[field] = fpath
        else:
            res[field] = os.path.join(cfg.config_folder, fpath)
    return res


@contextlib.contextmanager
def create_vms_ctx(ctx, cfg, config, already_has_count=0):
    if config['count'].startswith('='):
        count = int(config['count'][1:])
        if count <= already_has_count:
            logger.debug("Not need new vms")
            yield []
            return

    params = cfg.vm_configs[config['cfg_name']].copy()
    os_nodes_ids = []

    if not start_vms.is_connected():
        os_creds = get_OS_credentials(cfg, ctx)
    else:
        os_creds = None

    nova = start_vms.nova_connect(os_creds)

    params.update(config)
    params.update(get_vm_keypair(cfg))

    params['group_name'] = cfg.run_uuid
    params['keypair_name'] = cfg.vm_configs['keypair_name']

    if not config.get('skip_preparation', False):
        logger.info("Preparing openstack")
        start_vms.prepare_os_subpr(nova, params, os_creds)

    new_nodes = []
    old_nodes = ctx.nodes[:]
    try:
        for new_node, node_id in start_vms.launch_vms(nova, params, already_has_count):
            new_node.roles.append('testnode')
            ctx.nodes.append(new_node)
            os_nodes_ids.append(node_id)
            new_nodes.append(new_node)

        store_nodes_in_log(cfg, os_nodes_ids)
        ctx.openstack_nodes_ids = os_nodes_ids

        yield new_nodes

    finally:
        if not cfg.keep_vm:
            shut_down_vms_stage(cfg, ctx)
        ctx.nodes = old_nodes


def run_tests_stage(cfg, ctx):
    ctx.results = collections.defaultdict(lambda: [])

    for group in cfg.get('tests', []):

        if len(group.items()) != 1:
            msg = "Items in tests section should have len == 1"
            logger.error(msg)
            raise utils.StopTestError(msg)

        key, config = group.items()[0]

        if 'start_test_nodes' == key:
            if 'openstack' not in config:
                msg = "No openstack block in config - can't spawn vm's"
                logger.error(msg)
                raise utils.StopTestError(msg)

            num_test_nodes = 0
            for node in ctx.nodes:
                if 'testnode' in node.roles:
                    num_test_nodes += 1

            vm_ctx = create_vms_ctx(ctx, cfg, config['openstack'],
                                    num_test_nodes)
            tests = config.get('tests', [])
        else:
            vm_ctx = utils.empty_ctx([])
            tests = [group]

        if cfg.get('sensors') is None:
            sensor_ctx = utils.empty_ctx()
        else:
            sensor_ctx = with_sensors_util(cfg.get('sensors'), ctx.nodes)

        with vm_ctx as new_nodes:
            if len(new_nodes) != 0:
                connect_all(new_nodes, True)

            if not cfg.no_tests:
                for test_group in tests:
                    with sensor_ctx:
                        for tp, res in run_tests(cfg, test_group, ctx.nodes):
                            ctx.results[tp].extend(res)


def shut_down_vms_stage(cfg, ctx):
    vm_ids_fname = cfg.vm_ids_fname
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
    with open(cfg.vm_ids_fname, 'w') as fd:
        fd.write("\n".join(nodes_ids))


def clear_enviroment(cfg, ctx):
    if os.path.exists(cfg.vm_ids_fname):
        shut_down_vms_stage(cfg, ctx)


def disconnect_stage(cfg, ctx):
    ssh_utils.close_all_sessions()

    for node in ctx.nodes:
        if node.connection is not None:
            node.connection.close()


def store_raw_results_stage(cfg, ctx):
    if os.path.exists(cfg.raw_results):
        cont = yaml_load(open(cfg.raw_results).read())
    else:
        cont = []

    cont.extend(utils.yamable(ctx.results).items())
    raw_data = pretty_yaml.dumps(cont)

    with open(cfg.raw_results, "w") as fd:
        fd.write(raw_data)


def console_report_stage(cfg, ctx):
    first_report = True
    text_rep_fname = cfg.text_report_file
    with open(text_rep_fname, "w") as fd:
        for tp, data in ctx.results.items():
            if 'io' == tp and data is not None:
                rep_lst = []
                for result in data:
                    rep_lst.append(
                        IOPerfTest.format_for_console(list(result)))
                rep = "\n\n".join(rep_lst)
            elif tp in ['mysql', 'pgbench'] and data is not None:
                rep = MysqlTest.format_for_console(data)
            else:
                logger.warning("Can't generate text report for " + tp)
                continue

            fd.write(rep)
            fd.write("\n")

            if first_report:
                logger.info("Text report were stored in " + text_rep_fname)
                first_report = False

            print("\n" + rep + "\n")


def test_load_report_stage(cfg, ctx):
    load_rep_fname = cfg.load_report_file
    found = False
    for idx, (tp, data) in enumerate(ctx.results.items()):
        if 'io' == tp and data is not None:
            if found:
                logger.error("Making reports for more than one " +
                             "io block isn't supported! All " +
                             "report, except first are skipped")
                continue
            found = True
            report.make_load_report(idx, cfg['results'], load_rep_fname)


def html_report_stage(cfg, ctx):
    html_rep_fname = cfg.html_report_file
    found = False
    for tp, data in ctx.results.items():
        if 'io' == tp and data is not None:
            if found or len(data) > 1:
                logger.error("Making reports for more than one " +
                             "io block isn't supported! All " +
                             "report, except first are skipped")
                continue
            found = True
            report.make_io_report(list(data[0]),
                                  cfg.get('comment', ''),
                                  html_rep_fname,
                                  lab_info=ctx.hw_info)


def load_data_from_path(test_res_dir):
    files = get_test_files(test_res_dir)
    raw_res = yaml_load(open(files['raw_results']).read())
    res = collections.defaultdict(lambda: [])

    for tp, test_lists in raw_res:
        for tests in test_lists:
            for suite_name, suite_data in tests.items():
                result_folder = suite_data[0]
                res[tp].append(TOOL_TYPE_MAPPER[tp].load(suite_name, result_folder))

    return res


def load_data_from_path_stage(var_dir, _, ctx):
    for tp, vals in load_data_from_path(var_dir).items():
        ctx.results.setdefault(tp, []).extend(vals)


def load_data_from(var_dir):
    return functools.partial(load_data_from_path_stage, var_dir)
