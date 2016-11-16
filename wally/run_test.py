import os
import logging
import contextlib
from typing import List, Dict, Iterable, Iterator, Tuple, Optional, Union, cast
from concurrent.futures import ThreadPoolExecutor, Future

from .node_interfaces import NodeInfo, IRPCNode
from .test_run_class import TestRun
from .discover import discover
from . import pretty_yaml, utils, report, ssh_utils, start_vms, hw_info
from .node import setup_rpc, connect
from .config import ConfigBlock, Config

from .suits.mysql import MysqlTest
from .suits.itest import TestConfig
from .suits.io.fio import IOPerfTest
from .suits.postgres import PgBenchTest
from .suits.omgbench import OmgTest


TOOL_TYPE_MAPPER = {
    "io": IOPerfTest,
    "pgbench": PgBenchTest,
    "mysql": MysqlTest,
    "omg": OmgTest,
}


logger = logging.getLogger("wally")


def connect_all(nodes_info: List[NodeInfo], pool: ThreadPoolExecutor, conn_timeout: int = 30) -> List[IRPCNode]:
    """Connect to all nodes, log errors"""

    logger.info("Connecting to %s nodes", len(nodes_info))

    def connect_ext(node_info: NodeInfo) -> Tuple[bool, Union[IRPCNode, NodeInfo]]:
        try:
            ssh_node = connect(node_info, conn_timeout=conn_timeout)
            # TODO(koder): need to pass all required rpc bytes to this call
            return True, setup_rpc(ssh_node, b"")
        except Exception as exc:
            logger.error("During connect to {}: {!s}".format(node, exc))
            return False, node_info

    failed_testnodes = []  # type: List[NodeInfo]
    failed_nodes = []  # type: List[NodeInfo]
    ready = []  # type: List[IRPCNode]

    for ok, node in pool.map(connect_ext, nodes_info):
        if not ok:
            node = cast(NodeInfo, node)
            if 'testnode' in node.roles:
                failed_testnodes.append(node)
            else:
                failed_nodes.append(node)
        else:
            ready.append(cast(IRPCNode, node))

    if failed_nodes:
        msg = "Node(s) {} would be excluded - can't connect"
        logger.warning(msg.format(",".join(map(str, failed_nodes))))

    if failed_testnodes:
        msg = "Can't connect to testnode(s) " + \
              ",".join(map(str, failed_testnodes))
        logger.error(msg)
        raise utils.StopTestError(msg)

    if not failed_nodes:
        logger.info("All nodes connected successfully")

    return ready


def collect_info_stage(ctx: TestRun) -> None:
    futures = {}  # type: Dict[str, Future]

    with ctx.get_pool() as pool:
        for node in ctx.nodes:
            hw_info_path = "hw_info/{}".format(node.info.node_id())
            if hw_info_path not in ctx.storage:
                futures[hw_info_path] = pool.submit(hw_info.get_hw_info, node), node

            sw_info_path = "sw_info/{}".format(node.info.node_id())
            if sw_info_path not in ctx.storage:
                futures[sw_info_path] = pool.submit(hw_info.get_sw_info, node)

        for path, future in futures.items():
            ctx.storage[path] = future.result()


@contextlib.contextmanager
def suspend_vm_nodes_ctx(ctx: TestRun, unused_nodes: List[IRPCNode]) -> Iterator[List[int]]:

    pausable_nodes_ids = [cast(int, node.info.os_vm_id)
                          for node in unused_nodes
                          if node.info.os_vm_id is not None]

    non_pausable = len(unused_nodes) - len(pausable_nodes_ids)

    if non_pausable:
        logger.warning("Can't pause {} nodes".format(non_pausable))

    if pausable_nodes_ids:
        logger.debug("Try to pause {} unused nodes".format(len(pausable_nodes_ids)))
        with ctx.get_pool() as pool:
            start_vms.pause(ctx.os_connection, pausable_nodes_ids, pool)

    try:
        yield pausable_nodes_ids
    finally:
        if pausable_nodes_ids:
            logger.debug("Unpausing {} nodes".format(len(pausable_nodes_ids)))
            with ctx.get_pool() as pool:
                start_vms.unpause(ctx.os_connection, pausable_nodes_ids, pool)


def run_tests(ctx: TestRun, test_block: ConfigBlock, nodes: List[IRPCNode]) -> None:
    """Run test from test block"""

    test_nodes = [node for node in nodes if 'testnode' in node.info.roles]

    if not test_nodes:
        logger.error("No test nodes found")
        return

    for name, params in test_block.items():
        vm_count = params.get('node_limit', None)  # type: Optional[int]

        # select test nodes
        if vm_count is None:
            curr_test_nodes = test_nodes
            unused_nodes = []  # type: List[IRPCNode]
        else:
            curr_test_nodes = test_nodes[:vm_count]
            unused_nodes = test_nodes[vm_count:]

        if not curr_test_nodes:
            logger.error("No nodes found for test, skipping it.")
            continue

        # results_path = generate_result_dir_name(cfg.results_storage, name, params)
        # utils.mkdirs_if_unxists(results_path)

        # suspend all unused virtual nodes
        if ctx.config.get('suspend_unused_vms', True):
            suspend_ctx = suspend_vm_nodes_ctx(ctx, unused_nodes)
        else:
            suspend_ctx = utils.empty_ctx()

        resumable_nodes_ids = [cast(int, node.info.os_vm_id)
                               for node in curr_test_nodes
                               if node.info.os_vm_id is not None]

        if resumable_nodes_ids:
            logger.debug("Check and unpause {} nodes".format(len(resumable_nodes_ids)))

            with ctx.get_pool() as pool:
                start_vms.unpause(ctx.os_connection, resumable_nodes_ids, pool)

        with suspend_ctx:
            test_cls = TOOL_TYPE_MAPPER[name]
            remote_dir = ctx.config.default_test_local_folder.format(name=name, uuid=ctx.config.run_uuid)
            test_cfg = TestConfig(test_cls.__name__,
                                  params=params,
                                  run_uuid=ctx.config.run_uuid,
                                  nodes=test_nodes,
                                  storage=ctx.storage,
                                  remote_dir=remote_dir)

            test_cls(test_cfg).run()


def connect_stage(ctx: TestRun) -> None:
    ctx.clear_calls_stack.append(disconnect_stage)

    with ctx.get_pool() as pool:
        ctx.nodes = connect_all(ctx.nodes_info, pool)


def discover_stage(ctx: TestRun) -> None:
    """discover clusters and nodes stage"""

    # TODO(koder): Properly store discovery info and check if it available to skip phase

    discover_info = ctx.config.get('discover')
    if discover_info:
        if "discovered_nodes" in ctx.storage:
            nodes = ctx.storage.load_list("discovered_nodes", NodeInfo)
            ctx.fuel_openstack_creds = ctx.storage.load("fuel_openstack_creds", start_vms.OSCreds)
        else:
            discover_objs = [i.strip() for i in discover_info.strip().split(",")]

            ctx.fuel_openstack_creds, nodes = discover.discover(discover_objs,
                                                                ctx.config.clouds,
                                                                not ctx.config.dont_discover_nodes)

            ctx.storage["fuel_openstack_creds"] = ctx.fuel_openstack_creds  # type: ignore
            ctx.storage["discovered_nodes"] = nodes  # type: ignore
        ctx.nodes_info.extend(nodes)

    for url, roles in ctx.config.get('explicit_nodes', {}).items():
        creds = ssh_utils.parse_ssh_uri(url)
        roles = set(roles.split(","))
        ctx.nodes_info.append(NodeInfo(creds, roles))


def save_nodes_stage(ctx: TestRun) -> None:
    """Save nodes list to file"""
    ctx.storage['nodes'] = ctx.nodes_info   # type: ignore


def ensure_connected_to_openstack(ctx: TestRun) -> None:
    if not ctx.os_connection is None:
        if ctx.os_creds is None:
            ctx.os_creds = get_OS_credentials(ctx)
        ctx.os_connection = start_vms.os_connect(ctx.os_creds)


def reuse_vms_stage(ctx: TestRun) -> None:
    if "reused_nodes" in ctx.storage:
        ctx.nodes_info.extend(ctx.storage.load_list("reused_nodes", NodeInfo))
    else:
        reused_nodes = []
        vms_patterns = ctx.config.get('clouds/openstack/vms', [])
        private_key_path = get_vm_keypair_path(ctx.config)[0]

        for creds in vms_patterns:
            user_name, vm_name_pattern = creds.split("@", 1)
            msg = "Vm like {} lookup failed".format(vm_name_pattern)

            with utils.LogError(msg):
                msg = "Looking for vm with name like {0}".format(vm_name_pattern)
                logger.debug(msg)

                ensure_connected_to_openstack(ctx)

                for ip, vm_id in start_vms.find_vms(ctx.os_connection, vm_name_pattern):
                    creds = ssh_utils.ConnCreds(host=ip, user=user_name, key_file=private_key_path)
                    node_info = NodeInfo(creds, {'testnode'})
                    node_info.os_vm_id = vm_id
                    reused_nodes.append(node_info)
                    ctx.nodes_info.append(node_info)

        ctx.storage["reused_nodes"] = reused_nodes  # type: ignore


def get_OS_credentials(ctx: TestRun) -> start_vms.OSCreds:

    if "openstack_openrc" in ctx.storage:
        return ctx.storage.load("openstack_openrc", start_vms.OSCreds)

    creds = None
    os_creds = None
    force_insecure = False
    cfg = ctx.config

    if 'openstack' in cfg.clouds:
        os_cfg = cfg.clouds['openstack']
        if 'OPENRC' in os_cfg:
            logger.info("Using OS credentials from " + os_cfg['OPENRC'])
            creds_tuple = utils.get_creds_openrc(os_cfg['OPENRC'])
            os_creds = start_vms.OSCreds(*creds_tuple)
        elif 'ENV' in os_cfg:
            logger.info("Using OS credentials from shell environment")
            os_creds = start_vms.get_openstack_credentials()
        elif 'OS_TENANT_NAME' in os_cfg:
            logger.info("Using predefined credentials")
            os_creds = start_vms.OSCreds(os_cfg['OS_USERNAME'].strip(),
                                         os_cfg['OS_PASSWORD'].strip(),
                                         os_cfg['OS_TENANT_NAME'].strip(),
                                         os_cfg['OS_AUTH_URL'].strip(),
                                         os_cfg.get('OS_INSECURE', False))

        elif 'OS_INSECURE' in os_cfg:
            force_insecure = os_cfg.get('OS_INSECURE', False)

    if os_creds is None and 'fuel' in cfg.clouds and 'openstack_env' in cfg.clouds['fuel'] and \
            ctx.fuel_openstack_creds is not None:
        logger.info("Using fuel creds")
        creds = ctx.fuel_openstack_creds
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

    logger.debug(("OS_CREDS: user={0.name} tenant={0.tenant} " +
                  "auth_url={0.auth_url} insecure={0.insecure}").format(creds))

    ctx.storage["openstack_openrc"] = creds  # type: ignore
    return creds


def get_vm_keypair_path(cfg: Config) -> Tuple[str, str]:
    key_name = cfg.vm_configs['keypair_name']
    private_path = os.path.join(cfg.settings_dir, key_name + "_private.pem")
    public_path = os.path.join(cfg.settings_dir, key_name + "_public.pub")
    return (private_path, public_path)


@contextlib.contextmanager
def create_vms_ctx(ctx: TestRun, vm_config: ConfigBlock, already_has_count: int = 0) -> Iterator[List[NodeInfo]]:

    if 'spawned_vm_ids' in ctx.storage:
        os_nodes_ids = ctx.storage.get('spawned_vm_ids', [])  # type: List[int]
        new_nodes = []  # type: List[NodeInfo]

        # TODO(koder): reconnect to old VM's
        raise NotImplementedError("Reconnect to old vms is not implemented")
    else:
        os_nodes_ids = []
        new_nodes = []
        no_spawn = False
        if vm_config['count'].startswith('='):
            count = int(vm_config['count'][1:])
            if count <= already_has_count:
                logger.debug("Not need new vms")
                no_spawn = True

        if not no_spawn:
            ensure_connected_to_openstack(ctx)
            params = ctx.config.vm_configs[vm_config['cfg_name']].copy()
            params.update(vm_config)
            params.update(get_vm_keypair_path(ctx.config))
            params['group_name'] = ctx.config.run_uuid
            params['keypair_name'] = ctx.config.vm_configs['keypair_name']

            if not vm_config.get('skip_preparation', False):
                logger.info("Preparing openstack")
                start_vms.prepare_os(ctx.os_connection, params)

            with ctx.get_pool() as pool:
                for node_info in start_vms.launch_vms(ctx.os_connection, params, pool, already_has_count):
                    node_info.roles.add('testnode')
                    os_nodes_ids.append(node_info.os_vm_id)
                    new_nodes.append(node_info)

        ctx.storage['spawned_vm_ids'] = os_nodes_ids  # type: ignore
        yield new_nodes

        # keep nodes in case of error for future test restart
        if not ctx.config.keep_vm:
            shut_down_vms_stage(ctx, os_nodes_ids)

        del ctx.storage['spawned_vm_ids']


@contextlib.contextmanager
def sensor_monitoring(ctx: TestRun, cfg: ConfigBlock, nodes: List[IRPCNode]) -> Iterator[None]:
    yield


def run_tests_stage(ctx: TestRun) -> None:
    for group in ctx.config.get('tests', []):
        gitems = list(group.items())
        if len(gitems) != 1:
            msg = "Items in tests section should have len == 1"
            logger.error(msg)
            raise utils.StopTestError(msg)

        key, config = gitems[0]

        if 'start_test_nodes' == key:
            if 'openstack' not in config:
                msg = "No openstack block in config - can't spawn vm's"
                logger.error(msg)
                raise utils.StopTestError(msg)

            num_test_nodes = len([node for node in ctx.nodes if 'testnode' in node.info.roles])
            vm_ctx = create_vms_ctx(ctx, config['openstack'], num_test_nodes)
            tests = config.get('tests', [])
        else:
            vm_ctx = utils.empty_ctx([])
            tests = [group]

        # make mypy happy
        new_nodes = []  # type: List[NodeInfo]

        with vm_ctx as new_nodes:
            if new_nodes:
                with ctx.get_pool() as pool:
                    new_rpc_nodes = connect_all(new_nodes, pool)

            test_nodes = ctx.nodes + new_rpc_nodes

            if ctx.config.get('sensors'):
                sensor_ctx = sensor_monitoring(ctx, ctx.config.get('sensors'), test_nodes)
            else:
                sensor_ctx = utils.empty_ctx([])

            if not ctx.config.no_tests:
                for test_group in tests:
                    with sensor_ctx:
                        run_tests(ctx, test_group, test_nodes)

            for node in new_rpc_nodes:
                node.disconnect()


def shut_down_vms_stage(ctx: TestRun, nodes_ids: List[int]) -> None:
    if nodes_ids:
        logger.info("Removing nodes")
        start_vms.clear_nodes(ctx.os_connection, nodes_ids)
        logger.info("Nodes has been removed")


def clear_enviroment(ctx: TestRun) -> None:
    shut_down_vms_stage(ctx, ctx.storage.get('spawned_vm_ids', []))
    ctx.storage['spawned_vm_ids'] = []  # type: ignore


def disconnect_stage(ctx: TestRun) -> None:
    # TODO(koder): what next line was for?
    # ssh_utils.close_all_sessions()

    for node in ctx.nodes:
        node.disconnect()


def console_report_stage(ctx: TestRun) -> None:
    # TODO(koder): load data from storage
    raise NotImplementedError("...")
    # first_report = True
    # text_rep_fname = ctx.config.text_report_file
    #
    # with open(text_rep_fname, "w") as fd:
    #     for tp, data in ctx.results.items():
    #         if 'io' == tp and data is not None:
    #             rep_lst = []
    #             for result in data:
    #                 rep_lst.append(
    #                     IOPerfTest.format_for_console(list(result)))
    #             rep = "\n\n".join(rep_lst)
    #         elif tp in ['mysql', 'pgbench'] and data is not None:
    #             rep = MysqlTest.format_for_console(data)
    #         elif tp == 'omg':
    #             rep = OmgTest.format_for_console(data)
    #         else:
    #             logger.warning("Can't generate text report for " + tp)
    #             continue
    #
    #         fd.write(rep)
    #         fd.write("\n")
    #
    #         if first_report:
    #             logger.info("Text report were stored in " + text_rep_fname)
    #             first_report = False
    #
    #         print("\n" + rep + "\n")


# def test_load_report_stage(cfg: Config, ctx: TestRun) -> None:
#     load_rep_fname = cfg.load_report_file
#     found = False
#     for idx, (tp, data) in enumerate(ctx.results.items()):
#         if 'io' == tp and data is not None:
#             if found:
#                 logger.error("Making reports for more than one " +
#                              "io block isn't supported! All " +
#                              "report, except first are skipped")
#                 continue
#             found = True
#             report.make_load_report(idx, cfg['results'], load_rep_fname)
#
#

def html_report_stage(ctx: TestRun) -> None:
    # TODO(koder): load data from storage
    raise NotImplementedError("...")
    # html_rep_fname = cfg.html_report_file
    # found = False
    # for tp, data in ctx.results.items():
    #     if 'io' == tp and data is not None:
    #         if found or len(data) > 1:
    #             logger.error("Making reports for more than one " +
    #                          "io block isn't supported! All " +
    #                          "report, except first are skipped")
    #             continue
    #         found = True
    #         report.make_io_report(list(data[0]),
    #                               cfg.get('comment', ''),
    #                               html_rep_fname,
    #                               lab_info=ctx.nodes)

#
# def load_data_from_path(test_res_dir: str) -> Mapping[str, List[Any]]:
#     files = get_test_files(test_res_dir)
#     raw_res = yaml_load(open(files['raw_results']).read())
#     res = collections.defaultdict(list)
#
#     for tp, test_lists in raw_res:
#         for tests in test_lists:
#             for suite_name, suite_data in tests.items():
#                 result_folder = suite_data[0]
#                 res[tp].append(TOOL_TYPE_MAPPER[tp].load(suite_name, result_folder))
#
#     return res
#
#
# def load_data_from_path_stage(var_dir: str, _, ctx: TestRun) -> None:
#     for tp, vals in load_data_from_path(var_dir).items():
#         ctx.results.setdefault(tp, []).extend(vals)
#
#
# def load_data_from(var_dir: str) -> Callable[[TestRun], None]:
#     return functools.partial(load_data_from_path_stage, var_dir)
