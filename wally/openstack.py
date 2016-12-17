import os.path
import socket
import logging
from typing import Dict, Any, List, Tuple, cast, Optional

from .node_interfaces import NodeInfo
from .config import ConfigBlock, Config
from .ssh_utils import ConnCreds
from .openstack_api import (os_connect, find_vms,
                            OSCreds, get_openstack_credentials, prepare_os, launch_vms, clear_nodes)
from .test_run_class import TestRun
from .stage import Stage, StepOrder
from .utils import LogError, StopTestError, get_creds_openrc


logger = logging.getLogger("wally.discover")


def get_floating_ip(vm: Any) -> str:
    """Get VM floating IP address"""

    for net_name, ifaces in vm.addresses.items():
        for iface in ifaces:
            if iface.get('OS-EXT-IPS:type') == "floating":
                return iface['addr']

    raise ValueError("VM {} has no floating ip".format(vm))


def ensure_connected_to_openstack(ctx: TestRun) -> None:
    if not ctx.os_connection is None:
        if ctx.os_creds is None:
            ctx.os_creds = get_OS_credentials(ctx)
        ctx.os_connection = os_connect(ctx.os_creds)


def get_OS_credentials(ctx: TestRun) -> OSCreds:
    if "openstack_openrc" in ctx.storage:
        return ctx.storage.load(OSCreds, "openstack_openrc")

    creds = None
    os_creds = None
    force_insecure = False
    cfg = ctx.config

    if 'openstack' in cfg.clouds:
        os_cfg = cfg.clouds['openstack']
        if 'OPENRC' in os_cfg:
            logger.info("Using OS credentials from " + os_cfg['OPENRC'])
            creds_tuple = get_creds_openrc(os_cfg['OPENRC'])
            os_creds = OSCreds(*creds_tuple)
        elif 'ENV' in os_cfg:
            logger.info("Using OS credentials from shell environment")
            os_creds = get_openstack_credentials()
        elif 'OS_TENANT_NAME' in os_cfg:
            logger.info("Using predefined credentials")
            os_creds = OSCreds(os_cfg['OS_USERNAME'].strip(),
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
        raise StopTestError("Can't found OS credentials", None)

    if creds is None:
        creds = os_creds

    if force_insecure and not creds.insecure:
        creds = OSCreds(creds.name, creds.passwd, creds.tenant, creds.auth_url, True)

    logger.debug(("OS_CREDS: user={0.name} tenant={0.tenant} " +
                  "auth_url={0.auth_url} insecure={0.insecure}").format(creds))

    ctx.storage["openstack_openrc"] = creds  # type: ignore
    return creds


def get_vm_keypair_path(cfg: Config) -> Tuple[str, str]:
    key_name = cfg.vm_configs['keypair_name']
    private_path = os.path.join(cfg.settings_dir, key_name + "_private.pem")
    public_path = os.path.join(cfg.settings_dir, key_name + "_public.pub")
    return (private_path, public_path)


class DiscoverOSStage(Stage):
    """Discover openstack nodes and VMS"""

    config_block = 'openstack'

    # discover FUEL cluster first
    priority = StepOrder.DISCOVER + 1

    @classmethod
    def validate(cls, conf: ConfigBlock) -> None:
        pass

    def run(self, ctx: TestRun) -> None:
        cfg = ctx.config.openstack
        os_nodes_auth = cfg.auth  # type: str

        if os_nodes_auth.count(":") == 2:
            user, password, key_file = os_nodes_auth.split(":")  # type: str, Optional[str], Optional[str]
            if not password:
                password = None
        else:
            user, password = os_nodes_auth.split(":")
            key_file = None

        ensure_connected_to_openstack(ctx)

        if 'openstack_nodes' in ctx.storage:
            ctx.nodes_info.extend(ctx.storage.load_list(NodeInfo, "openstack_nodes"))
        else:
            openstack_nodes = []  # type: List[NodeInfo]
            services = ctx.os_connection.nova.services.list()  # type: List[Any]
            host_services_mapping = {}  # type: Dict[str, List[str]]

            for service in services:
                ip = cast(str, socket.gethostbyname(service.host))
                host_services_mapping.get(ip, []).append(service.binary)

            logger.debug("Found %s openstack service nodes" % len(host_services_mapping))

            for host, services in host_services_mapping.items():
                creds = ConnCreds(host=host, user=user, passwd=password, key_file=key_file)
                openstack_nodes.append(NodeInfo(creds, set(services)))

            ctx.nodes_info.extend(openstack_nodes)
            ctx.storage['openstack_nodes'] = openstack_nodes  # type: ignore

        if "reused_os_nodes" in ctx.storage:
            ctx.nodes_info.extend(ctx.storage.load_list(NodeInfo, "reused_nodes"))
        else:
            reused_nodes = []  # type: List[NodeInfo]
            private_key_path = get_vm_keypair_path(ctx.config)[0]

            vm_creds = None  # type: str
            for vm_creds in cfg.get("vms", []):
                user_name, vm_name_pattern = vm_creds.split("@", 1)
                msg = "Vm like {} lookup failed".format(vm_name_pattern)

                with LogError(msg):
                    msg = "Looking for vm with name like {0}".format(vm_name_pattern)
                    logger.debug(msg)

                    ensure_connected_to_openstack(ctx)

                    for ip, vm_id in find_vms(ctx.os_connection, vm_name_pattern):
                        creds = ConnCreds(host=ip, user=user_name, key_file=private_key_path)
                        node_info = NodeInfo(creds, {'testnode'})
                        node_info.os_vm_id = vm_id
                        reused_nodes.append(node_info)

            ctx.nodes_info.extend(reused_nodes)
            ctx.storage["reused_os_nodes"] = reused_nodes  # type: ignore


class CreateOSVMSStage(Stage):
    "Spawn new VM's in Openstack cluster"

    priority = StepOrder.SPAWN  # type: int
    config_block = 'spawn_os_vms'  # type: str

    def run(self, ctx: TestRun) -> None:
        vm_spawn_config = ctx.config.spawn_os_vms
        vm_image_config = ctx.config.vm_configs[vm_spawn_config.cfg_name]

        if 'spawned_os_nodes' in ctx.storage:
            ctx.nodes_info.extend(ctx.storage.load_list(NodeInfo, "spawned_os_nodes"))
        else:
            ensure_connected_to_openstack(ctx)
            params = vm_image_config.copy()
            params.update(vm_spawn_config)
            params.update(get_vm_keypair_path(ctx.config))
            params['group_name'] = ctx.config.run_uuid
            params['keypair_name'] = ctx.config.vm_configs['keypair_name']

            if not ctx.config.openstack.get("skip_preparation", False):
                logger.info("Preparing openstack")
                prepare_os(ctx.os_connection, params)

            new_nodes = []
            ctx.os_spawned_nodes_ids = []
            with ctx.get_pool() as pool:
                for node_info in launch_vms(ctx.os_connection, params, pool):
                    node_info.roles.add('testnode')
                    ctx.os_spawned_nodes_ids.append(node_info.os_vm_id)
                    new_nodes.append(node_info)

            ctx.storage['spawned_os_nodes'] = new_nodes  # type: ignore

    def cleanup(self, ctx: TestRun) -> None:
        # keep nodes in case of error for future test restart
        if not ctx.config.keep_vm and ctx.os_spawned_nodes_ids:
            logger.info("Removing nodes")

            clear_nodes(ctx.os_connection, ctx.os_spawned_nodes_ids)
            del ctx.storage['spawned_os_nodes']

            logger.info("Nodes has been removed")



# @contextlib.contextmanager
# def suspend_vm_nodes_ctx(ctx: TestRun, unused_nodes: List[IRPCNode]) -> Iterator[List[int]]:
#
#     pausable_nodes_ids = [cast(int, node.info.os_vm_id)
#                           for node in unused_nodes
#                           if node.info.os_vm_id is not None]
#
#     non_pausable = len(unused_nodes) - len(pausable_nodes_ids)
#
#     if non_pausable:
#         logger.warning("Can't pause {} nodes".format(non_pausable))
#
#     if pausable_nodes_ids:
#         logger.debug("Try to pause {} unused nodes".format(len(pausable_nodes_ids)))
#         with ctx.get_pool() as pool:
#             openstack_api.pause(ctx.os_connection, pausable_nodes_ids, pool)
#
#     try:
#         yield pausable_nodes_ids
#     finally:
#         if pausable_nodes_ids:
#             logger.debug("Unpausing {} nodes".format(len(pausable_nodes_ids)))
#             with ctx.get_pool() as pool:
#                 openstack_api.unpause(ctx.os_connection, pausable_nodes_ids, pool)
# def clouds_connect_stage(ctx: TestRun) -> None:
    # TODO(koder): need to use this to connect to openstack in upper code
    # conn = ctx.config['clouds/openstack']
    # user, passwd, tenant = parse_creds(conn['creds'])
    # auth_data = dict(auth_url=conn['auth_url'],
    #                  username=user,
    #                  api_key=passwd,
    #                  project_id=tenant)  # type: Dict[str, str]
    # logger.debug("Discovering openstack nodes with connection details: %r", conn)
    # connect to openstack, fuel

    # # parse FUEL REST credentials
    # username, tenant_name, password = parse_creds(fuel_data['creds'])
    # creds = {"username": username,
    #          "tenant_name": tenant_name,
    #          "password": password}
    #
    # # connect to FUEL
    # conn = fuel_rest_api.KeystoneAuth(fuel_data['url'], creds, headers=None)
    # pass