import os.path
import socket
import logging
from typing import Dict, Any, List, Tuple, cast

from cephlib.common import to_ip

from .node_interfaces import NodeInfo
from .config import ConfigBlock, Config
from .ssh_utils import ConnCreds
from .openstack_api import (os_connect, find_vms,
                            OSCreds, get_openstack_credentials, prepare_os, launch_vms, clear_nodes)
from .test_run_class import TestRun
from .stage import Stage, StepOrder
from .utils import LogError, StopTestError, get_creds_openrc


logger = logging.getLogger("wally")


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
    stored = ctx.storage.get("openstack_openrc", None)
    if stored is not None:
        return OSCreds(*cast(List, stored))

    creds = None  # type: OSCreds
    os_creds = None  # type: OSCreds
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

    ctx.storage.put(list(creds), "openstack_openrc")
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
        if 'openstack' not in ctx.config.discover:
            logger.debug("Skip openstack discovery due to settings")
            return

        if 'all_nodes' in ctx.storage:
            logger.debug("Skip openstack discovery, use previously discovered nodes")
            return

        ensure_connected_to_openstack(ctx)

        cfg = ctx.config.openstack
        os_nodes_auth = cfg.auth  # type: str
        if os_nodes_auth.count(":") == 2:
            user, password, key_file = os_nodes_auth.split(":")  # type: str, Optional[str], Optional[str]
            if not password:
                password = None
        else:
            user, password = os_nodes_auth.split(":")
            key_file = None

        if 'metadata' not in ctx.config.discover:
            services = ctx.os_connection.nova.services.list()  # type: List[Any]
            host_services_mapping = {}  # type: Dict[str, List[str]]

            for service in services:
                ip = cast(str, socket.gethostbyname(service.host))
                host_services_mapping.get(ip, []).append(service.binary)

            logger.debug("Found %s openstack service nodes" % len(host_services_mapping))

            for host, services in host_services_mapping.items():
                host_ip = to_ip(host)
                if host != host_ip:
                    logger.info("Will use ip_addr %r instead of hostname %r", host_ip, host)
                creds = ConnCreds(host=host_ip, user=user, passwd=password, key_file=key_file)
                ctx.merge_node(creds, set(services))
            # TODO: log OS nodes discovery results
        else:
            logger.info("Skip OS cluster discovery due to 'discovery' setting value")

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
                    creds = ConnCreds(host=to_ip(ip), user=user_name, key_file=private_key_path)
                    info = NodeInfo(creds, {'testnode'})
                    info.os_vm_id = vm_id
                    nid = info.node_id
                    if nid in ctx.nodes_info:
                        logger.error("Test VM node has the same id(%s), as existing node %s", nid, ctx.nodes_info[nid])
                        raise StopTestError()
                    ctx.nodes_info[nid] = info


class CreateOSVMSStage(Stage):
    "Spawn new VM's in Openstack cluster"

    priority = StepOrder.SPAWN  # type: int
    config_block = 'spawn_os_vms'  # type: str

    def run(self, ctx: TestRun) -> None:
        if 'all_nodes' in ctx.storage:
            ctx.os_spawned_nodes_ids = ctx.storage.get('os_spawned_nodes_ids')
            logger.info("Skipping OS VMS discovery/spawn as all data found in storage")
            return

        if 'os_spawned_nodes_ids' in ctx.storage:
            logger.error("spawned_os_nodes_ids is found in storage, but no nodes_info is stored." +
                         "Fix this before continue")
            raise StopTestError()

        vm_spawn_config = ctx.config.spawn_os_vms
        vm_image_config = ctx.config.vm_configs[vm_spawn_config.cfg_name]

        ensure_connected_to_openstack(ctx)
        params = vm_image_config.copy()
        params.update(vm_spawn_config)
        params.update(get_vm_keypair_path(ctx.config))
        params['group_name'] = ctx.config.run_uuid
        params['keypair_name'] = ctx.config.vm_configs['keypair_name']

        if not ctx.config.openstack.get("skip_preparation", False):
            logger.info("Preparing openstack")
            prepare_os(ctx.os_connection, params)
        else:
            logger.info("Scip openstack preparation as 'skip_preparation' is set")

        ctx.os_spawned_nodes_ids = []
        with ctx.get_pool() as pool:
            for info in launch_vms(ctx.os_connection, params, pool):
                info.roles.add('testnode')
                nid = info.node_id
                if nid in ctx.nodes_info:
                    logger.error("Test VM node has the same id(%s), as existing node %s", nid, ctx.nodes_info[nid])
                    raise StopTestError()
                ctx.nodes_info[nid] = info
                ctx.os_spawned_nodes_ids.append(info.os_vm_id)

        ctx.storage.put(ctx.os_spawned_nodes_ids, 'os_spawned_nodes_ids')

    def cleanup(self, ctx: TestRun) -> None:
        # keep nodes in case of error for future test restart
        if not ctx.config.keep_vm and ctx.os_spawned_nodes_ids:
            logger.info("Removing nodes")

            clear_nodes(ctx.os_connection, ctx.os_spawned_nodes_ids)
            ctx.storage.rm('spawned_os_nodes')

            logger.info("OS spawned nodes has been successfully removed")
