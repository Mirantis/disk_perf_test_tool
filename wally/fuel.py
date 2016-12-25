import logging
from typing import Dict, List, NamedTuple, Union, cast

from paramiko.ssh_exception import AuthenticationException

from .fuel_rest_api import get_cluster_id, reflect_cluster, FuelInfo, KeystoneAuth
from .ssh_utils import ConnCreds
from .utils import StopTestError, parse_creds, to_ip
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .node import connect, setup_rpc
from .config import ConfigBlock
from .openstack_api import OSCreds


logger = logging.getLogger("wally")


FuelNodeInfo = NamedTuple("FuelNodeInfo",
                          [("version", List[int]),
                           ("fuel_ext_iface", str),
                           ("openrc", Dict[str, Union[str, bool]])])



class DiscoverFuelStage(Stage):
    """"Fuel nodes discovery, also can get openstack openrc"""

    priority = StepOrder.DISCOVER
    config_block = 'fuel'

    @classmethod
    def validate(cls, cfg: ConfigBlock) -> None:
        # msg = "openstack_env should be provided in fuel config"
        # check_input_param('openstack_env' in fuel_data, msg)
        # fuel.openstack_env
        pass

    def run(self, ctx: TestRun) -> None:
        discovery = ctx.config.get("discovery")
        if discovery == 'disable':
            logger.info("Skip FUEL discovery due to config setting")
            return

        if "fuel_os_creds" in ctx.storage and 'fuel_version' in ctx.storage:
            logger.debug("Skip FUEL credentials discovery, use previously discovered info")
            ctx.fuel_openstack_creds = OSCreds(*cast(List, ctx.storage.get('fuel_os_creds')))
            ctx.fuel_version = ctx.storage.get('fuel_version')
            if 'all_nodes' in ctx.storage:
                logger.debug("Skip FUEL nodes discovery, use data from DB")
                return
            elif discovery == 'metadata':
                logger.debug("Skip FUEL nodes  discovery due to discovery settings")
                return

        fuel = ctx.config.fuel
        fuel_node_info = ctx.merge_node(fuel.ssh_creds, {'fuel_master'})
        creds = dict(zip(("user", "passwd", "tenant"), parse_creds(fuel.creds)))
        fuel_conn = KeystoneAuth(fuel.url, creds)

        cluster_id = get_cluster_id(fuel_conn, fuel.openstack_env)
        cluster = reflect_cluster(fuel_conn, cluster_id)

        if ctx.fuel_version is None:
            ctx.fuel_version = FuelInfo(fuel_conn).get_version()
            ctx.storage.put(ctx.fuel_version, "fuel_version")

            logger.info("Found FUEL {0}".format(".".join(map(str, ctx.fuel_version))))
            openrc = cluster.get_openrc()

            if openrc:
                auth_url = cast(str, openrc['os_auth_url'])
                if ctx.fuel_version >= [8, 0] and auth_url.startswith("https://"):
                    logger.warning("Fixing FUEL 8.0 AUTH url - replace https://->http://")
                    auth_url = auth_url.replace("https", "http", 1)

                os_creds = OSCreds(name=cast(str, openrc['username']),
                                   passwd=cast(str, openrc['password']),
                                   tenant=cast(str, openrc['tenant_name']),
                                   auth_url=cast(str, auth_url),
                                   insecure=cast(bool, openrc['insecure']))

                ctx.fuel_openstack_creds = os_creds
            else:
                ctx.fuel_openstack_creds = None

            ctx.storage.put(list(ctx.fuel_openstack_creds), "fuel_os_creds")

        if discovery == 'metadata':
            logger.debug("Skip FUEL nodes  discovery due to discovery settings")
            return

        try:
            fuel_rpc = setup_rpc(connect(fuel_node_info),
                                 ctx.rpc_code,
                                 ctx.default_rpc_plugins,
                                 log_level=ctx.config.rpc_log_level)
        except AuthenticationException:
            msg = "FUEL nodes discovery failed - wrong FUEL master SSH credentials"
            if discovery != 'ignore_errors':
                raise StopTestError(msg)
            logger.warning(msg)
            return
        except Exception as exc:
            if discovery != 'ignore_errors':
                logger.exception("While connection to FUEL")
                raise StopTestError("Failed to connect to FUEL")
            logger.warning("Failed to connect to FUEL - %s", exc)
            return

        logger.debug("Downloading FUEL node ssh master key")
        fuel_key = fuel_rpc.get_file_content('/root/.ssh/id_rsa')
        network = 'fuelweb_admin' if ctx.fuel_version >= [6, 0] else 'admin'

        count = 0
        for count, fuel_node in enumerate(list(cluster.get_nodes())):
            ip = str(fuel_node.get_ip(network))
            ctx.merge_node(ConnCreds(to_ip(ip), "root", key=fuel_key), set(fuel_node.get_roles()))

        logger.debug("Found {} FUEL nodes for env {}".format(count, fuel.openstack_env))
