import logging
from typing import Dict, List, NamedTuple, Union, cast

from paramiko.ssh_exception import AuthenticationException

from .fuel_rest_api import get_cluster_id, reflect_cluster, FuelInfo, KeystoneAuth
from .node_interfaces import NodeInfo
from .ssh_utils import ConnCreds, parse_ssh_uri
from .utils import check_input_param, StopTestError, parse_creds
from .stage import Stage, StepOrder
from .test_run_class import TestRun
from .node import connect, setup_rpc
from .config import ConfigBlock
from .openstack_api import OSCreds


logger = logging.getLogger("wally.discover")


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
        if 'fuel' in ctx.storage:
            ctx.nodes_info.extend(ctx.storage.load_list(NodeInfo, 'fuel/nodes'))
            ctx.fuel_openstack_creds = ctx.storage['fuel/os_creds']  # type: ignore
            ctx.fuel_version = ctx.storage['fuel/version']  # type: ignore
        else:
            fuel = ctx.config.fuel
            discover_nodes = (fuel.discover != "fuel_openrc_only")
            fuel_node_info = NodeInfo(parse_ssh_uri(fuel.ssh_creds), {'fuel_master'})
            fuel_nodes = [fuel_node_info]

            creds = dict(zip(("user", "passwd", "tenant"), parse_creds(fuel.creds)))
            fuel_conn = KeystoneAuth(fuel.url, creds)

            # get cluster information from REST API
            cluster_id = get_cluster_id(fuel_conn, fuel.openstack_env)
            cluster = reflect_cluster(fuel_conn, cluster_id)
            ctx.fuel_version = FuelInfo(fuel_conn).get_version()
            logger.info("Found fuel {0}".format(".".join(map(str, ctx.fuel_version))))
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

            if discover_nodes:

                try:
                    fuel_rpc = setup_rpc(connect(fuel_node_info), ctx.rpc_code)
                except AuthenticationException:
                    raise StopTestError("Wrong fuel credentials")
                except Exception:
                    logger.exception("While connection to FUEL")
                    raise StopTestError("Failed to connect to FUEL")

                logger.debug("Downloading FUEL node ssh master key")
                fuel_key = fuel_rpc.get_file_content('/root/.ssh/id_rsa')
                network = 'fuelweb_admin' if ctx.fuel_version >= [6, 0] else 'admin'

                for fuel_node in list(cluster.get_nodes()):
                    ip = str(fuel_node.get_ip(network))
                    fuel_nodes.append(NodeInfo(ConnCreds(ip, "root", key=fuel_key),
                                               roles=set(fuel_node.get_roles())))

                ctx.storage['fuel_nodes'] = fuel_nodes
                ctx.nodes_info.extend(fuel_nodes)
                ctx.nodes_info.append(fuel_node_info)
                logger.debug("Found {} FUEL nodes for env {}".format(len(fuel_nodes) - 1, fuel.openstack_env))
            else:
                logger.debug("Skip FUEL nodes  discovery, as 'fuel_openrc_only' is set to fuel.discover option")

            ctx.storage["fuel/nodes"] = fuel_nodes
            ctx.storage["fuel/os_creds"] = ctx.fuel_openstack_creds
            ctx.storage["fuel/version"] = ctx.fuel_version
