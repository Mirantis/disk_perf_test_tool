import logging
import socket
from typing import Dict, Any, Tuple, List, NamedTuple, Union, cast
from urllib.parse import urlparse

from .. import fuel_rest_api
from ..node_interfaces import NodeInfo, IRPCNode
from ..ssh_utils import ConnCreds
from ..utils import check_input_param

logger = logging.getLogger("wally.discover")


FuelNodeInfo = NamedTuple("FuelNodeInfo",
                          [("version", List[int]),
                           ("fuel_ext_iface", str),
                           ("openrc", Dict[str, Union[str, bool]])])


def discover_fuel_nodes(fuel_master_node: IRPCNode,
                        fuel_conn: fuel_rest_api.Connection,
                        fuel_data: Dict[str, Any],
                        discover_nodes: bool = True) -> Tuple[List[NodeInfo], FuelNodeInfo]:
    """Discover nodes in fuel cluster, get openrc for selected cluster"""

    msg = "openstack_env should be provided in fuel config"
    check_input_param('openstack_env' in fuel_data, msg)

    # get cluster information from REST API
    cluster_id = fuel_rest_api.get_cluster_id(fuel_conn, fuel_data['openstack_env'])
    cluster = fuel_rest_api.reflect_cluster(fuel_conn, cluster_id)
    version = fuel_rest_api.FuelInfo(fuel_conn).get_version()

    if not discover_nodes:
        logger.warning("Skip fuel cluster discovery")
        return [], FuelNodeInfo(version, None, cluster.get_openrc())  # type: ignore

    logger.info("Found fuel {0}".format(".".join(map(str, version))))

    # get FUEL master key to connect to cluster nodes via ssh
    logger.debug("Downloading fuel master key")
    fuel_key = fuel_master_node.get_file_content('/root/.ssh/id_rsa')

    network = 'fuelweb_admin' if version >= [6, 0] else 'admin'
    fuel_ip = socket.gethostbyname(fuel_conn.host)
    fuel_ext_iface = fuel_master_node.get_interface(fuel_ip)

    nodes = []
    for fuel_node in list(cluster.get_nodes()):
        ip = str(fuel_node.get_ip(network))
        creds = ConnCreds(ip, "root", key=fuel_key)
        nodes.append(NodeInfo(creds, roles=set(fuel_node.get_roles())))

    logger.debug("Found {} fuel nodes for env {}".format(len(nodes), fuel_data['openstack_env']))

    return nodes, FuelNodeInfo(version, fuel_ext_iface,
                               cast(Dict[str, Union[str, bool]], cluster.get_openrc()))

