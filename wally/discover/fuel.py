import socket
import logging
from typing import Dict, Any, Tuple, List
from urllib.parse import urlparse


from .. import fuel_rest_api
from ..utils import parse_creds, check_input_param
from ..node import NodeInfo, Node, FuelNodeInfo


logger = logging.getLogger("wally.discover")


def discover_fuel_nodes(fuel_master_node: Node,
                        fuel_data: Dict[str, Any],
                        discover_nodes: bool=True) -> Tuple[List[NodeInfo], FuelNodeInfo]:
    """Discover nodes in fuel cluster, get openrc for selected cluster"""

    # parse FUEL REST credentials
    username, tenant_name, password = parse_creds(fuel_data['creds'])
    creds = {"username": username,
             "tenant_name": tenant_name,
             "password": password}

    # connect to FUEL
    conn = fuel_rest_api.KeystoneAuth(fuel_data['url'], creds, headers=None)
    msg = "openstack_env should be provided in fuel config"
    check_input_param('openstack_env' in fuel_data, msg)

    # get cluster information from REST API
    cluster_id = fuel_rest_api.get_cluster_id(conn, fuel_data['openstack_env'])
    cluster = fuel_rest_api.reflect_cluster(conn, cluster_id)
    version = fuel_rest_api.FuelInfo(conn).get_version()

    if not discover_nodes:
        logger.warning("Skip fuel cluster discovery")
        return [], FuelNodeInfo(version, None, cluster.get_openrc())

    fuel_nodes = list(cluster.get_nodes())

    logger.info("Found FUEL {0}".format(".".join(map(str, version))))

    network = 'fuelweb_admin' if version >= [6, 0] else 'admin'

    fuel_host = urlparse(fuel_data['url']).hostname
    fuel_ip = socket.gethostbyname(fuel_host)
    fuel_ext_iface = fuel_master_node.get_interface(fuel_ip)

    # get FUEL master key to connect to cluster nodes via ssh
    logger.debug("Downloading fuel master key")
    fuel_key = fuel_master_node.get_file_content('/root/.ssh/id_rsa')

    # forward ports of cluster nodes to FUEL master
    logger.info("Forwarding ssh ports from FUEL nodes to localhost")
    ips = [str(fuel_node.get_ip(network)) for fuel_node in fuel_nodes]
    port_fw = [fuel_master_node.forward_port(ip, 22) for ip in ips]
    listen_ip = fuel_master_node.get_ip()

    nodes = []
    for port, fuel_node, ip in zip(port_fw, fuel_nodes, ips):
        logger.debug("SSH port forwarding {} => {}:{}".format(ip, listen_ip, port))
        conn_url = "ssh://root@{}:{}".format(listen_ip, port)
        nodes.append(NodeInfo(conn_url, fuel_node['roles'], listen_ip, fuel_key))

    logger.debug("Found {} fuel nodes for env {}".format(len(nodes), fuel_data['openstack_env']))

    return nodes, FuelNodeInfo(version, fuel_ext_iface, cluster.get_openrc())

