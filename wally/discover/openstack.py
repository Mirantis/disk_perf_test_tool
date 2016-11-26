import socket
import logging
from typing import Dict, Any, List, Optional, cast

from ..node_interfaces import NodeInfo
from ..config import ConfigBlock
from ..ssh_utils import ConnCreds
from ..start_vms import OSConnection, NovaClient


logger = logging.getLogger("wally.discover")


def get_floating_ip(vm: Any) -> str:
    """Get VM floating IP address"""

    for net_name, ifaces in vm.addresses.items():
        for iface in ifaces:
            if iface.get('OS-EXT-IPS:type') == "floating":
                return iface['addr']

    raise ValueError("VM {} has no floating ip".format(vm))


def discover_vms(client: NovaClient, search_data: str) -> List[NodeInfo]:
    """Discover virtual machines"""
    name, user, key_file = search_data.split(",")
    servers = client.servers.list(search_opts={"name": name})
    logger.debug("Found %s openstack vms" % len(servers))

    nodes = []  # type: List[NodeInfo]
    for server in servers:
        ip = get_floating_ip(server)
        creds = ConnCreds(host=ip, user=user, key_file=key_file)
        nodes.append(NodeInfo(creds, roles={"test_vm"}))

    return nodes


def discover_openstack_nodes(conn: OSConnection, conf: ConfigBlock) -> List[NodeInfo]:
    """Discover openstack services for given cluster"""
    os_nodes_auth = conf['auth']  # type: str

    if os_nodes_auth.count(":") == 2:
        user, password, key_file = os_nodes_auth.split(":")  # type: str, Optional[str], Optional[str]
        if not password:
            password = None
    else:
        user, password = os_nodes_auth.split(":")
        key_file = None

    services = conn.nova.services.list()  # type: List[Any]
    host_services_mapping = {}  # type: Dict[str, List[str]]

    for service in services:
        ip = cast(str, socket.gethostbyname(service.host))
        host_services_mapping.get(ip, []).append(service.binary)

    logger.debug("Found %s openstack service nodes" % len(host_services_mapping))

    nodes = []  # type: List[NodeInfo]
    for host, services in host_services_mapping.items():
        creds = ConnCreds(host=host, user=user, passwd=password, key_file=key_file)
        nodes.append(NodeInfo(creds, set(services)))

    return nodes
