import socket
import logging
from typing import Dict, Any, List


from novaclient.client import Client

from ..node_interfaces import NodeInfo
from ..config import ConfigBlock
from ..utils import parse_creds


logger = logging.getLogger("wally.discover")


def get_floating_ip(vm: Any) -> str:
    """Get VM floating IP address"""

    for net_name, ifaces in vm.addresses.items():
        for iface in ifaces:
            if iface.get('OS-EXT-IPS:type') == "floating":
                return iface['addr']

    raise ValueError("VM {} has no floating ip".format(vm))


def get_ssh_url(user: str, password: str, ip: str, key: str) -> str:
    """Get ssh connection URL from parts"""

    if password is not None:
        assert key is None, "Both key and password provided"
        return "ssh://{}:{}@{}".format(user, password, ip)
    else:
        assert key is not None, "None of key/password provided"
        return "ssh://{}@{}::{}".format(user, ip, key)


def discover_vms(client: Client, search_opts: Dict) -> List[NodeInfo]:
    """Discover virtual machines"""
    user, password, key = parse_creds(search_opts.pop('auth'))

    servers = client.servers.list(search_opts=search_opts)
    logger.debug("Found %s openstack vms" % len(servers))

    nodes = []  # type: List[NodeInfo]
    for server in servers:
        ip = get_floating_ip(server)
        nodes.append(NodeInfo(get_ssh_url(user, password, ip, key), roles={"test_vm"}))
    return nodes


def discover_services(client: Client, opts: Dict[str, Any]) -> List[NodeInfo]:
    """Discover openstack services for given cluster"""
    user, password, key = parse_creds(opts.pop('auth'))

    services = []
    if opts['service'] == "all":
        services = client.services.list()
    else:
        if isinstance(opts['service'], str):
            opts['service'] = [opts['service']]

        for s in opts['service']:
            services.extend(client.services.list(binary=s))

    host_services_mapping = {}  # type: Dict[str, [str]]

    for service in services:
        ip = socket.gethostbyname(service.host)
        host_services_mapping.get(ip, []).append(service.binary)

    logger.debug("Found %s openstack service nodes" %
                 len(host_services_mapping))

    nodes = []  # type: List[NodeInfo]
    for host, services in host_services_mapping.items():
        ssh_url = get_ssh_url(user, password, host, key)
        nodes.append(NodeInfo(ssh_url, services))

    return nodes


def discover_openstack_nodes(conn_details: Dict[str, str], conf: ConfigBlock) -> List[NodeInfo]:
    """Discover vms running in openstack
    conn_details - dict with openstack connection details -
        auth_url, api_key (password), username
    conf - test configuration object
    """
    client = Client(version='1.1', **conn_details)

    if conf.get('discover'):
        services_to_discover = conf['discover'].get('nodes')
        if services_to_discover:
            return discover_services(client, services_to_discover)

    return []
