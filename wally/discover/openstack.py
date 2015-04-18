import socket
import logging


from novaclient.client import Client

from .node import Node
from wally.utils import parse_creds


logger = logging.getLogger("io-perf-tool.discover")


def get_floating_ip(vm):
    addrs = vm.addresses
    for net_name, ifaces in addrs.items():
        for iface in ifaces:
            if iface.get('OS-EXT-IPS:type') == "floating":
                return iface['addr']


def discover_vms(client, search_opts):
    user, password, key = parse_creds(search_opts.pop('auth'))

    servers = client.servers.list(search_opts=search_opts)
    logger.debug("Found %s openstack vms" % len(servers))
    return [Node(get_floating_ip(server), ["test_vm"], username=user,
                 password=password, key_path=key)
            for server in servers if get_floating_ip(server)]


def discover_services(client, opts):
    user, password, key = parse_creds(opts.pop('auth'))

    services = []
    if opts['service'] == "all":
        services = client.services.list()
    else:
        if isinstance(opts['service'], basestring):
            opts['service'] = [opts['service']]

        for s in opts['service']:
            services.extend(client.services.list(binary=s))

    host_services_mapping = {}

    for service in services:
        ip = socket.gethostbyname(service.host)
        host_services_mapping[ip].append(service.binary)

    logger.debug("Found %s openstack service nodes" %
                 len(host_services_mapping))
    return [Node(host, services, username=user,
                 password=password, key_path=key) for host, services in
            host_services_mapping.items()]


def discover_openstack_nodes(conn_details, conf):
    """Discover vms running in openstack
    :param conn_details - dict with openstack connection details -
        auth_url, api_key (password), username
    """
    client = Client(version='1.1', **conn_details)
    nodes = []
    if conf.get('discover'):
        services_to_discover = conf['discover'].get('nodes')
        if services_to_discover:
            nodes.extend(discover_services(client, services_to_discover))

    return nodes
