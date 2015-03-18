import node

from novaclient.client import Client


def get_floating_ip(vm):
    addrs = vm.addresses
    for net_name, ifaces in addrs.items():
        for iface in ifaces:
            if iface.get('OS-EXT-IPS:type') == "floating":
                return iface['addr']
    raise Exception("No floating ip found for VM %s" % repr(vm))


def discover_openstack_vms(conn_details):
    """Discover vms running in openstack
    :param conn_details - dict with openstack connection details -
        auth_url, api_key (password), username
    """
    client = Client(version='1.1', **conn_details)
    servers = client.servers.list(search_opts={"all_tenant": True})
    return [node.Node(get_floating_ip(server), ["test_vm"])
            for server in servers]


def discover_openstack_nodes(conn_details):
    """Discover openstack nodes
    :param conn_details - dict with openstack connection details -
        auth_url, api_key (password), username
    """
    client = Client(version='1.1', **conn_details)
    services = client.services.list()
    return [node.Node(server.ip, ["test_vm"]) for server in services]
