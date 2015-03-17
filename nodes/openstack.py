from novaclient.v1_1 import client as novacl


def discover_openstack_nodes(conn_details):
    """Discover openstack nodes
    :param connection_details - dict with openstack connection details -
        auth_url, api_key (password), username
    """
    client = novacl.Client(**conn_details)
    servers = client.servers.list(search_opts={"all_tenant": True})
    return servers
