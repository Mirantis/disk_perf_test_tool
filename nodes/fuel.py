import logging


from node import Node
import fuel_rest_api


logger = logging.getLogger("io-perf-tool")


def discover_fuel_nodes(root_url, credentials, cluster_name):
    """Discover Fuel nodes"""
    assert credentials.count(':') >= 2
    user, passwd_tenant = credentials.split(":", 1)
    passwd, tenant = passwd_tenant.rsplit(":", 1)
    creds = dict(
        username=user,
        password=passwd,
        tenant_name=tenant,
    )

    connection = fuel_rest_api.KeystoneAuth(root_url, creds)
    fi = fuel_rest_api.FuelInfo(connection)

    clusters_id = fuel_rest_api.get_cluster_id(connection, cluster_name)

    nodes = []

    for node in fi.nodes:
        if node.cluster == clusters_id:
            nodes.append(node)
    res = [Node(n.ip, n.get_roles()) for n in nodes]
    logger.debug("Found %s fuel nodes for env %r" % (len(res), cluster_name))
    return res
