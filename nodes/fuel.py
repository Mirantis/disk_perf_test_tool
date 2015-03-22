import logging


import node
import fuel_rest_api
from disk_perf_test_tool.utils import parse_creds


logger = logging.getLogger("io-perf-tool")


def discover_fuel_nodes(root_url, credentials, roles):
    """Discover Fuel nodes"""
    user, passwd, tenant = parse_creds(credentials['creds'])

    creds = dict(
        username=user,
        password=passwd,
        tenant_name=tenant,
    )

    connection = fuel_rest_api.KeystoneAuth(root_url, creds)
    fi = fuel_rest_api.FuelInfo(connection)
    nodes = []
    for role in roles:
        nodes.extend(getattr(fi.nodes, role))
    logger.debug("Found %s fuel nodes" % len(fi.nodes))
    return [node.Node(n.ip, n.get_roles()) for n in nodes]
