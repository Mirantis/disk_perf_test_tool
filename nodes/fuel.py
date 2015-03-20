import node
import fuel_rest_api
import logging


logger = logging.getLogger("io-perf-tool")


def discover_fuel_nodes(root_url, credentials, roles):
    """Discover Fuel nodes"""
    connection = fuel_rest_api.KeystoneAuth(root_url, credentials)
    fi = fuel_rest_api.FuelInfo(connection)
    nodes = []
    for role in roles:
        nodes.extend(getattr(fi.nodes, role))
    logger.debug("Found %s fuel nodes" % len(fi.nodes))
    return [node.Node(n.ip, n.get_roles()) for n in nodes]