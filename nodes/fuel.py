import node
import fuel_rest_api


def discover_fuel_nodes(root_url, credentials):
    """Discover Fuel nodes"""
    connection = fuel_rest_api.KeystoneAuth(root_url, credentials)
    fi = fuel_rest_api.FuelInfo(connection)

    return [node.Node(n.ip, n.get_roles()) for n in fi.nodes]