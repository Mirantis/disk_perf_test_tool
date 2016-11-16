from typing import Any, Dict, Union, List
from .fuel_rest_api import KeystoneAuth, FuelInfo


def total_lab_info(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    lab_data = {'nodes_count': len(nodes),
                'total_memory': 0,
                'total_disk': 0,
                'processor_count': 0}  # type: Dict[str, int]

    for node in nodes:
        lab_data['total_memory'] += node['memory']['total']
        lab_data['processor_count'] += len(node['processors'])

        for disk in node['disks']:
            lab_data['total_disk'] += disk['size']

    lab_data['total_memory'] /= (1024 ** 3)
    lab_data['total_disk'] /= (1024 ** 3)

    return lab_data


def collect_lab_data(url: str, cred: Dict[str, str]) -> Dict[str, Union[List[Dict[str, str]], str]]:
    finfo = FuelInfo(KeystoneAuth(url, cred))

    nodes = []  # type: List[Dict[str, str]]
    result = {}

    for node in finfo.get_nodes():
        node_info = {
            'name': node['name'],
            'processors': [],
            'interfaces': [],
            'disks': [],
            'devices': [],
            'memory': node['meta']['memory'].copy()
        }

        for processor in node['meta']['cpu']['spec']:
            node_info['processors'].append(processor)

        for iface in node['meta']['interfaces']:
            node_info['interfaces'].append(iface)

        for disk in node['meta']['disks']:
            node_info['disks'].append(disk)

        nodes.append(node_info)

    result['nodes'] = nodes
    result['fuel_version'] = finfo.get_version()
    result['total_info'] = total_lab_info(nodes)
    return result
