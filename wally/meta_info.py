from urlparse import urlparse
from keystone import KeystoneAuth


def total_lab_info(data):
    lab_data = {}
    lab_data['nodes_count'] = len(data['nodes'])
    lab_data['total_memory'] = 0
    lab_data['total_disk'] = 0
    lab_data['processor_count'] = 0

    for node in data['nodes']:
        lab_data['total_memory'] += node['memory']['total']
        lab_data['processor_count'] += len(node['processors'])

        for disk in node['disks']:
            lab_data['total_disk'] += disk['size']

    def to_gb(x):
        return x / (1024 ** 3)

    lab_data['total_memory'] = to_gb(lab_data['total_memory'])
    lab_data['total_disk'] = to_gb(lab_data['total_disk'])
    return lab_data


def collect_lab_data(url, cred):
    u = urlparse(url)
    keystone = KeystoneAuth(root_url=url, creds=cred, admin_node_ip=u.hostname)
    lab_info = keystone.do(method='get', path="/api/nodes")
    fuel_version = keystone.do(method='get', path="/api/version/")

    nodes = []
    result = {}

    for node in lab_info:
        # <koder>: give p,i,d,... vars meaningful names
        d = {}
        d['name'] = node['name']
        p = []
        i = []
        disks = []
        devices = []

        for processor in node['meta']['cpu']['spec']:
            p.append(processor)

        for iface in node['meta']['interfaces']:
            i.append(iface)

        m = node['meta']['memory'].copy()

        for disk in node['meta']['disks']:
            disks.append(disk)

        d['memory'] = m
        d['disks'] = disks
        d['devices'] = devices
        d['interfaces'] = i
        d['processors'] = p

        nodes.append(d)

    result['nodes'] = nodes
    result['fuel_version'] = fuel_version['release']

    return result
