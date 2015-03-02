from urlparse import urlparse
from keystone import KeystoneAuth


def total_lab_info(data):
    # <koder>: give 'd' meaningful name
    d = {}
    d['nodes_count'] = len(data['nodes'])
    d['total_memory'] = 0
    d['total_disk'] = 0
    d['processor_count'] = 0

    for node in data['nodes']:
        d['total_memory'] += node['memory']['total']
        d['processor_count'] += len(node['processors'])

        for disk in node['disks']:
            d['total_disk'] += disk['size']

    to_gb = lambda x: x / (1024 ** 3)
    d['total_memory'] = format(to_gb(d['total_memory']), ',d')
    d['total_disk'] = format(to_gb(d['total_disk']), ',d')
    return d


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
    # result['name'] = 'Perf-1 Env'
    result['fuel_version'] = fuel_version['release']

    return result