""" Analize test results for finding bottlenecks """

import sys
import os.path
import argparse
import collections


import yaml


from wally.utils import b2ssize


class SensorsData(object):
    def __init__(self, source_id, hostname, ctime, values):
        self.source_id = source_id
        self.hostname = hostname
        self.ctime = ctime
        self.values = values  # [((dev, sensor), value)]


def load_results(fd):
    res = []
    source_id2nostname = {}

    for line in fd:
        line = line.strip()
        if line != "":
            _, data = eval(line)
            ctime = data.pop('time')
            source_id = data.pop('source_id')
            hostname = data.pop('hostname')

            data = [(k.split('.'), v) for k, v in data.items()]

            sd = SensorsData(source_id, hostname, ctime, data)
            res.append((ctime, sd))
            source_id2nostname[source_id] = hostname

    res.sort(key=lambda x: x[0])
    return res, source_id2nostname


critical_values = dict(
    io_queue=1,
    mem_usage_percent=0.8)


class SensorInfo(object):
    def __init__(self, name, native_ext, to_bytes_coef):
        self.name = name
        self.native_ext = native_ext
        self.to_bytes_coef = to_bytes_coef

SINFO = [
    SensorInfo('recv_bytes', 'B', 1),
    SensorInfo('send_bytes', 'B', 1),
    SensorInfo('sectors_written', 'Sect', 512),
    SensorInfo('sectors_read', 'Sect', 512),
]


SINFO_MAP = dict((sinfo.name, sinfo) for sinfo in SINFO)


class AggregatedData(object):
    def __init__(self, sensor_name):
        self.sensor_name = sensor_name

        # (node, device): count
        self.per_device = collections.defaultdict(lambda: 0)

        # node: count
        self.per_node = collections.defaultdict(lambda: 0)

        # role: count
        self.per_role = collections.defaultdict(lambda: 0)

        # (role_or_node, device_or_*): count
        self.all_together = collections.defaultdict(lambda: 0)

    def __str__(self):
        res = "<AggregatedData({0})>\n".format(self.sensor_name)
        for (role_or_node, device), val in self.all_together.items():
            res += "    {0}:{1} = {2}\n".format(role_or_node, device, val)
        return res


def total_consumption(sensors_data, roles_map):
    result = {}

    for _, item in sensors_data:
        for (dev, sensor), val in item.values:

            try:
                ad = result[sensor]
            except KeyError:
                ad = result[sensor] = AggregatedData(sensor)

            ad.per_device[(item.hostname, dev)] += val

    for ad in result.values():
        for (hostname, dev), val in ad.per_device.items():
            ad.per_node[hostname] += val

            for role in roles_map[hostname]:
                ad.per_role[role] += val

            ad.all_together[(hostname, dev)] = val

        for role, val in ad.per_role.items():
            ad.all_together[(role, '*')] = val

        for node, val in ad.per_node.items():
            ad.all_together[(node, '*')] = val

    return result


def avg_load(data):
    load = {}

    min_time = 0xFFFFFFFFFFF
    max_time = 0

    for tm, item in data:

        min_time = min(min_time, item.ctime)
        max_time = max(max_time, item.ctime)

        for name, max_val in critical_values.items():
            for (dev, sensor), val in item.values:
                if sensor == name and val > max_val:
                    load[(item.hostname, dev, sensor)] += 1
    return load, max_time - min_time


def print_bottlenecks(data_iter, max_bottlenecks=15):
    load, duration = avg_load(data_iter)
    rev_items = ((v, k) for (k, v) in load.items())

    res = sorted(rev_items, reverse=True)[:max_bottlenecks]

    max_name_sz = max(len(name) for _, name in res)
    frmt = "{{0:>{0}}} | {{1:>4}}".format(max_name_sz)
    table = [frmt.format("Component", "% times load > 100%")]

    for (v, k) in res:
        table.append(frmt.format(k, int(v * 100.0 / duration + 0.5)))

    return "\n".join(table)


def print_consumption(agg, roles, min_transfer=0):
    rev_items = []
    for (node_or_role, dev), v in agg.all_together.items():
        rev_items.append((int(v), node_or_role + ':' + dev))

    res = sorted(rev_items, reverse=True)
    sinfo = SINFO_MAP[agg.sensor_name]

    if sinfo.to_bytes_coef is not None:
        res = [(v, k)
               for (v, k) in res
               if v * sinfo.to_bytes_coef >= min_transfer]

    if len(res) == 0:
        return None

    res = [(b2ssize(v) + sinfo.native_ext, k) for (v, k) in res]

    max_name_sz = max(len(name) for _, name in res)
    max_val_sz = max(len(val) for val, _ in res)

    frmt = " {{0:>{0}}} | {{1:>{1}}} ".format(max_name_sz, max_val_sz)
    table = [frmt.format("Component", "Usage")]

    for (v, k) in res:
        table.append(frmt.format(k, v))

    return "\n".join(table)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time_period', nargs=2,
                        type=int, default=None,
                        help="Begin and end time for tests")
    parser.add_argument('-m', '--max-bottlenek', type=int,
                        default=15, help="Max bottlenek to show")
    parser.add_argument('-d', '--debug-ver', action='store_true',
                        help="Full report with original data")
    parser.add_argument('-u', '--user-ver', action='store_true',
                        default=True,
                        help="Avg load report")
    parser.add_argument('results_folder')
    return parser.parse_args(args[1:])


def make_roles_mapping(source_id_mapping, source_id2hostname):
    result = {}
    for ssh_url, roles in source_id_mapping.items():
        if '@' in ssh_url:
            source_id = ssh_url.split('@')[1]
        else:
            source_id = ssh_url.split('://')[1]

        if source_id.count(':') == 2:
            source_id = source_id.rsplit(":", 1)[0]

        if source_id.endswith(':'):
            source_id += "22"

        if source_id in source_id2hostname:
            result[source_id] = roles
            result[source_id2hostname[source_id]] = roles

    for testnode_src in (set(source_id2hostname) - set(result)):
        result[testnode_src] = ['testnode']
        result[source_id2hostname[testnode_src]] = ['testnode']

    return result


def get_testdata_size(consumption):
    max_data = 0
    for sensor_name, agg in consumption.items():
        if sensor_name in SINFO_MAP:
            tb = SINFO_MAP[sensor_name].to_bytes_coef
            if tb is not None:
                max_data = max(max_data, agg.per_role.get('testnode', 0) * tb)
    return max_data


def main(argv):
    opts = parse_args(argv)

    sensors_data_fname = os.path.join(opts.results_folder,
                                      'sensor_storage.txt')

    roles_file = os.path.join(opts.results_folder,
                              'nodes.yaml')

    src2roles = yaml.load(open(roles_file))

    with open(sensors_data_fname) as fd:
        data, source_id2hostname = load_results(fd)

    roles_map = make_roles_mapping(src2roles, source_id2hostname)

    # print print_bottlenecks(data, opts.max_bottlenek)
    # print print_bottlenecks(data, opts.max_bottlenek)

    consumption = total_consumption(data, roles_map)

    testdata_sz = get_testdata_size(consumption) // 1024
    for name in ('recv_bytes', 'send_bytes',
                 'sectors_read', 'sectors_written'):
        table = print_consumption(consumption[name], roles_map, testdata_sz)
        if table is None:
            print "Consumption of", name, "is negligible"
        else:
            ln = max(map(len, table.split('\n')))
            print '-' * ln
            print name.center(ln)
            print '-' * ln
            print table
            print '-' * ln
            print


if __name__ == "__main__":
    exit(main(sys.argv))
