""" Analize test results for finding bottlenecks """

import re
import sys
import csv
import time
import bisect
import os.path
import argparse
import collections


import yaml
import texttable

try:
    import pygraphviz as pgv
except ImportError:
    pgv = None

sys.path.append("/mnt/other/work/disk_perf_test_tool")
from wally.run_test import load_data_from
from wally.utils import b2ssize, b2ssize_10


class SensorInfo(object):
    def __init__(self, name, print_name, native_ext, to_bytes_coef):
        self.name = name
        self.print_name = print_name
        self.native_ext = native_ext
        self.to_bytes_coef = to_bytes_coef


_SINFO = [
    SensorInfo('recv_bytes', 'net_recv', 'B', 1),
    SensorInfo('send_bytes', 'net_send', 'B', 1),
    SensorInfo('sectors_written', 'hdd_write', 'Sect', 512),
    SensorInfo('sectors_read', 'hdd_read', 'Sect', 512),
    SensorInfo('reads_completed', 'read_op', 'OP', None),
    SensorInfo('writes_completed', 'write_op', 'OP', None),
    SensorInfo('procs_blocked', 'blocked_procs', 'P', None),
]

SINFO_MAP = dict((sinfo.name, sinfo) for sinfo in _SINFO)
to_bytes = dict((sinfo.name, sinfo.to_bytes_coef)
                for sinfo in _SINFO
                if sinfo.to_bytes_coef is not None)


class NodeSensorsData(object):
    def __init__(self, source_id, hostname, headers, values):
        self.source_id = source_id
        self.hostname = hostname
        self.headers = headers
        self.values = values
        self.times = None

    def finalize(self):
        self.times = [v[0] for v in self.values]

    def get_data_for_interval(self, beg, end):
        p1 = bisect.bisect_left(self.times, beg)
        p2 = bisect.bisect_right(self.times, end)

        obj = self.__class__(self.source_id,
                             self.hostname,
                             self.headers,
                             self.values[p1:p2])
        obj.times = self.times[p1:p2]
        return obj

    def __getitem__(self, name):
        idx = self.headers.index(name.split('.'))
        # +1 as first is a time
        return [val[idx] for val in self.values]


def load_results_csv(fd):
    data = fd.read()
    results = {}
    for block in data.split("NEW_DATA"):
        block = block.strip()
        if len(block) == 0:
            continue

        it = csv.reader(block.split("\n"))
        headers = next(it)
        sens_data = [map(float, vals) for vals in it]
        source_id, hostname = headers[:2]
        headers = [(None, 'time')] + \
                  [header.split('.') for header in headers[2:]]
        assert set(map(len, headers)) == set([2])

        results[source_id] = NodeSensorsData(source_id, hostname,
                                             headers, sens_data)

    return results


def load_test_timings(fname, max_diff=1000):
    raw_map = collections.defaultdict(lambda: [])

    class data(object):
        pass

    load_data_from(fname)(None, data)
    for test_type, test_results in data.results.items():
        if test_type == 'io':
            for tests_res in test_results:
                raw_map[tests_res.config.name].append(tests_res.run_interval)

    result = {}
    for name, intervals in raw_map.items():
        intervals.sort()
        curr_start, curr_stop = intervals[0]
        curr_result = []

        for (start, stop) in intervals[1:]:
            if abs(curr_start - start) < max_diff:
                # if abs(curr_stop - stop) > 2:
                #     print abs(curr_stop - stop)
                assert abs(curr_stop - stop) < max_diff
            else:
                assert start + max_diff >= curr_stop
                assert stop > curr_stop
                curr_result.append((curr_start, curr_stop))
                curr_start, curr_stop = start, stop
        curr_result.append((curr_start, curr_stop))

        merged_res = []
        curr_start, curr_stop = curr_result[0]
        for start, stop in curr_result[1:]:
            if abs(curr_stop - start) < max_diff:
                curr_stop = stop
            else:
                merged_res.append((curr_start, curr_stop))
                curr_start, curr_stop = start, stop
        merged_res.append((curr_start, curr_stop))
        result[name] = merged_res

    return result


critical_values = dict(
    io_queue=1,
    usage_percent=0.8,
    procs_blocked=1,
    procs_queue=1)


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

    for name, sensor_data in sensors_data.items():
        for pos, (dev, sensor) in enumerate(sensor_data.headers):

            if 'time' == sensor:
                continue

            try:
                ad = result[sensor]
            except KeyError:
                ad = result[sensor] = AggregatedData(sensor)

            val = sum(vals[pos] for vals in sensor_data.values)

            ad.per_device[(sensor_data.hostname, dev)] += val

    # vals1 = sensors_data['localhost:22']['sdc.sectors_read']
    # vals2 = sensors_data['localhost:22']['sdb.sectors_written']

    # from matplotlib import pyplot as plt
    # plt.plot(range(len(vals1)), vals1)
    # plt.plot(range(len(vals2)), vals2)
    # plt.show()
    # exit(1)

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


def avg_load(sensors_data):
    load = collections.defaultdict(lambda: 0)

    min_time = 0xFFFFFFFFFFF
    max_time = 0

    for sensor_data in sensors_data.values():

        min_time = min(min_time, min(sensor_data.times))
        max_time = max(max_time, max(sensor_data.times))

        for name, max_val in critical_values.items():
            for pos, (dev, sensor) in enumerate(sensor_data.headers):
                if sensor == name:
                    for vals in sensor_data.values:
                        if vals[pos] > max_val:
                            load[(sensor_data.hostname, dev, sensor)] += 1
    return load, max_time - min_time


def print_bottlenecks(sensors_data, max_bottlenecks=15):
    load, duration = avg_load(sensors_data)

    if not load:
        return "\n*** No bottlenecks found *** \n"

    rev_items = ((v, k) for (k, v) in load.items())

    res = sorted(rev_items, reverse=True)[:max_bottlenecks]

    max_name_sz = max(len(name) for _, name in res)
    frmt = "{{0:>{0}}} | {{1:>4}}".format(max_name_sz)
    table = [frmt.format("Component", "% times load > 100%")]

    for (v, k) in res:
        table.append(frmt.format(k, int(v * 100.0 / duration + 0.5)))

    return "\n".join(table)


def print_consumption(agg, min_transfer=None):
    rev_items = []
    for (node_or_role, dev), v in agg.all_together.items():
        rev_items.append((int(v), node_or_role + ':' + dev))

    res = sorted(rev_items, reverse=True)

    if min_transfer is not None:
        res = [(v, k)
               for (v, k) in res
               if v >= min_transfer]

    if len(res) == 0:
        return None

    res = [(b2ssize(v) + "B", k) for (v, k) in res]

    max_name_sz = max(len(name) for _, name in res)
    max_val_sz = max(len(val) for val, _ in res)

    frmt = " {{0:>{0}}} | {{1:>{1}}} ".format(max_name_sz, max_val_sz)
    table = [frmt.format("Component", "Usage")]

    for (v, k) in res:
        table.append(frmt.format(k, v))

    return "\n".join(table)


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
    for name, sens in SINFO_MAP.items():
        if sens.to_bytes_coef is not None:
            agg = consumption.get(name)
            if agg is not None:
                cdt = agg.per_role.get('testnode', 0) * sens.to_bytes_coef
                max_data = max(max_data, cdt)
    return max_data


def get_testop_cout(consumption):
    max_op = 0
    for name, sens in SINFO_MAP.items():
        if sens.to_bytes_coef is None:
            agg = consumption.get(name)
            if agg is not None:
                max_op = max(max_op, agg.per_role.get('testnode', 0))
    return max_op


def get_data_for_intervals(data, intervals):
    res = {}
    for begin, end in intervals:
        for name, node_data in data.items():
            ndata = node_data.get_data_for_interval(begin, end)
            res[name] = ndata
    return res


class Host(object):
    def __init__(self, name=None):
        self.name = name
        self.hdd_devs = {}
        self.net_devs = None


def plot_consumption(per_consumer_table, fields, refload):
    if pgv is None:
        return

    hosts = {}
    storage_sensors = ('sectors_written', 'sectors_read')

    for (hostname, dev), consumption in per_consumer_table.items():
        if hostname not in hosts:
            hosts[hostname] = Host(hostname)

        host = hosts[hostname]
        cons_map = dict(zip(fields, consumption))

        for sn in storage_sensors:
            vl = cons_map.get(sn, 0)
            if vl > 0:
                host.hdd_devs.setdefault(dev, {})[sn] = vl

    p = pgv.AGraph(name='system', directed=True)

    net = "Network"
    p.add_node(net)

    in_color = 'red'
    out_color = 'green'

    for host in hosts.values():
        g = p.subgraph(name="cluster_" + host.name, label=host.name,
                       color="blue")
        g.add_node(host.name, shape="diamond")
        p.add_edge(host.name, net)
        p.add_edge(net, host.name)

        for dev_name, values in host.hdd_devs.items():
            if dev_name == '*':
                continue

            to = values.get('sectors_written', 0)
            frm = values.get('sectors_read', 0)
            to_pw = 7 * to / refload
            frm_pw = 7 * frm / refload
            min_with = 0.1

            if to_pw > min_with or frm_pw > min_with:
                dev_fqn = host.name + "." + dev_name
                g.add_node(dev_fqn)

                if to_pw > min_with:
                    g.add_edge(host.name, dev_fqn,
                               label=b2ssize(to) + "B",
                               penwidth=to_pw,
                               fontcolor=out_color,
                               color=out_color)

                if frm_pw > min_with:
                    g.add_edge(dev_fqn, host.name,
                               label=b2ssize(frm) + "B",
                               penwidth=frm_pw,
                               color=in_color,
                               fontcolor=in_color)

    return p.string()


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time_period', nargs=2,
                        type=int, default=None,
                        help="Begin and end time for tests")
    parser.add_argument('-m', '--max-bottlenek', type=int,
                        default=15, help="Max bottleneck to show")
    parser.add_argument('-x', '--max-diff', type=int,
                        default=10, help="Max bottleneck to show in" +
                        "0.1% from test nodes summ load")
    parser.add_argument('-d', '--debug-ver', action='store_true',
                        help="Full report with original data")
    parser.add_argument('-u', '--user-ver', action='store_true',
                        default=True, help="Avg load report")
    parser.add_argument('-s', '--select-loads', nargs='*', default=[])
    parser.add_argument('-f', '--fields', nargs='*', default=[])
    parser.add_argument('results_folder')
    return parser.parse_args(args[1:])


def main(argv):
    opts = parse_args(argv)

    stor_dir = os.path.join(opts.results_folder, 'sensor_storage')
    data = {}
    source_id2hostname = {}

    csv_files = os.listdir(stor_dir)
    for fname in csv_files:
        assert re.match(r"\d+_\d+.csv$", fname)

    csv_files.sort(key=lambda x: int(x.split('_')[0]))

    for fname in csv_files:
        with open(os.path.join(stor_dir, fname)) as fd:
            for name, node_sens_data in load_results_csv(fd).items():
                if name in data:
                    assert data[name].hostname == node_sens_data.hostname
                    assert data[name].source_id == node_sens_data.source_id
                    assert data[name].headers == node_sens_data.headers
                    data[name].values.extend(node_sens_data.values)
                else:
                    data[name] = node_sens_data

    for nd in data.values():
        assert nd.source_id not in source_id2hostname
        source_id2hostname[nd.source_id] = nd.hostname
        nd.finalize()

    roles_file = os.path.join(opts.results_folder,
                              'nodes.yaml')

    src2roles = yaml.load(open(roles_file))

    timings = load_test_timings(opts.results_folder)

    roles_map = make_roles_mapping(src2roles, source_id2hostname)
    max_diff = float(opts.max_diff) / 1000

    fields = ('recv_bytes', 'send_bytes',
              'sectors_read', 'sectors_written',
              'reads_completed', 'writes_completed')

    if opts.fields != []:
        fields = [field for field in fields if field in opts.fields]

    for test_name, intervals in sorted(timings.items()):
        if opts.select_loads != []:
            if test_name not in opts.select_loads:
                continue

        data_chunks = get_data_for_intervals(data, intervals)

        consumption = total_consumption(data_chunks, roles_map)

        bottlenecks = print_bottlenecks(data_chunks)

        testdata_sz = get_testdata_size(consumption) * max_diff
        testop_count = get_testop_cout(consumption) * max_diff

        per_consumer_table = {}
        per_consumer_table_str = {}

        all_consumers = set()#consumption.values()[0].all_together)
        for value in consumption.values():
            all_consumers = all_consumers | set(value.all_together)
        fields = [field for field in fields if field in consumption]
        all_consumers_sum = []

        for consumer in all_consumers:
            tb_str = per_consumer_table_str[consumer] = []
            tb = per_consumer_table[consumer] = []
            vl = 0
            for name in fields:
                val = consumption[name].all_together[consumer]
                if SINFO_MAP[name].to_bytes_coef is None:
                    if val < testop_count:
                        tb_str.append('0')
                    else:
                        tb_str.append(b2ssize_10(int(val)))
                else:
                    val = int(val) * SINFO_MAP[name].to_bytes_coef
                    if val < testdata_sz:
                        tb_str.append('-')
                    else:
                        tb_str.append(b2ssize(val) + "B")
                tb.append(int(val))
                vl += int(val)
            all_consumers_sum.append((vl, consumer))

        all_consumers_sum.sort(reverse=True)

        plot_consumption(per_consumer_table, fields,
                         testdata_sz / max_diff)

        tt = texttable.Texttable(max_width=130)
        tt.set_cols_align(["l"] + ["r"] * len(fields))

        header = ["Name"]
        for fld in fields:
            if fld in SINFO_MAP:
                header.append(SINFO_MAP[fld].print_name)
            else:
                header.append(fld)
        tt.header(header)

        for summ, consumer in all_consumers_sum:
            if summ > 0:
                tt.add_row([":".join(consumer)] +
                           per_consumer_table_str[consumer])

        tt.set_deco(texttable.Texttable.VLINES | texttable.Texttable.HEADER)
        res = tt.draw()
        max_len = max(map(len, res.split("\n")))
        print test_name.center(max_len)
        print res
        print bottlenecks


if __name__ == "__main__":
    exit(main(sys.argv))
