""" Analize test results for finding bottlenecks """

import sys
import csv
import time
import bisect
import os.path
import argparse
import collections


import yaml
import texttable


from wally.utils import b2ssize


class SensorsData(object):
    def __init__(self, source_id, hostname, ctime, values):
        self.source_id = source_id
        self.hostname = hostname
        self.ctime = ctime
        self.values = values  # [((dev, sensor), value)]


class SensorInfo(object):
    def __init__(self, name, native_ext, to_bytes_coef):
        self.name = name
        self.native_ext = native_ext
        self.to_bytes_coef = to_bytes_coef

_SINFO = [
    SensorInfo('recv_bytes', 'B', 1),
    SensorInfo('send_bytes', 'B', 1),
    SensorInfo('sectors_written', 'Sect', 512),
    SensorInfo('sectors_read', 'Sect', 512),
]

SINFO_MAP = dict((sinfo.name, sinfo) for sinfo in _SINFO)
to_bytes = dict((sinfo.name, sinfo.to_bytes_coef) for sinfo in _SINFO)


def load_results(fd):
    data = fd.read(100)
    fd.seek(0, os.SEEK_SET)

    # t = time.time()
    if '(' in data or '{' in data:
        res, source_id2nostname = load_results_eval(fd)
    else:
        res, source_id2nostname = load_results_csv(fd)

    # print int(((time.time() - t) * 1000000) / len(res)), len(res)

    return res, source_id2nostname


def load_results_csv(fd):

    fields = {}
    res = []
    source_id2nostname = {}
    coefs = {}

    # cached for performance
    ii = int
    zz = zip
    SD = SensorsData
    ra = res.append

    for row in csv.reader(fd):
        if len(row) == 0:
            continue
        ip, port = row[:2]
        ip_port = (ip, ii(port))

        if ip_port not in fields:
            sensors = [i.split('.') for i in row[4:]]
            fields[ip_port] = row[2:4] + sensors
            source_id2nostname[row[2]] = row[3]
            coefs[ip_port] = [to_bytes.get(s[1], 1) for s in sensors]
        else:
            fld = fields[ip_port]
            processed_data = []
            a = processed_data.append

            # this cycle is critical for performance
            # don't "refactor" it, unles you are confident
            # in what you are doing
            for dev_sensor, val, coef in zz(fld[2:], row[3:], coefs[ip_port]):
                a((dev_sensor, ii(val) * coef))

            ctime = ii(row[2])
            sd = SD(fld[0], fld[1], ctime, processed_data)
            ra((ctime, sd))

    res.sort(key=lambda x: x[0])
    return res, source_id2nostname


def load_results_eval(fd):
    res = []
    source_id2nostname = {}

    for line in fd:
        if line.strip() == "":
            continue

        _, data = eval(line)
        ctime = data.pop('time')
        source_id = data.pop('source_id')
        hostname = data.pop('hostname')

        processed_data = []
        for k, v in data.items():
            dev, sensor = k.split('.')
            processed_data.append(((dev, sensor),
                                   v * to_bytes.get(sensor, 1)))

        sd = SensorsData(source_id, hostname, ctime, processed_data)
        res.append((ctime, sd))
        source_id2nostname[source_id] = hostname

    res.sort(key=lambda x: x[0])
    return res, source_id2nostname


def load_test_timings(fd):
    result = {}  # test name - [(start_time, finish_time)]
    data = yaml.load(fd.read())
    assert len(data) == 1
    test_type, test_data = data[0]
    assert test_type == 'io'
    for test_names, interval in test_data['__meta__']['timings']:
        assert len(set(test_names)) == 1
        if test_names[0] not in result:
            result[test_names[0]] = interval
    return result


critical_values = dict(
    io_queue=1,
    mem_usage_percent=0.8)


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
            max_data = max(max_data, agg.per_role.get('testnode', 0))
    return max_data


def get_data_for_interval(data, interval):
    begin, end = interval
    times = [ctime for ctime, _ in data]
    b_p = bisect.bisect_left(times, begin)
    e_p = bisect.bisect_right(times, end)
    return data[b_p:e_p]


def main(argv):
    opts = parse_args(argv)

    sensors_data_fname = os.path.join(opts.results_folder,
                                      'sensor_storage.txt')

    roles_file = os.path.join(opts.results_folder,
                              'nodes.yaml')

    raw_results_file = os.path.join(opts.results_folder,
                                    'raw_results.yaml')

    src2roles = yaml.load(open(roles_file))
    timings = load_test_timings(open(raw_results_file))
    with open(sensors_data_fname) as fd:
        data, source_id2hostname = load_results(fd)

    roles_map = make_roles_mapping(src2roles, source_id2hostname)

    # print print_bottlenecks(data, opts.max_bottlenek)
    # print print_bottlenecks(data, opts.max_bottlenek)

    for name, interval in sorted(timings.items()):
        print
        print
        print "-" * 30 + " " + name + " " + "-" * 30
        print

        data_chunk = get_data_for_interval(data, interval)

        consumption = total_consumption(data_chunk, roles_map)

        testdata_sz = get_testdata_size(consumption) // 100

        fields = ('recv_bytes', 'send_bytes',
                  'sectors_read', 'sectors_written')
        per_consumer_table = {}

        all_consumers = set(consumption.values()[0].all_together)
        all_consumers_sum = []

        for consumer in all_consumers:
            tb = per_consumer_table[consumer] = []
            vl = 0
            for name in fields:
                val = consumption[name].all_together[consumer]
                if val < testdata_sz:
                    val = 0
                vl += int(val)
                tb.append(b2ssize(int(val)) + "B")
            all_consumers_sum.append((vl, consumer))

        all_consumers_sum.sort(reverse=True)
        tt = texttable.Texttable(max_width=130)
        tt.set_cols_align(["l"] + ["r"] * len(fields))
        tt.header(["Name"] + list(fields))

        for summ, consumer in all_consumers_sum:
            if summ > 0:
                tt.add_row([".".join(consumer)] +
                           [v if v != '0B' else '-'
                            for v in per_consumer_table[consumer]])

        tt.set_deco(texttable.Texttable.VLINES | texttable.Texttable.HEADER)
        print tt.draw()

        # if name in consumption:
        #     table = print_consumption(consumption[name], testdata_sz)
        #     if table is None:
        #         print "Consumption of", name, "is negligible"
        #     else:
        #         ln = max(map(len, table.split('\n')))
        #         print '-' * ln
        #         print name.center(ln)
        #         print '-' * ln
        #         print table
        #         print '-' * ln
        #         print

if __name__ == "__main__":
    exit(main(sys.argv))
