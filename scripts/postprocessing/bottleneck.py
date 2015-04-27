""" Analize test results for finding bottlenecks """

import sys
import argparse

import texttable as TT

from collections import namedtuple


Record = namedtuple("Record", ['name', 'max_value'])
MetricValue = namedtuple("MetricValue", ['value', 'time'])
Bottleneck = namedtuple("Bottleneck", ['node', 'value', 'count'])

sortRuleByValue = lambda x: x.value
sortRuleByMaxValue = lambda x: x.max_value
sortRuleByCount = lambda x: x.count
sortRuleByTime = lambda x: x.time

critical_values = [
    Record("io_queue", 1),
    Record("procs_blocked", 1),
    Record("mem_usage_percent", 0.8)
    ]


def get_name_from_sourceid(source_id):
    """ Cut port """
    pos = source_id.rfind(":")
    return source_id[:pos]


def create_table(header, rows, signs=1):
    """ Return texttable view """
    tab = TT.Texttable()
    tab.set_deco(tab.VLINES)
    tab.set_precision(signs)
    tab.add_row(header)
    tab.header = header

    for row in rows:
        tab.add_row(row)

    return tab.draw()


def load_results(period, rfile):
    """ Read raw results from dir and return
        data from provided period"""
    results = {}
    if period is not None:
        begin_time, end_time = period
    with open(rfile, "r") as f:
        for line in f:

            if len(line) <= 1:
                continue
            if " : " in line:
                # old format
                ttime, _, raw_data = line.partition(" : ")
                raw_data = raw_data.strip('"\n\r')
                itime = float(ttime)
            else:
                # new format without time
                raw_data = line.strip('"\n\r')

            _, data = eval(raw_data)
            sid = get_name_from_sourceid(data.pop("source_id"))
            itime = data.pop("time")

            if period is None or (itime >= begin_time and itime <= end_time):
                serv_data = results.setdefault(sid, {})
                for key, value in data.items():
                    # select device and metric names
                    dev, _, metric = key.partition(".")
                    # create dict for metric
                    metric_dict = serv_data.setdefault(metric, {})
                    # set value for metric on dev
                    cur_val = metric_dict.setdefault(dev, [])
                    cur_val.append(MetricValue(value, itime))

        # sort by time
        for ms in results.values():
            for dev in ms.values():
                for d in dev.keys():
                    dev[d] = sorted(dev[d], key=sortRuleByTime)

        return results


def find_time_load_percent(data, params):
    """ Find avg load of components by time
        and return sorted table """

    header = ["Component", "Avg load %"]
    name_fmt = "{0}.{1}"
    value_fmt = "{0:.1f}"
    loads = []
    for node, metrics in data.items():
        for metric, max_value in params:
            if metric in metrics:
                item = metrics[metric]
                # count time it was > max_value
                # count times it was > max_value
                for dev, vals in item.items():
                    num_l = 0
                    times = []
                    i = 0
                    while i < len(vals):
                        if vals[i].value >= max_value:
                            num_l += 1
                            b_time = vals[i].time
                            while i < len(vals) and \
                                  vals[i].value >= max_value:
                                i += 1
                            times.append(vals[i-1].time - b_time)
                        i += 1
                    if num_l > 0:
                        avg_time = sum(times) / float(num_l)
                        total_time = vals[-1].time - vals[0].time
                        avg_load = (avg_time / total_time) * 100
                        loads.append(Record(name_fmt.format(node, dev), avg_load))

    rows = [[name, value_fmt.format(value)]
            for name, value in sorted(loads, key=sortRuleByMaxValue, reverse=True)]
    return create_table(header, rows)



def print_bottlenecks(data, params, max_bottlenecks=3):
    """ Print bottlenecks in table format,
        search in data by fields in params"""
    # all bottlenecks
    rows = []
    val_format = "{0}: {1}, {2} times it was >= {3}"

    # max_bottlenecks most slowests places
    # Record metric : [Bottleneck nodes (max 3)]
    max_values = {}

    for node, metrics in data.items():
        node_rows = []
        for metric, max_value in params:
            if metric in metrics:
                item = metrics[metric]
                # find max val for dev
                # count times it was > max_value
                for dev, vals in item.items():
                    num_l = 0
                    max_v = -1
                    for val in vals:
                        if val >= max_value:
                            num_l += 1
                            if max_v < val:
                                max_v = val
                    if num_l > 0:
                        key = Record(metric, max_value)
                        # add to most slowest
                        btnk = max_values.setdefault(key, [])
                        # just add all data at first
                        btnk.append(Bottleneck(node, max_v, num_l))
                         #add to common table
                        c_val = val_format.format(metric, max_v,
                                                  num_l, max_value)
                        node_rows.append([dev, c_val])
        if len(node_rows) > 0:
            rows.append([node, ""])
            rows.extend(node_rows)

    tab = TT.Texttable()
    #tab.set_deco(tab.VLINES)

    header = ["Server, device", "Critical value"]
    tab.add_row(header)
    tab.header = header

    for row in rows:
        tab.add_row(row)

    most_slowest_header = [metric for metric, max_value in max_values.keys()]
    most_slowest = []
    # select most slowest
    for metric, btnks in max_values.items():
        m_data = []
        worst = sorted(btnks, key=sortRuleByValue, reverse=True)[:max_bottlenecks]
        longest = sorted(btnks, key=sortRuleByCount, reverse=True)[:max_bottlenecks]
        m_data.append("{0} worst by value: ".format(max_bottlenecks))
        for btnk in worst:
            m_data.append(val_format.format(btnk.node, btnk.value,
                                                  btnk.count,
                                                  metric.max_value))
        m_data.append("{0} worst by times it was bad: ".format(max_bottlenecks))
        for btnk in longest:
            m_data.append(val_format.format(btnk.node, btnk.value,
                                                  btnk.count,
                                                  metric.max_value))
        most_slowest.append(m_data)


    rows2 = zip(*most_slowest)
    
    tab2 = TT.Texttable()
    #tab2.set_deco(tab.VLINES)

    tab2.add_row(most_slowest_header)
    tab2.header = most_slowest_header

    for row in rows2:
        tab2.add_row(row)
    return tab.draw(), tab2.draw()



def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time_period', nargs=2,
                        type=float, default=None,
                        help="Begin and end time for tests")
    parser.add_argument('-d', '--debug-ver', action='store_true',
                        help="Full report with original data")
    parser.add_argument('-u', '--user-ver', action='store_true',
                        default=True,
                        help="Avg load report")
    parser.add_argument('sensors_result', type=str,
                        default=None, nargs='?')
    return parser.parse_args(args[1:])


def main(argv):
    opts = parse_args(argv)

    results = load_results(opts.time_period, opts.sensors_result)

    if opts.debug_ver:
        tab_all, tab_max = print_bottlenecks(results, critical_values)
        print "Maximum values on provided metrics"
        print tab_max
        print "All loaded values"
        print tab_all

    else:
        print find_time_load_percent(results, critical_values)


if __name__ == "__main__":
    exit(main(sys.argv))
