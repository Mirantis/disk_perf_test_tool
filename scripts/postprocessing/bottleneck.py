""" Analize test results for finding bottlenecks """

import sys
import argparse

import texttable as TT

from collections import namedtuple


Record = namedtuple("Record", ['name', 'max_value'])

critical_values = [
    Record("io_queue", 1),
    Record("procs_blocked", 1),
    Record("mem_usage_percent", 0.8)
    ]

def load_results(begin_time, end_time, rfile):
    """ Read raw results from dir and return
        data from provided period"""
    results = {}

    with open(rfile, "r") as f:
        for line in f:

            if len(line) <= 1:
                continue
            ttime, _, raw_data = line.partition(" : ")
            raw_data = raw_data.strip('"\n\r')
            itime = float(ttime)
            if itime >= begin_time and itime <= end_time:
                addr, data = eval(raw_data)
                sid = data.pop("source_id")
                data.pop("time")
                serv = "{0}({1})".format(addr[0], sid)
                serv_data = results.setdefault(serv, {})
                for key, value in data.items():
                    # select device and metric names
                    dev, _, metric = key.partition(".")
                    # create dict for metric
                    metric_dict = serv_data.setdefault(metric, {})
                    # set value for metric on dev
                    cur_val = metric_dict.setdefault(dev, [])
                    cur_val.append(value)
        print results
        return results



def print_bottlenecks(data, params):
    """ Print bottlenecks in table format,
        search in data by fields in params"""
    tab = TT.Texttable()
    tab.set_deco(tab.VLINES)

    header = ["Server, device", "Critical value"]
    tab.add_row(header)
    tab.header = header

    rows = []
    val_format = "{0}: {1}, {2} times it was >= {3}"
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
                        c_val = val_format.format(metric, max_v,
                                                  num_l, max_value)
                        node_rows.append([dev, c_val])
        if len(node_rows) > 0:
            rows.append([node, ""])
            rows.extend(node_rows)

    for row in rows:
        tab.add_row(row)

    print tab.draw()



def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--time_period', nargs=2,
                        type=float,
                        help="Begin and end time for tests")
    parser.add_argument('sensors_result', type=str,
                        default=None, nargs='?')
    return parser.parse_args(args[1:])


def main(argv):
    opts = parse_args(argv)

    results = load_results(opts.time_period[0], opts.time_period[1], opts.sensors_result)

    print_bottlenecks(results, critical_values)

if __name__ == "__main__":
    exit(main(sys.argv))
