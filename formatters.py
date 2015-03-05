import itertools
import json
import math


def get_formatter(test_type):
    if test_type == "iozone" or test_type == "fio":
        return format_io_stat
    elif test_type == "pgbench":
        return format_pgbench_stat
    else:
        raise Exception("Cannot get formatter for type %s" % test_type)


def format_io_stat(res):
    if len(res) != 0:
        bw_mean = 0.0
        for measurement in res:
            bw_mean += measurement["bw_mean"]

        bw_mean /= len(res)

        it = ((bw_mean - measurement["bw_mean"]) ** 2 for measurement in res)
        bw_dev = sum(it) ** 0.5

        meta = res[0]['__meta__']

        sync = meta['sync']
        direct = meta['direct_io']

        if sync and direct:
            ss = "d+"
        elif sync:
            ss = "s"
        elif direct:
            ss = "d"
        else:
            ss = "a"

        key = "{0} {1} {2} {3}k".format(meta['action'], ss,
                                        meta['concurence'],
                                        meta['blocksize'])

        data = json.dumps({key: (int(bw_mean), int(bw_dev))})

        return data


def format_pgbench_stat(res):
    """
    Receives results in format:
    "<num_clients> <num_transactions>: <tps>
     <num_clients> <num_transactions>: <tps>
     ....
    "
    """
    if res:
        data = {}
        grouped_res = itertools.groupby(res, lambda x: x[0])
        for key, group in grouped_res:
            results = list(group)
            sum_res = sum([r[1] for r in results])
            mean = sum_res/len(results)
            sum_sq = sum([(r[1] - mean) ** 2 for r in results])
            if len(results) > 1:
                dev = math.sqrt(sum_sq / (len(results) - 1))
            else:
                dev = 0
            data[key] = (mean, dev)
        return data

