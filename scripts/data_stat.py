import math
import itertools


def med_dev(vals):
    med = sum(vals) / len(vals)
    dev = ((sum(abs(med - i) ** 2 for i in vals) / len(vals)) ** 0.5)
    return int(med), int(dev)


def round_deviation(med_dev):
    med, dev = med_dev

    if dev < 1E-7:
        return med_dev

    dev_div = 10.0 ** (math.floor(math.log10(dev)) - 1)
    dev = int(dev / dev_div) * dev_div
    med = int(med / dev_div) * dev_div
    return (type(med_dev[0])(med),
            type(med_dev[1])(dev))


def groupby_globally(data, key_func):
    grouped = {}
    grouped_iter = itertools.groupby(data, key_func)

    for (bs, cache_tp, act, conc), curr_data_it in grouped_iter:
        key = (bs, cache_tp, act, conc)
        grouped.setdefault(key, []).extend(curr_data_it)

    return grouped


def read_data_agent_result(fname):
    data = []
    with open(fname) as fc:
        block = None
        for line in fc:
            if line.startswith("{'__meta__':"):
                block = line
            elif block is not None:
                block += line

            if block is not None:
                if block.count('}') == block.count('{'):
                    data.append(eval(block))
                    block = None
    return data
