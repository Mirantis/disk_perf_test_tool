import sys
import math
import itertools

from colorama import Fore, Style


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

    for (bs, cache_tp, act), curr_data_it in grouped_iter:
        key = (bs, cache_tp, act)
        grouped.setdefault(key, []).extend(curr_data_it)

    return grouped


class Data(object):
    def __init__(self, name):
        self.name = name
        self.series = {}
        self.processed_series = {}


def process_inplace(data):
    processed = {}
    for key, values in data.series.items():
        processed[key] = round_deviation(med_dev(values))
    data.processed_series = processed


def diff_table(*datas):
    res_table = {}

    for key in datas[0].processed_series:
        baseline = datas[0].processed_series[key]
        base_max = baseline[0] + baseline[1]
        base_min = baseline[0] - baseline[1]

        res_line = [baseline]

        for data in datas[1:]:
            val, dev = data.processed_series[key]
            val_min = val - dev
            val_max = val + dev

            diff_1 = int(float(val_min - base_max) / base_max * 100)
            diff_2 = int(float(val_max - base_min) / base_max * 100)

            diff_max = max(diff_1, diff_2)
            diff_min = min(diff_1, diff_2)

            res_line.append((diff_max, diff_min))
        res_table[key] = res_line

    return [data.name for data in datas], res_table


def print_table(headers, table):
    lines = []
    items = sorted(table.items())
    lines.append([(len(i), i) for i in [""] + headers])
    item_frmt = "{0}{1:>4}{2} ~ {3}{4:>4}{5}"

    for key, vals in items:
        ln1 = "{0:>4} {1} {2:>9} {3}".format(*map(str, key))
        ln2 = "{0:>4} ~ {1:>3}".format(*vals[0])

        line = [(len(ln1), ln1), (len(ln2), ln2)]

        for idx, val in enumerate(vals[1:], 2):
            cval = []
            for vl in val:
                if vl < -10:
                    cval.extend([Fore.RED, vl, Style.RESET_ALL])
                elif vl > 10:
                    cval.extend([Fore.GREEN, vl, Style.RESET_ALL])
                else:
                    cval.extend(["", vl, ""])

            ln = len(item_frmt.format("", cval[1], "", "", cval[4], ""))
            line.append((ln, item_frmt.format(*cval)))

        lines.append(line)

    max_columns_with = []
    for idx in range(len(lines[0])):
        max_columns_with.append(
            max(line[idx][0] for line in lines))

    sep = '-' * (4 + sum(max_columns_with) + 3 * (len(lines[0]) - 1))

    print sep
    for idx, line in enumerate(lines):
        cline = []
        for (curr_len, txt), exp_ln in zip(line, max_columns_with):
            cline.append(" " * (exp_ln - curr_len) + txt)
        print "| " + " | ".join(cline) + " |"
        if 0 == idx:
            print sep
    print sep


def key_func(x):
    return (x['__meta__']['blocksize'],
            'd' if 'direct' in x['__meta__'] else 's',
            x['__meta__']['name'])


template = "{bs:>4}  {action:>12}  {cache_tp:>3}  {conc:>4}"
template += " | {iops[0]:>6} ~ {iops[1]:>5} | {bw[0]:>7} ~ {bw[1]:>6}"
template += " | {lat[0]:>6} ~ {lat[1]:>5} |"

headers = dict(bs="BS",
               action="operation",
               cache_tp="S/D",
               conc="CONC",
               iops=("IOPS", "dev"),
               bw=("BW kBps", "dev"),
               lat=("LAT ms", "dev"))


def load_io_py_file(fname):
    with open(fname) as fc:
        block = None
        for line in fc:
            if line.startswith("{"):
                block = line
            elif block is not None:
                block += line

            if block is not None and block.count('}') == block.count('{'):
                cut = block.rfind('}')
                block = block[0:cut+1]
                yield eval(block)
                block = None

    if block is not None and block.count('}') == block.count('{'):
        yield eval(block)


def main(argv):
    items = []
    CONC_POS = 3
    for hdr_fname in argv[1:]:
        hdr, fname = hdr_fname.split("=", 1)
        data = list(load_io_py_file(fname))
        item = Data(hdr)
        for key, vals in groupby_globally(data, key_func).items():
            item.series[key] = [val['iops'] * key[CONC_POS] for val in vals]
        process_inplace(item)
        items.append(item)

    print_table(*diff_table(*items))

    # print template.format(**headers)

    # for (bs, cache_tp, act, conc), curr_data in sorted(grouped.items()):
    #     iops = med_dev([i['iops'] * int(conc) for i in curr_data])
    #     bw_mean = med_dev([i['bw_mean'] * int(conc) for i in curr_data])
    #     lat = med_dev([i['lat'] / 1000 for i in curr_data])

    #     iops = round_deviation(iops)
    #     bw_mean = round_deviation(bw_mean)
    #     lat = round_deviation(lat)

    #     params = dict(
    #         bs=bs,
    #         action=act,
    #         cache_tp=cache_tp,
    #         iops=iops,
    #         bw=bw_mean,
    #         lat=lat,
    #         conc=conc
    #     )

    #     print template.format(**params)


if __name__ == "__main__":
    exit(main(sys.argv))

    # vals = [(123, 23), (125678, 5678), (123.546756, 23.77),
    #         (123.546756, 102.77), (0.1234, 0.0224),
    #         (0.001234, 0.000224), (0.001234, 0.0000224)]
    # for val in :
    #     print val, "=>", round_deviation(val)
