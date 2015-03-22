import sys
import math
import itertools


def key(x):
    return (x['__meta__']['blocksize'],
            'd' if x['__meta__']['direct_io'] else 's',
            x['__meta__']['action'],
            x['__meta__']['concurence'])


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


def main(argv):
    data = []

    with open(argv[1]) as fc:
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

    grouped = groupby_globally(data, key)

    print template.format(**headers)

    for (bs, cache_tp, act, conc), curr_data in sorted(grouped.items()):
        iops = med_dev([i['iops'] * int(conc) for i in curr_data])
        bw_mean = med_dev([i['bw_mean'] * int(conc) for i in curr_data])
        lat = med_dev([i['lat'] / 1000 for i in curr_data])

        iops = round_deviation(iops)
        bw_mean = round_deviation(bw_mean)
        lat = round_deviation(lat)

        params = dict(
            bs=bs,
            action=act,
            cache_tp=cache_tp,
            iops=iops,
            bw=bw_mean,
            lat=lat,
            conc=conc
        )

        print template.format(**params)


if __name__ == "__main__":
    exit(main(sys.argv))

    # vals = [(123, 23), (125678, 5678), (123.546756, 23.77),
    #         (123.546756, 102.77), (0.1234, 0.0224),
    #         (0.001234, 0.000224), (0.001234, 0.0000224)]
    # for val in :
    #     print val, "=>", round_deviation(val)
