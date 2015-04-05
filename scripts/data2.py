import sys
from data_stat import med_dev, round_deviation, groupby_globally
from data_stat import read_data_agent_result


def key(x):
    return (x['__meta__']['blocksize'],
            'd' if x['__meta__']['direct_io'] else 's',
            x['__meta__']['action'],
            x['__meta__']['concurence'])


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
    data = read_data_agent_result(sys.argv[1])
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
