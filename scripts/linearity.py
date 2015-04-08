import re
import sys
from collections import defaultdict

from disk_perf_test_tool.utils import ssize_to_b
from disk_perf_test_tool.tests import disk_test_agent
from disk_perf_test_tool.scripts.postprocessing import data_stat


def filter_data(data, *params, **filters):
    for result in data:
        for k, v in filters.items():
            if v == result.get(k):
                yield map(result.get, params)


raw_data = open(sys.argv[1]).read()
data = list(disk_test_agent.parse_output(raw_data))[0]

processed_data = defaultdict(lambda: [])
pref = len("linearity_test_rrd")

for key, val in data['res'].items():
    val['blocksize'] = key[pref:].split('th')[0]

    info = key[pref - 3:]
    sz = info[3:].split("th")[0]
    sinfo = info[:3]

    if val['iops'] != []:
        med, dev = map(int, data_stat.med_dev(val['iops']))
        sdata = "{0:>4} ~ {1:>2}".format(med, dev)
        processed_data[sinfo].append([sdata, sz, med, dev])
    else:
        processed_data[sinfo].append(["None", sz, "None", "None"])


def sort_func(x):
    return ssize_to_b(x[1])


for sinfo, iops_sz in sorted(processed_data.items()):
    for siops, sz, _, _ in sorted(iops_sz, key=sort_func):
        print "{0} {1:>6} {2}".format(sinfo, sz, siops)


import math
import matplotlib.pyplot as plt


prep = lambda x: x
max_xz = 10000000


def add_plt(plt, processed_data, flt, marker):
    x = []
    y = []
    e = []

    for sinfo, iops_sz in sorted(processed_data.items()):
        if sinfo == flt:
            for siops, sz, med, dev in sorted(iops_sz, key=sort_func):
                if ssize_to_b(sz) < max_xz:
                    iotime_us = 1000. // med
                    iotime_max = 1000. // (med - dev * 3)
                    x.append(prep(ssize_to_b(sz) / 1024))
                    y.append(prep(iotime_us))
                    e.append(prep(iotime_max) - prep(iotime_us))

    plt.errorbar(x, y, e, linestyle='None', marker=marker)
    plt.plot([x[0], x[-1]], [y[0], y[-1]])

add_plt(plt, processed_data, 'rwd', '*')
add_plt(plt, processed_data, 'rws', '^')
add_plt(plt, processed_data, 'rrd', '+')

plt.show()
