import sys

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval

from disk_perf_test_tool.tests.io_results_loader import load_data, filter_data


def linearity_plot(plt, data, types):
    fields = 'blocksize_b', 'iops_mediana', 'iops_stddev'

    names = {}
    for tp1 in ('rand', 'seq'):
        for oper in ('read', 'write'):
            for sync in ('sync', 'direct', 'async'):
                sq = (tp1, oper, sync)
                name = "{0} {1} {2}".format(*sq)
                names["".join(word[0] for word in sq)] = name

    for tp in types:
        filtered_data = filter_data('linearity_test_' + tp, fields)
        x = []
        y = []
        e = []

        for sz, med, dev in sorted(filtered_data(data)):
            iotime_ms = 1000. // med
            iotime_max = 1000. // (med - dev * 3)

            x.append(sz / 1024)
            y.append(iotime_ms)
            e.append(iotime_max - iotime_ms)

        plt.errorbar(x, y, e, linestyle='None', marker=names[tp])
    plt.legend(loc=2)


def th_plot(data, tt):
    fields = 'concurence', 'iops_mediana', 'lat_mediana'
    conc_4k = filter_data('concurrence_test_' + tt, fields, blocksize='4k')
    filtered_data = sorted(list(conc_4k(data)))

    x, iops, lat = zip(*filtered_data)

    _, ax1 = plt.subplots()

    xnew = np.linspace(min(x), max(x), 50)
    # plt.plot(xnew, power_smooth, 'b-', label='iops')
    ax1.plot(x, iops, 'b*')

    for degree in (3,):
        c = chebfit(x, iops, degree)
        vals = chebval(xnew, c)
        ax1.plot(xnew, vals, 'g--')

    # ax1.set_xlabel('thread count')
    # ax1.set_ylabel('iops')

    # ax2 = ax1.twinx()
    # lat = [i / 1000 for i in lat]
    # ax2.plot(x, lat, 'r*')

    # tck = splrep(x, lat, s=0.0)
    # power_smooth = splev(xnew, tck)
    # ax2.plot(xnew, power_smooth, 'r-', label='lat')

    # xp = xnew[0]
    # yp = power_smooth[0]
    # for _x, _y in zip(xnew[1:], power_smooth[1:]):
    #     if _y >= 100:
    #         xres = (_y - 100.) / (_y - yp) * (_x - xp) + xp
    #         ax2.plot([xres, xres], [min(power_smooth), max(power_smooth)], 'g--')
    #         break
    #     xp = _x
    #     yp = _y

    # ax2.plot([min(x), max(x)], [20, 20], 'g--')
    # ax2.plot([min(x), max(x)], [100, 100], 'g--')

    # ax2.set_ylabel("lat ms")
    # plt.legend(loc=2)


def main(argv):
    data = list(load_data(open(argv[1]).read()))
    # linearity_plot(data)
    th_plot(data, 'rws')
    # th_plot(data, 'rrs')
    plt.show()


if __name__ == "__main__":
    exit(main(sys.argv))
