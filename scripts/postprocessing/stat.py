import sys
import time

from copy import deepcopy

import numpy
import scipy.optimize as scp
import matplotlib.pyplot as plt

import io_py_result_processor as io_test

key_pos = {'blocksize': 0, 'direct_io': 1, 'name': 2}
actions = ['randwrite', 'randread', 'read', 'write']
types = ['s', 'd']
colors = ['red', 'green', 'blue', 'cyan',
          'magenta', 'black', 'yellow', 'burlywood']


def get_key(x, no):
    """ x = (), no = key_pos key """
    keys = deepcopy(key_pos)
    del keys[no]
    key = [x[n] for n in keys.values()]
    return tuple(key), x[key_pos[no]]


def generate_groups(data, group_id):
    """ select data for plot by group_id
        data - processed_series"""
    grouped = {}

    for key, val in data.items():
        new_key, group_val = get_key(key, group_id)
        group = grouped.setdefault(new_key, {})
        group[group_val] = val

    return grouped


def gen_dots(val):
    """Generate dots from real data
       val = dict (x:y)
       return ox, oy lists """
    oy = []
    ox = []
    for x in sorted(val.keys()):
        ox.append(int(x[:-1]))
        if val[x][0] != 0:
            oy.append(1.0/val[x][0])
        else:
            oy.append(0)
    return ox, oy


def gen_line_numpy(x, y):
    A = numpy.vstack([x, numpy.ones(len(x))]).T
    coef = numpy.linalg.lstsq(A, y)[0]
    funcLine = lambda tpl, x: tpl[0] * x + tpl[1]
    print coef
    return x, funcLine(coef, x)


def gen_line_scipy(x, y):
    funcLine = lambda tpl, x: tpl[0] * x + tpl[1]
    ErrorFunc = lambda tpl, x, y: 1.0 - y/funcLine(tpl, x)
    tplInitial = (1.0, 0.0)
    # print x, y
    tplFinal, success = scp.leastsq(ErrorFunc, tplInitial[:], args=(x, y),
                                    diag=(1./x.mean(), 1./y.mean()))
    if success not in range(1, 4):
        raise ValueError("No line for this dots")
    xx = numpy.linspace(x.min(), x.max(), 50)
    print tplFinal
    # print x, ErrorFunc(tplFinal, x, y)
    return xx, funcLine(tplFinal, xx)


def gen_app_plot(key, val, plot, color):
    """ Plots with fake line and real dots around"""
    ox, oy = gen_dots(val)
    name = "_".join(str(k) for k in key)
    if len(ox) < 2:
        # skip single dots
        return False
    # create approximation
    x = numpy.array(ox)#numpy.log(ox))
    y = numpy.array(oy)#numpy.log(oy))
    print x, y
    try:
        print name
        x1, y1 = gen_line_scipy(x, y)
        plot.plot(x1, y1, color=color)
        # 
        #plot.loglog(x1, y1, color=color)
    except ValueError:
        # just don't draw it - it's ok
        # we'll see no appr and bad dots
        # not return False, because we need see dots
        pass
    plot.plot(x, y, '^', label=name, markersize=7, color=color)
    #plot.loglog(x, y, '^', label=name, markersize=7, color=color)
    return True


def save_plot(key, val):
    """ one plot from one dict item with value list"""
    ox, oy = gen_dots(val)
    name = "_".join(str(k) for k in key)
    plt.plot(ox, oy, label=name)


def plot_generation(fname, group_by):
    """ plots for value group_by in imgs by actions"""
    data = list(io_test.load_io_py_file(fname))
    item = io_test.Data("hdr")
    for key, vals in io_test.groupby_globally(data, io_test.key_func).items():
        item.series[key] = [val['iops'] for val in vals]
    io_test.process_inplace(item)

    pr_data = generate_groups(item.processed_series, group_by)
    print pr_data

    #fig = plt.figure()
    plot = plt.subplot(111)

    for action in actions:
        for tp in types:
            color = 0
            hasPlot = False
            for key, val in pr_data.items():
                if action in key and tp in key:
                    ok = gen_app_plot(key, val, plot, colors[color])
                    hasPlot = hasPlot or ok
                    color += 1
                    # use it for just connect dots
                    #save_plot(key, val)
            if hasPlot:
                # Shrink current axis by 10%
                box = plot.get_position()
                plot.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

                # Put a legend to the bottom
                plot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                            fancybox=True, shadow=True, ncol=4,
                            fontsize='xx-small')
                plt.title("Plot for %s on %s" % (group_by, action))
                plt.ylabel("time")
                plt.xlabel(group_by)
                plt.grid()
                # use it if want scale plot somehow
                # plt.axis([0.0, 5000.0, 0.0, 64.0])
                name = "%s__%s_%s.png" % (group_by, action, tp)
                plt.savefig(name, format='png', dpi=100)
            plt.clf()
            plot = plt.subplot(111)
            color = 0


def deviation_on_deviation(groups_list, data):
    """ calc deviation of data all and by selection groups"""
    total_dev = io_test.round_deviation(io_test.med_dev(data))
    grouped_dev = [total_dev]
    for group in groups_list:
        beg = 0
        end = group
        local_dev = []
        while end <= len(data):
            local_dev.append(io_test.round_deviation(io_test.med_dev(data[beg:end]))[0])
            beg += group
            end += group
        grouped_dev.append(io_test.round_deviation(io_test.med_dev(local_dev)))
    return grouped_dev



def deviation_generation(fname, groups_list):
    """ Print deviation by groups for data from fname """
    CONC_POS = key_pos['concurence']
    int_list = [int(i) for i in groups_list]
    data = list(io_test.load_io_py_file(fname))
    item = io_test.Data("hdr")
    for key, vals in io_test.groupby_globally(data, io_test.key_func).items():
        item.series[key] = [val['iops'] * key[CONC_POS] for val in vals]
        print deviation_on_deviation(int_list, item.series[key])


def main(argv):
    if argv[1] == "plot":
        plot_generation(argv[2], argv[3])
    elif argv[1] == "dev":
        deviation_generation(argv[2], argv[3:])
    

if __name__ == "__main__":
    exit(main(sys.argv))




