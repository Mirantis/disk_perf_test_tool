import os
import csv
import bisect
import logging
import itertools
import collections
from cStringIO import StringIO

try:
    import numpy
    import scipy
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import wally
from wally.utils import ssize2b
from wally.statistic import round_3_digit
from wally.suits.io.fio_task_parser import (get_test_sync_mode,
                                            get_test_summary,
                                            parse_all_in_1,
                                            abbv_name_to_full)


logger = logging.getLogger("wally.report")


class DiskInfo(object):
    def __init__(self):
        self.direct_iops_r_max = 0
        self.direct_iops_w_max = 0

        # 64 used instead of 4k to faster feed caches
        self.direct_iops_w64_max = 0

        self.rws4k_10ms = 0
        self.rws4k_30ms = 0
        self.rws4k_100ms = 0
        self.bw_write_max = 0
        self.bw_read_max = 0


report_funcs = []


class Attrmapper(object):
    def __init__(self, dct):
        self.__dct = dct

    def __getattr__(self, name):
        try:
            return self.__dct[name]
        except KeyError:
            raise AttributeError(name)


class PerfInfo(object):
    def __init__(self, name, summary, intervals, params, testnodes_count):
        self.name = name
        self.bw = None
        self.iops = None
        self.lat = None
        self.lat_50 = None
        self.lat_95 = None

        self.raw_bw = []
        self.raw_iops = []
        self.raw_lat = []

        self.params = params
        self.intervals = intervals
        self.testnodes_count = testnodes_count
        self.summary = summary
        self.p = Attrmapper(self.params.vals)

        self.sync_mode = get_test_sync_mode(self.params)
        self.concurence = self.params.vals.get('numjobs', 1)


# disk_info = None
# base = None
# linearity = None


def group_by_name(test_data):
    name_map = collections.defaultdict(lambda: [])

    for data in test_data:
        name_map[(data.name, data.summary())].append(data)

    return name_map


def report(name, required_fields):
    def closure(func):
        report_funcs.append((required_fields.split(","), name, func))
        return func
    return closure


def get_test_lcheck_params(pinfo):
    res = [{
        's': 'sync',
        'd': 'direct',
        'a': 'async',
        'x': 'sync direct'
    }[pinfo.sync_mode]]

    res.append(pinfo.p.rw)

    return " ".join(res)


def get_emb_data_svg(plt):
    sio = StringIO()
    plt.savefig(sio, format='svg')
    img_start = "<!-- Created with matplotlib (http://matplotlib.org/) -->"
    return sio.getvalue().split(img_start, 1)[1]


def get_template(templ_name):
    very_root_dir = os.path.dirname(os.path.dirname(wally.__file__))
    templ_dir = os.path.join(very_root_dir, 'report_templates')
    templ_file = os.path.join(templ_dir, templ_name)
    return open(templ_file, 'r').read()


def group_by(data, func):
    if len(data) < 2:
        yield data
        return

    ndata = [(func(dt), dt) for dt in data]
    ndata.sort(key=func)
    pkey, dt = ndata[0]
    curr_list = [dt]

    for key, val in ndata[1:]:
        if pkey != key:
            yield curr_list
            curr_list = [val]
        else:
            curr_list.append(val)
        pkey = key

    yield curr_list


@report('linearity', 'linearity_test')
def linearity_report(processed_results, lab_info, comment):
    labels_and_data_mp = collections.defaultdict(lambda: [])
    vls = {}

    # plot io_time = func(bsize)
    for res in processed_results.values():
        if res.name.startswith('linearity_test'):
            iotimes = [1000. / val for val in res.iops.raw]

            op_summ = get_test_summary(res.params)[:3]

            labels_and_data_mp[op_summ].append(
                [res.p.blocksize, res.iops.raw, iotimes])

            cvls = res.params.vals.copy()
            del cvls['blocksize']
            del cvls['rw']

            cvls.pop('sync', None)
            cvls.pop('direct', None)
            cvls.pop('buffered', None)

            if op_summ not in vls:
                vls[op_summ] = cvls
            else:
                assert cvls == vls[op_summ]

    all_labels = None
    _, ax1 = plt.subplots()
    for name, labels_and_data in labels_and_data_mp.items():
        labels_and_data.sort(key=lambda x: ssize2b(x[0]))

        labels, _, iotimes = zip(*labels_and_data)

        if all_labels is None:
            all_labels = labels
        else:
            assert all_labels == labels

        plt.boxplot(iotimes)
        if len(labels_and_data) > 2 and \
           ssize2b(labels_and_data[-2][0]) >= 4096:

            xt = range(1, len(labels) + 1)

            def io_time(sz, bw, initial_lat):
                return sz / bw + initial_lat

            x = numpy.array(map(ssize2b, labels))
            y = numpy.array([sum(dt) / len(dt) for dt in iotimes])
            popt, _ = scipy.optimize.curve_fit(io_time, x, y, p0=(100., 1.))

            y1 = io_time(x, *popt)
            plt.plot(xt, y1, linestyle='--',
                     label=name + ' LS linear approx')

            for idx, (sz, _, _) in enumerate(labels_and_data):
                if ssize2b(sz) >= 4096:
                    break

            bw = (x[-1] - x[idx]) / (y[-1] - y[idx])
            lat = y[-1] - x[-1] / bw
            y2 = io_time(x, bw, lat)
            plt.plot(xt, y2, linestyle='--',
                     label=abbv_name_to_full(name) +
                     ' (4k & max) linear approx')

    plt.setp(ax1, xticklabels=labels)

    plt.xlabel("Block size")
    plt.ylabel("IO time, ms")

    plt.subplots_adjust(top=0.85)
    plt.legend(bbox_to_anchor=(0.5, 1.15),
               loc='upper center',
               prop={'size': 10}, ncol=2)
    plt.grid()
    iotime_plot = get_emb_data_svg(plt)
    plt.clf()

    # plot IOPS = func(bsize)
    _, ax1 = plt.subplots()

    for name, labels_and_data in labels_and_data_mp.items():
        labels_and_data.sort(key=lambda x: ssize2b(x[0]))
        _, data, _ = zip(*labels_and_data)
        plt.boxplot(data)
        avg = [float(sum(arr)) / len(arr) for arr in data]
        xt = range(1, len(data) + 1)
        plt.plot(xt, avg, linestyle='--',
                 label=abbv_name_to_full(name) + " avg")

    plt.setp(ax1, xticklabels=labels)
    plt.xlabel("Block size")
    plt.ylabel("IOPS")
    plt.legend(bbox_to_anchor=(0.5, 1.15),
               loc='upper center',
               prop={'size': 10}, ncol=2)
    plt.grid()
    plt.subplots_adjust(top=0.85)

    iops_plot = get_emb_data_svg(plt)

    res = set(get_test_lcheck_params(res) for res in processed_results.values())
    ncount = list(set(res.testnodes_count for res in processed_results.values()))
    conc = list(set(res.concurence for res in processed_results.values()))

    assert len(conc) == 1
    assert len(ncount) == 1

    descr = {
        'vm_count': ncount[0],
        'concurence': conc[0],
        'oper_descr': ", ".join(res).capitalize()
    }

    params_map = {'iotime_vs_size': iotime_plot,
                  'iops_vs_size': iops_plot,
                  'descr': descr}

    return get_template('report_linearity.html').format(**params_map)


@report('lat_vs_iops', 'lat_vs_iops')
def lat_vs_iops(processed_results, lab_info, comment):
    lat_iops = collections.defaultdict(lambda: [])
    requsted_vs_real = collections.defaultdict(lambda: {})

    for res in processed_results.values():
        if res.name.startswith('lat_vs_iops'):
            lat_iops[res.concurence].append((res.lat.average / 1000.0,
                                             res.lat.deviation / 1000.0,
                                             res.iops.average,
                                             res.iops.deviation))
            requested_iops = res.p.rate_iops * res.concurence
            requsted_vs_real[res.concurence][requested_iops] = \
                (res.iops.average, res.iops.deviation)

    colors = ['red', 'green', 'blue', 'orange', 'magenta', "teal"]
    colors_it = iter(colors)
    for conc, lat_iops in sorted(lat_iops.items()):
        lat, dev, iops, iops_dev = zip(*lat_iops)
        plt.errorbar(iops, lat, xerr=iops_dev, yerr=dev, fmt='ro',
                     label=str(conc) + " threads",
                     color=next(colors_it))

    plt.xlabel("IOPS")
    plt.ylabel("Latency, ms")
    plt.grid()
    plt.legend(loc=0)
    plt_iops_vs_lat = get_emb_data_svg(plt)
    plt.clf()

    colors_it = iter(colors)
    for conc, req_vs_real in sorted(requsted_vs_real.items()):
        req, real = zip(*sorted(req_vs_real.items()))
        iops, dev = zip(*real)
        plt.errorbar(req, iops, yerr=dev, fmt='ro',
                     label=str(conc) + " threads",
                     color=next(colors_it))
    plt.xlabel("Requested IOPS")
    plt.ylabel("Get IOPS")
    plt.grid()
    plt.legend(loc=0)
    plt_iops_vs_requested = get_emb_data_svg(plt)

    res1 = processed_results.values()[0]
    params_map = {'iops_vs_lat': plt_iops_vs_lat,
                  'iops_vs_requested': plt_iops_vs_requested,
                  'oper_descr': get_test_lcheck_params(res1).capitalize()}

    return get_template('report_iops_vs_lat.html').format(**params_map)


def render_all_html(comment, info, lab_description, images, templ_name):
    data = info.__dict__.copy()
    for name, val in data.items():
        if not name.startswith('__'):
            if val is None:
                data[name] = '-'
            elif isinstance(val, (int, float, long)):
                data[name] = round_3_digit(val)

    data['bw_read_max'] = (data['bw_read_max'][0] // 1024,
                           data['bw_read_max'][1])
    data['bw_write_max'] = (data['bw_write_max'][0] // 1024,
                            data['bw_write_max'][1])

    images.update(data)
    return get_template(templ_name).format(lab_info=lab_description,
                                           comment=comment,
                                           **images)


def io_chart(title, concurence,
             latv, latv_min, latv_max,
             iops_or_bw, iops_or_bw_err,
             legend, log=False,
             boxplots=False,
             latv_50=None, latv_95=None):
    points = " MiBps" if legend == 'BW' else ""
    lc = len(concurence)
    width = 0.35
    xt = range(1, lc + 1)

    op_per_vm = [v / (vm * th) for v, (vm, th) in zip(iops_or_bw, concurence)]
    fig, p1 = plt.subplots()
    xpos = [i - width / 2 for i in xt]

    p1.bar(xpos, iops_or_bw,
           width=width,
           yerr=iops_or_bw_err,
           ecolor='m',
           color='y',
           label=legend)

    p1.grid(True)
    p1.plot(xt, op_per_vm, '--', label=legend + "/thread", color='black')
    handles1, labels1 = p1.get_legend_handles_labels()

    p2 = p1.twinx()

    if latv_50 is None:
        p2.plot(xt, latv_max, label="lat max")
        p2.plot(xt, latv, label="lat avg")
        p2.plot(xt, latv_min, label="lat min")
    else:
        p2.plot(xt, latv_50, label="lat med")
        p2.plot(xt, latv_95, label="lat 95%")

    plt.xlim(0.5, lc + 0.5)
    plt.xticks(xt, ["{0} * {1}".format(vm, th) for (vm, th) in concurence])
    p1.set_xlabel("VM Count * Thread per VM")
    p1.set_ylabel(legend + points)
    p2.set_ylabel("Latency ms")
    plt.title(title)
    handles2, labels2 = p2.get_legend_handles_labels()

    plt.legend(handles1 + handles2, labels1 + labels2,
               loc='center left', bbox_to_anchor=(1.1, 0.81))

    if log:
        p1.set_yscale('log')
        p2.set_yscale('log')
    plt.subplots_adjust(right=0.68)

    return get_emb_data_svg(plt)


def make_plots(processed_results, plots):
    """
    processed_results: [PerfInfo]
    plots = [(test_name_prefix:str, fname:str, description:str)]
    """
    files = {}
    for name_pref, fname, desc in plots:
        chart_data = []

        for res in processed_results:
            summ = res.name + "_" + res.summary
            if summ.startswith(name_pref):
                chart_data.append(res)

        if len(chart_data) == 0:
            raise ValueError("Can't found any date for " + name_pref)

        use_bw = ssize2b(chart_data[0].p.blocksize) > 16 * 1024

        chart_data.sort(key=lambda x: x.params['vals']['numjobs'])

        lat = None
        lat_min = None
        lat_max = None
        lat_50 = [x.lat_50 for x in chart_data]
        lat_95 = [x.lat_95 for x in chart_data]

        testnodes_count = x.testnodes_count
        concurence = [(testnodes_count, x.concurence)
                      for x in chart_data]

        if use_bw:
            data = [x.bw.average / 1000 for x in chart_data]
            data_dev = [x.bw.confidence / 1000 for x in chart_data]
            name = "BW"
        else:
            data = [x.iops.average for x in chart_data]
            data_dev = [x.iops.confidence for x in chart_data]
            name = "IOPS"

        fc = io_chart(title=desc,
                      concurence=concurence,
                      
                      latv=lat,
                      latv_min=lat_min,
                      latv_max=lat_max,

                      iops_or_bw=data,
                      iops_or_bw_err=data_dev,

                      legend=name,

                      latv_50=lat_50,
                      latv_95=lat_95)
        files[fname] = fc

    return files


def find_max_where(processed_results, sync_mode, blocksize, rw, iops=True):
    result = None
    attr = 'iops' if iops else 'bw'
    for measurement in processed_results:
        ok = measurement.sync_mode == sync_mode
        ok = ok and (measurement.p.blocksize == blocksize)
        ok = ok and (measurement.p.rw == rw)

        if ok:
            field = getattr(measurement, attr)

            if result is None:
                result = field
            elif field.average > result.average:
                result = field

    return result


def get_disk_info(processed_results):
    di = DiskInfo()
    di.direct_iops_w_max = find_max_where(processed_results,
                                          'd', '4k', 'randwrite')
    di.direct_iops_r_max = find_max_where(processed_results,
                                          'd', '4k', 'randread')

    di.direct_iops_w64_max = find_max_where(processed_results,
                                            'd', '64k', 'randwrite')

    for sz in ('16m', '64m'):
        di.bw_write_max = find_max_where(processed_results,
                                         'd', sz, 'randwrite', False)
        if di.bw_write_max is not None:
            break

    if di.bw_write_max is None:
        di.bw_write_max = find_max_where(processed_results,
                                         'd', '1m', 'write', False)

    for sz in ('16m', '64m'):
        di.bw_read_max = find_max_where(processed_results,
                                        'd', sz, 'randread', False)
        if di.bw_read_max is not None:
            break

    if di.bw_read_max is None:
        di.bw_read_max = find_max_where(processed_results,
                                        'd', '1m', 'read', False)

    rws4k_iops_lat_th = []
    for res in processed_results:
        if res.sync_mode in 'xs' and res.p.blocksize == '4k':
            if res.p.rw != 'randwrite':
                continue
            rws4k_iops_lat_th.append((res.iops.average,
                                      res.lat.average,
                                      res.concurence))

    rws4k_iops_lat_th.sort(key=lambda (_1, _2, conc): conc)

    latv = [lat for _, lat, _ in rws4k_iops_lat_th]

    for tlatv_ms in [10, 30, 100]:
        tlat = tlatv_ms * 1000
        pos = bisect.bisect_left(latv, tlat)
        if 0 == pos:
            setattr(di, 'rws4k_{}ms'.format(tlatv_ms), 0)
        elif pos == len(latv):
            iops3, _, _ = rws4k_iops_lat_th[-1]
            setattr(di, 'rws4k_{}ms'.format(tlatv_ms), ">=" + str(iops3))
        else:
            lat1 = latv[pos - 1]
            lat2 = latv[pos]

            iops1, _, th1 = rws4k_iops_lat_th[pos - 1]
            iops2, _, th2 = rws4k_iops_lat_th[pos]

            th_lat_coef = (th2 - th1) / (lat2 - lat1)
            th3 = th_lat_coef * (tlat - lat1) + th1

            th_iops_coef = (iops2 - iops1) / (th2 - th1)
            iops3 = th_iops_coef * (th3 - th1) + iops1
            setattr(di, 'rws4k_{}ms'.format(tlatv_ms), int(iops3))

    hdi = DiskInfo()

    def pp(x):
        med, conf = x.rounded_average_conf()
        conf_perc = int(float(conf) / med * 100)
        return (med, conf_perc)

    hdi.direct_iops_r_max = pp(di.direct_iops_r_max)

    if di.direct_iops_w_max is not None:
        hdi.direct_iops_w_max = pp(di.direct_iops_w_max)
    else:
        hdi.direct_iops_w_max = None

    if di.direct_iops_w64_max is not None:
        hdi.direct_iops_w64_max = pp(di.direct_iops_w64_max)
    else:
        hdi.direct_iops_w64_max = None

    hdi.bw_write_max = pp(di.bw_write_max)
    hdi.bw_read_max = pp(di.bw_read_max)

    hdi.rws4k_10ms = di.rws4k_10ms if 0 != di.rws4k_10ms else None
    hdi.rws4k_30ms = di.rws4k_30ms if 0 != di.rws4k_30ms else None
    hdi.rws4k_100ms = di.rws4k_100ms if 0 != di.rws4k_100ms else None
    return hdi


@report('hdd', 'hdd')
def make_hdd_report(processed_results, lab_info, comment):
    plots = [
        ('hdd_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
        ('hdd_rwx4k', 'rand_write_4k', 'Random write 4k sync IOPS')
    ]
    perf_infos = [res.disk_perf_info() for res in processed_results]
    images = make_plots(perf_infos, plots)
    di = get_disk_info(perf_infos)
    return render_all_html(comment, di, lab_info, images, "report_hdd.html")


@report('cinder_iscsi', 'cinder_iscsi')
def make_cinder_iscsi_report(processed_results, lab_info, comment):
    plots = [
        ('cinder_iscsi_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
        ('cinder_iscsi_rwx4k', 'rand_write_4k', 'Random write 4k sync IOPS')
    ]
    perf_infos = [res.disk_perf_info() for res in processed_results]
    try:
        images = make_plots(perf_infos, plots)
    except ValueError:
        plots = [
            ('cinder_iscsi_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
            ('cinder_iscsi_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS')
        ]
        images = make_plots(perf_infos, plots)
    di = get_disk_info(perf_infos)
    return render_all_html(comment, di, lab_info, images, "report_cinder_iscsi.html")


@report('ceph', 'ceph')
def make_ceph_report(processed_results, lab_info, comment):
    plots = [
        ('ceph_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
        ('ceph_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS'),
        ('ceph_rrd16m', 'rand_read_16m', 'Random read 16m direct MiBps'),
        ('ceph_rwd16m', 'rand_write_16m',
         'Random write 16m direct MiBps'),
    ]

    perf_infos = [res.disk_perf_info() for res in processed_results]
    images = make_plots(perf_infos, plots)
    di = get_disk_info(perf_infos)
    return render_all_html(comment, di, lab_info, images, "report_ceph.html")


@report('mixed', 'mixed')
def make_mixed_report(processed_results, lab_info, comment):
    #
    # IOPS(X% read) = 100 / ( X / IOPS_W + (100 - X) / IOPS_R )
    #

    perf_infos = [res.disk_perf_info() for res in processed_results]
    mixed = collections.defaultdict(lambda: [])

    is_ssd = False
    for res in perf_infos:
        if res.name.startswith('mixed'):
            if res.name.startswith('mixed-ssd'):
                is_ssd = True
            mixed[res.concurence].append((res.p.rwmixread,
                                          res.lat.average / 1000.0,
                                          res.lat.deviation / 1000.0,
                                          res.iops.average,
                                          res.iops.deviation))

    if len(mixed) == 0:
        raise ValueError("No mixed load found")

    fig, p1 = plt.subplots()
    p2 = p1.twinx()

    colors = ['red', 'green', 'blue', 'orange', 'magenta', "teal"]
    colors_it = iter(colors)
    for conc, mix_lat_iops in sorted(mixed.items()):
        mix_lat_iops = sorted(mix_lat_iops)
        read_perc, lat, dev, iops, iops_dev = zip(*mix_lat_iops)
        p1.errorbar(read_perc, iops, color=next(colors_it),
                    yerr=iops_dev, label=str(conc) + " th")

        p2.errorbar(read_perc, lat, color=next(colors_it),
                    ls='--', yerr=dev, label=str(conc) + " th lat")

    if is_ssd:
        p1.set_yscale('log')
        p2.set_yscale('log')

    p1.set_xlim(-5, 105)

    read_perc = set(read_perc)
    read_perc.add(0)
    read_perc.add(100)
    read_perc = sorted(read_perc)

    plt.xticks(read_perc, map(str, read_perc))

    p1.grid(True)
    p1.set_xlabel("% of reads")
    p1.set_ylabel("Mixed IOPS")
    p2.set_ylabel("Latency, ms")

    handles1, labels1 = p1.get_legend_handles_labels()
    handles2, labels2 = p2.get_legend_handles_labels()
    plt.subplots_adjust(top=0.85)
    plt.legend(handles1 + handles2, labels1 + labels2,
               bbox_to_anchor=(0.5, 1.15),
               loc='upper center',
               prop={'size': 12}, ncol=3)
    plt.show()


def make_load_report(idx, results_dir, fname):
    dpath = os.path.join(results_dir, "io_" + str(idx))
    files = sorted(os.listdir(dpath))
    gf = lambda x: "_".join(x.rsplit(".", 1)[0].split('_')[:3])

    for key, group in itertools.groupby(files, gf):
        fname = os.path.join(dpath, key + ".fio")

        cfgs = list(parse_all_in_1(open(fname).read(), fname))

        fname = os.path.join(dpath, key + "_lat.log")

        curr = []
        arrays = []

        with open(fname) as fd:
            for offset, lat, _, _ in csv.reader(fd):
                offset = int(offset)
                lat = int(lat)
                if len(curr) > 0 and curr[-1][0] > offset:
                    arrays.append(curr)
                    curr = []
                curr.append((offset, lat))
            arrays.append(curr)
        conc = int(cfgs[0].vals.get('numjobs', 1))

        if conc != 5:
            continue

        assert len(arrays) == len(cfgs) * conc

        garrays = [[(0, 0)] for _ in range(conc)]

        for offset in range(len(cfgs)):
            for acc, new_arr in zip(garrays, arrays[offset * conc:(offset + 1) * conc]):
                last = acc[-1][0]
                for off, lat in new_arr:
                    acc.append((off / 1000. + last, lat / 1000.))

        for cfg, arr in zip(cfgs, garrays):
            plt.plot(*zip(*arr[1:]))
        plt.show()
        exit(1)


def make_io_report(dinfo, comment, path, lab_info=None):
    lab_info = {
        "total_disk": "None",
        "total_memory": "None",
        "nodes_count": "None",
        "processor_count": "None"
    }

    try:
        res_fields = sorted(v.name for v in dinfo)

        found = False
        for fields, name, func in report_funcs:
            for field in fields:
                pos = bisect.bisect_left(res_fields, field)

                if pos == len(res_fields):
                    break

                if not res_fields[pos].startswith(field):
                    break
            else:
                found = True
                hpath = path.format(name)

                try:
                    report = func(dinfo, lab_info, comment)
                except:
                    logger.exception("Diring {0} report generation".format(name))
                    continue

                if report is not None:
                    try:
                        with open(hpath, "w") as fd:
                            fd.write(report)
                    except:
                        logger.exception("Diring saving {0} report".format(name))
                        continue
                    logger.info("Report {0} saved into {1}".format(name, hpath))
                else:
                    logger.warning("No report produced by {0!r}".format(name))

        if not found:
            logger.warning("No report generator found for this load")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        logger.error("Failed to generate html report:" + str(exc))
