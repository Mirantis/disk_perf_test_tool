import os
import bisect
import logging
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
from wally.statistic import round_3_digit, data_property
from wally.suits.io.fio_task_parser import get_test_sync_mode


logger = logging.getLogger("wally.report")


class DiskInfo(object):
    def __init__(self):
        self.direct_iops_r_max = 0
        self.direct_iops_w_max = 0
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
        name_map[(data.config.name, data.summary())].append(data)

    return name_map


def process_disk_info(test_data):
    name_map = group_by_name(test_data)
    data = {}
    for (name, summary), results in name_map.items():
        testnodes_count_set = set(dt.vm_count for dt in results)

        assert len(testnodes_count_set) == 1
        testnodes_count, = testnodes_count_set
        assert len(results) % testnodes_count == 0

        intervals = [result.run_interval for result in results]
        p = results[0].config
        pinfo = PerfInfo(p.name, result.summary(), intervals,
                         p, testnodes_count)

        pinfo.raw_bw = [result.results['bw'] for result in results]
        pinfo.raw_iops = [result.results['iops'] for result in results]
        pinfo.raw_lat = [result.results['lat'] for result in results]

        pinfo.bw = data_property(map(sum, zip(*pinfo.raw_bw)))
        pinfo.iops = data_property(map(sum, zip(*pinfo.raw_iops)))
        pinfo.lat = data_property(sum(pinfo.raw_lat, []))

        data[(p.name, summary)] = pinfo
    return data


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


@report('linearity', 'linearity_test')
def linearity_report(processed_results, path, lab_info):
    labels_and_data = []

    vls = processed_results.values()[0].params.vals.copy()
    del vls['blocksize']

    for res in processed_results.values():
        if res.name.startswith('linearity_test'):
            iotimes = [1000. / val for val in res.iops.raw]
            labels_and_data.append([res.p.blocksize, res.iops.raw, iotimes])
            cvls = res.params.vals.copy()
            del cvls['blocksize']
            assert cvls == vls

    labels_and_data.sort(key=lambda x: ssize2b(x[0]))
    _, ax1 = plt.subplots()

    labels, data, iotimes = zip(*labels_and_data)
    plt.boxplot(iotimes)

    if len(labels_and_data) > 2 and ssize2b(labels_and_data[-2][0]) >= 4096:
        xt = range(1, len(labels) + 1)

        def io_time(sz, bw, initial_lat):
            return sz / bw + initial_lat

        x = numpy.array(map(ssize2b, labels))
        y = numpy.array([sum(dt) / len(dt) for dt in iotimes])
        popt, _ = scipy.optimize.curve_fit(io_time, x, y, p0=(100., 1.))

        y1 = io_time(x, *popt)
        plt.plot(xt, y1, linestyle='--', label='LS linear approxomation')

        for idx, (sz, _, _) in enumerate(labels_and_data):
            if ssize2b(sz) >= 4096:
                break

        bw = (x[-1] - x[idx]) / (y[-1] - y[idx])
        lat = y[-1] - x[-1] / bw
        y2 = io_time(x, bw, lat)

        plt.plot(xt, y2, linestyle='--',
                 label='(4k & max) linear approxomation')

    plt.setp(ax1, xticklabels=labels)

    plt.xlabel("Block size")
    plt.ylabel("IO time, ms")

    plt.subplots_adjust(top=0.85)
    plt.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center')
    plt.grid()
    iotime_plot = get_emb_data_svg(plt)
    plt.clf()

    _, ax1 = plt.subplots()
    plt.boxplot(data)
    plt.setp(ax1, xticklabels=labels)

    plt.xlabel("Block size")
    plt.ylabel("IOPS")
    plt.grid()
    plt.subplots_adjust(top=0.85)

    iops_plot = get_emb_data_svg(plt)

    res1 = processed_results.values()[0]
    descr = {
        'vm_count': res1.testnodes_count,
        'concurence': res1.concurence,
        'oper_descr': get_test_lcheck_params(res1).capitalize()
    }

    params_map = {'iotime_vs_size': iotime_plot,
                  'iops_vs_size': iops_plot,
                  'descr': descr}

    with open(path, 'w') as fd:
        fd.write(get_template('report_linearity.html').format(**params_map))


@report('lat_vs_iops', 'lat_vs_iops')
def lat_vs_iops(processed_results, path, lab_info):
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

    with open(path, 'w') as fd:
        fd.write(get_template('report_iops_vs_lat.html').format(**params_map))


def render_all_html(dest, info, lab_description, images, templ_name):
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
    report = get_template(templ_name).format(lab_info=lab_description,
                                             **images)

    with open(dest, 'w') as fd:
        fd.write(report)


def io_chart(title, concurence,
             latv, latv_min, latv_max,
             iops_or_bw, iops_or_bw_err,
             legend, log=False,
             boxplots=False):
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
    p2.plot(xt, latv_max, label="lat max")
    p2.plot(xt, latv, label="lat avg")
    p2.plot(xt, latv_min, label="lat min")

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
    files = {}
    for name_pref, fname, desc in plots:
        chart_data = []

        for res in processed_results.values():
            if res.name.startswith(name_pref):
                chart_data.append(res)

        if len(chart_data) == 0:
            raise ValueError("Can't found any date for " + name_pref)

        use_bw = ssize2b(chart_data[0].p.blocksize) > 16 * 1024

        chart_data.sort(key=lambda x: x.concurence)

        #  if x.lat.average < max_lat]
        lat = [x.lat.average / 1000 for x in chart_data]
        lat_min = [x.lat.min / 1000 for x in chart_data]
        lat_max = [x.lat.max / 1000 for x in chart_data]

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
                      latv=lat, latv_min=lat_min, latv_max=lat_max,
                      iops_or_bw=data,
                      iops_or_bw_err=data_dev,
                      legend=name)
        files[fname] = fc

    return files


def find_max_where(processed_results, sync_mode, blocksize, rw, iops=True):
    result = None
    attr = 'iops' if iops else 'bw'
    for measurement in processed_results.values():
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
    rws4k_iops_lat_th = []

    di.direct_iops_w_max = find_max_where(processed_results,
                                          'd', '4k', 'randwrite')
    di.direct_iops_r_max = find_max_where(processed_results,
                                          'd', '4k', 'randread')

    di.bw_write_max = find_max_where(processed_results,
                                     'd', '16m', 'randwrite', False)
    if di.bw_write_max is None:
        di.bw_write_max = find_max_where(processed_results,
                                         'd', '1m', 'write', False)

    di.bw_read_max = find_max_where(processed_results,
                                    'd', '16m', 'randread', False)
    if di.bw_read_max is None:
        di.bw_read_max = find_max_where(processed_results,
                                        'd', '1m', 'read', False)

    for res in processed_results.values():
        if res.sync_mode == 's' and res.p.blocksize == '4k':
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
            iops3 = 0
        elif pos == len(latv):
            iops3 = latv[-1]
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
    hdi.direct_iops_w_max = pp(di.direct_iops_w_max)
    hdi.bw_write_max = pp(di.bw_write_max)
    hdi.bw_read_max = pp(di.bw_read_max)

    hdi.rws4k_10ms = di.rws4k_10ms if 0 != di.rws4k_10ms else None
    hdi.rws4k_30ms = di.rws4k_30ms if 0 != di.rws4k_30ms else None
    hdi.rws4k_100ms = di.rws4k_100ms if 0 != di.rws4k_100ms else None
    return hdi


@report('HDD', 'hdd_test')
def make_hdd_report(processed_results, path, lab_info):
    plots = [
        ('hdd_test_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
        ('hdd_test_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS')
    ]
    images = make_plots(processed_results, plots)
    di = get_disk_info(processed_results)
    render_all_html(path, di, lab_info, images, "report_hdd.html")


@report('Ceph', 'ceph_test')
def make_ceph_report(processed_results, path, lab_info):
    plots = [
        ('ceph_test_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
        ('ceph_test_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS'),
        ('ceph_test_rrd16m', 'rand_read_16m', 'Random read 16m direct MiBps'),
        ('ceph_test_rwd16m', 'rand_write_16m',
         'Random write 16m direct MiBps'),
    ]

    images = make_plots(processed_results, plots)
    di = get_disk_info(processed_results)
    render_all_html(path, di, lab_info, images, "report_ceph.html")


def make_io_report(dinfo, results, path, lab_info=None):
    lab_info = {
        "total_disk": "None",
        "total_memory": "None",
        "nodes_count": "None",
        "processor_count": "None"
    }

    try:
        res_fields = sorted(v.name for v in dinfo.values())

        for fields, name, func in report_funcs:
            for field in fields:
                pos = bisect.bisect_left(res_fields, field)

                if pos == len(res_fields):
                    break

                if not res_fields[pos].startswith(field):
                    break
            else:
                hpath = path.format(name)
                logger.debug("Generatins report " + name + " into " + hpath)
                func(dinfo, hpath, lab_info)
                break
        else:
            logger.warning("No report generator found for this load")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        logger.error("Failed to generate html report:" + str(exc))
