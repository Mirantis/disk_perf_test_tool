import os
import bisect
import logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import wally
from wally import charts
from wally.utils import ssize2b
from wally.statistic import round_3_digit, data_property


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


class PerfInfo(object):
    def __init__(self, name, raw, meta):
        self.name = name
        self.bw = None
        self.iops = None
        self.lat = None
        self.raw = raw
        self.meta = meta


def split_and_add(data, block_size):
    assert len(data) % block_size == 0
    res = [0] * block_size

    for idx, val in enumerate(data):
        res[idx % block_size] += val

    return res


def process_disk_info(test_data):
    data = {}
    vm_count = test_data['__test_meta__']['testnodes_count']
    for name, results in test_data['res'].items():
        assert len(results['bw']) % vm_count == 0
        block_count = len(results['bw']) // vm_count

        pinfo = PerfInfo(name, results, test_data['__test_meta__'])
        pinfo.bw = data_property(split_and_add(results['bw'], block_count))
        pinfo.iops = data_property(split_and_add(results['iops'],
                                                 block_count))

        pinfo.lat = data_property(results['lat'])
        data[name] = pinfo
    return data


def report(name, required_fields):
    def closure(func):
        report_funcs.append((required_fields.split(","), name, func))
        return func
    return closure


def linearity_report(processed_results, path, lab_info):
    names = {}
    for tp1 in ('rand', 'seq'):
        for oper in ('read', 'write'):
            for sync in ('sync', 'direct', 'async'):
                sq = (tp1, oper, sync)
                name = "{0} {1} {2}".format(*sq)
                names["".join(word[0] for word in sq)] = name

    colors = ['red', 'green', 'blue', 'cyan',
              'magenta', 'black', 'yellow', 'burlywood']
    markers = ['*', '^', 'x', 'o', '+', '.']
    color = 0
    marker = 0

    plot_data = {}

    name_pref = 'linearity_test_rrd'

    for res in processed_results.values():
        if res.name.startswith(name_pref):
            iotime = 1000000. / res.iops
            iotime_max = iotime * (1 + res.dev * 3)
            bsize = ssize2b(res.raw['blocksize'])
            plot_data[bsize] = (iotime, iotime_max)

    min_sz = min(plot_data)
    min_iotime, _ = plot_data.pop(min_sz)

    x = []
    y = []
    e = []

    for k, (v, vmax) in sorted(plot_data.items()):
        y.append(v - min_iotime)
        x.append(k)
        e.append(y[-1] - (vmax - min_iotime))

    tp = 'rrd'
    plt.errorbar(x, y, e, linestyle='None', label=names[tp],
                 color=colors[color], ecolor="black",
                 marker=markers[marker])
    plt.yscale('log')
    plt.xscale('log')
    # plt.show()

    # ynew = approximate_line(ax, ay, ax, True)
    # plt.plot(ax, ynew, color=colors[color])
    # color += 1
    # marker += 1
    # plt.legend(loc=2)
    # plt.title("Linearity test by %i dots" % (len(vals)))


if plt:
    linearity_report = report('linearity', 'linearity_test')(linearity_report)


def render_all_html(dest, info, lab_description, img_ext, templ_name):
    very_root_dir = os.path.dirname(os.path.dirname(wally.__file__))
    templ_dir = os.path.join(very_root_dir, 'report_templates')
    templ_file = os.path.join(templ_dir, templ_name)
    templ = open(templ_file, 'r').read()

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

    report = templ.format(lab_info=lab_description, img_ext=img_ext,
                          **data)
    open(dest, 'w').write(report)


def render_hdd_html(dest, info, lab_description, img_ext):
    render_all_html(dest, info, lab_description, img_ext,
                    "report_hdd.html")


def render_ceph_html(dest, info, lab_description, img_ext):
    render_all_html(dest, info, lab_description, img_ext,
                    "report_ceph.html")


def io_chart(title, concurence,
             latv, latv_min, latv_max,
             iops_or_bw, iops_or_bw_dev,
             legend, fname):
    bar_data = iops_or_bw
    bar_dev = iops_or_bw_dev
    legend = [legend]

    iops_or_bw_per_vm = []
    for iops, conc in zip(iops_or_bw, concurence):
        iops_or_bw_per_vm.append(iops / conc)

    bar_dev_bottom = []
    bar_dev_top = []
    for val, err in zip(bar_data, bar_dev):
        bar_dev_top.append(val + err)
        bar_dev_bottom.append(val - err)

    charts.render_vertical_bar(title, legend, [bar_data], [bar_dev_top],
                               [bar_dev_bottom], file_name=fname,
                               scale_x=concurence, label_x="clients",
                               label_y=legend[0],
                               lines=[
                                    (latv, "msec", "rr", "lat"),
                                    # (latv_min, None, None, "lat_min"),
                                    # (latv_max, None, None, "lat_max"),
                                    (iops_or_bw_per_vm, None, None,
                                     legend[0] + " per client")
                                ])


def io_chart_mpl(title, concurence,
                 latv, latv_min, latv_max,
                 iops_or_bw, iops_or_bw_err,
                 legend, fname):
    points = " MiBps" if legend == 'BW' else ""
    lc = len(concurence)
    width = 0.35
    xt = range(1, lc + 1)

    op_per_vm = [v / c for v, c in zip(iops_or_bw, concurence)]
    fig, p1 = plt.subplots()
    xpos = [i - width / 2 for i in xt]

    p1.bar(xpos, iops_or_bw, width=width, yerr=iops_or_bw_err,
           color='y',
           label=legend)

    p1.set_yscale('log')
    p1.grid(True)
    p1.plot(xt, op_per_vm, label=legend + " per vm")
    p1.legend()

    p2 = p1.twinx()
    p2.set_yscale('log')
    p2.plot(xt, latv_max, label="latency max")
    p2.plot(xt, latv, label="latency avg")
    p2.plot(xt, latv_min, label="latency min")

    plt.xlim(0.5, lc + 0.5)
    plt.xticks(xt, map(str, concurence))
    p1.set_xlabel("Threads")
    p1.set_ylabel(legend + points)
    p2.set_ylabel("Latency ms")
    plt.title(title)
    # plt.legend(, loc=2, borderaxespad=0.)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.legend(loc=2)
    plt.savefig(fname, format=fname.split('.')[-1])


def make_hdd_plots(processed_results, charts_dir):
    plots = [
        ('hdd_test_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
        ('hdd_test_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS')
    ]
    return make_plots(processed_results, charts_dir, plots)


def make_ceph_plots(processed_results, charts_dir):
    plots = [
        ('ceph_test_rrd4k', 'rand_read_4k', 'Random read 4k direct IOPS'),
        ('ceph_test_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS'),
        ('ceph_test_rrd16m', 'rand_read_16m',
         'Random read 16m direct MiBps'),
        ('ceph_test_rwd16m', 'rand_write_16m',
            'Random write 16m direct MiBps'),
    ]
    return make_plots(processed_results, charts_dir, plots)


def make_plots(processed_results, charts_dir, plots):
    file_ext = None
    for name_pref, fname, desc in plots:
        chart_data = []

        for res in processed_results.values():
            if res.name.startswith(name_pref):
                chart_data.append(res)

        if len(chart_data) == 0:
            raise ValueError("Can't found any date for " + name_pref)

        use_bw = ssize2b(chart_data[0].raw['blocksize']) > 16 * 1024

        chart_data.sort(key=lambda x: x.raw['concurence'])

        #  if x.lat.average < max_lat]
        lat = [x.lat.average / 1000 for x in chart_data]
        lat_min = [x.lat.min / 1000 for x in chart_data]
        lat_max = [x.lat.max / 1000 for x in chart_data]

        vm_count = x.meta['testnodes_count']
        concurence = [x.raw['concurence'] * vm_count for x in chart_data]

        if use_bw:
            data = [x.bw.average / 1000 for x in chart_data]
            data_dev = [x.bw.confidence / 1000 for x in chart_data]
            name = "BW"
        else:
            data = [x.iops.average for x in chart_data]
            data_dev = [x.iops.confidence for x in chart_data]
            name = "IOPS"

        fname = os.path.join(charts_dir, fname)
        if plt is not None:
            io_chart_mpl(desc, concurence, lat, lat_min, lat_max,
                         data, data_dev, name, fname + '.svg')
            file_ext = 'svg'
        else:
            io_chart(desc, concurence, lat, lat_min, lat_max,
                     data, data_dev, name, fname + '.png')
            file_ext = 'png'
    return file_ext


def find_max_where(processed_results, sync_mode, blocksize, rw, iops=True):
    result = None
    attr = 'iops' if iops else 'bw'
    for measurement in processed_results.values():
        ok = measurement.raw['sync_mode'] == sync_mode
        ok = ok and (measurement.raw['blocksize'] == blocksize)
        ok = ok and (measurement.raw['rw'] == rw)

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
        if res.raw['sync_mode'] == 's' and res.raw['blocksize'] == '4k':
            if res.raw['rw'] != 'randwrite':
                continue
            rws4k_iops_lat_th.append((res.iops.average,
                                      res.lat.average,
                                      res.raw['concurence']))

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


@report('HDD', 'hdd_test_rrd4k,hdd_test_rws4k')
def make_hdd_report(processed_results, path, charts_path, lab_info):
    img_ext = make_hdd_plots(processed_results, charts_path)
    di = get_disk_info(processed_results)
    render_hdd_html(path, di, lab_info, img_ext)


@report('Ceph', 'ceph_test')
def make_ceph_report(processed_results, path, charts_path, lab_info):
    img_ext = make_ceph_plots(processed_results, charts_path)
    di = get_disk_info(processed_results)
    render_ceph_html(path, di, lab_info, img_ext)


def make_io_report(dinfo, results, path, charts_path, lab_info=None):
    lab_info = {
        "total_disk": "None",
        "total_memory": "None",
        "nodes_count": "None",
        "processor_count": "None"
    }

    try:
        res_fields = sorted(dinfo.keys())
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
                func(dinfo, hpath, charts_path, lab_info)
                break
        else:
            logger.warning("No report generator found for this load")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        logger.error("Failed to generate html report:" + str(exc))
