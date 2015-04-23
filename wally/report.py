import os
import bisect
import logging

import wally
from wally import charts
from wally.utils import parse_creds
from wally.suits.io.results_loader import process_disk_info
from wally.meta_info import total_lab_info, collect_lab_data


logger = logging.getLogger("wally.report")


def render_html(dest, info, lab_description):
    very_root_dir = os.path.dirname(os.path.dirname(wally.__file__))
    templ_dir = os.path.join(very_root_dir, 'report_templates')
    templ_file = os.path.join(templ_dir, "report.html")
    templ = open(templ_file, 'r').read()
    report = templ.format(lab_info=lab_description, **info.__dict__)
    open(dest, 'w').write(report)


def io_chart(title, concurence, latv, iops_or_bw, iops_or_bw_dev,
             legend, fname):
    bar_data, bar_dev = iops_or_bw, iops_or_bw_dev
    legend = [legend]

    iops_or_bw_per_vm = []
    for i in range(len(concurence)):
        iops_or_bw_per_vm.append(iops_or_bw[i] / concurence[i])

    bar_dev_bottom = []
    bar_dev_top = []
    for i in range(len(bar_data)):
        bar_dev_top.append(bar_data[i] + bar_dev[i])
        bar_dev_bottom.append(bar_data[i] - bar_dev[i])

    latv = [lat / 1000 for lat in latv]
    ch = charts.render_vertical_bar(title, legend, [bar_data], [bar_dev_top],
                                    [bar_dev_bottom], file_name=fname,
                                    scale_x=concurence,
                                    lines=[
                                        (latv, "msec", "rr", "lat"),
                                        (iops_or_bw_per_vm, None, None,
                                            "IOPS per vm")
                                    ])
    return str(ch)


def make_plots(processed_results, path):
    name_filters = [
        ('hdd_test_rrd4k', 'rand_read_4k', 'Random read 4k sync IOPS'),
        ('hdd_test_rws4k', 'rand_write_4k', 'Random write 4k sync IOPS')
    ]

    for name_pref, fname, desc in name_filters:
        chart_data = []
        for res in processed_results.values():
            if res.name.startswith(name_pref):
                chart_data.append(res)

        chart_data.sort(key=lambda x: x.raw['concurence'])

        lat = [x.lat for x in chart_data]
        concurence = [x.raw['concurence'] for x in chart_data]
        iops = [x.iops for x in chart_data]
        iops_dev = [x.iops * x.dev for x in chart_data]

        io_chart(desc, concurence, lat, iops, iops_dev, 'bw', fname)


class DiskInfo(object):
    def __init__(self):
        self.direct_iops_r_max = 0
        self.direct_iops_w_max = 0
        self.rws4k_10ms = 0
        self.rws4k_30ms = 0
        self.rws4k_100ms = 0
        self.bw_write_max = 0
        self.bw_read_max = 0


def get_disk_info(processed_results):
    di = DiskInfo()
    rws4k_iops_lat_th = []

    for res in processed_results.values():
        if res.raw['sync_mode'] == 'd' and res.raw['blocksize'] == '4k':
            if res.raw['rw'] == 'randwrite':
                di.direct_iops_w_max = max(di.direct_iops_w_max, res.iops)
            elif res.raw['rw'] == 'randread':
                di.direct_iops_r_max = max(di.direct_iops_r_max, res.iops)
        elif res.raw['sync_mode'] == 's' and res.raw['blocksize'] == '4k':
            if res.raw['rw'] != 'randwrite':
                continue

            rws4k_iops_lat_th.append((res.iops, res.lat,
                                      res.raw['concurence']))

        elif res.raw['sync_mode'] == 'd' and res.raw['blocksize'] == '1m':

            if res.raw['rw'] == 'write':
                di.bw_write_max = max(di.bw_write_max, res.bw)
            elif res.raw['rw'] == 'read':
                di.bw_read_max = max(di.bw_read_max, res.bw)

    di.bw_write_max /= 1000
    di.bw_read_max /= 1000

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

            th1 = rws4k_iops_lat_th[pos - 1][2]
            th2 = rws4k_iops_lat_th[pos][2]

            iops1 = rws4k_iops_lat_th[pos - 1][0]
            iops2 = rws4k_iops_lat_th[pos][0]

            th_lat_coef = (th2 - th1) / (lat2 - lat1)
            th3 = th_lat_coef * (tlat - lat1) + th1

            th_iops_coef = (iops2 - iops1) / (th2 - th1)
            iops3 = th_iops_coef * (th3 - th1) + iops1
        setattr(di, 'rws4k_{}ms'.format(tlatv_ms), int(iops3))

    hdi = DiskInfo()
    hdi.direct_iops_r_max = di.direct_iops_r_max
    hdi.direct_iops_w_max = di.direct_iops_w_max
    hdi.rws4k_10ms = di.rws4k_10ms if 0 != di.rws4k_10ms else '-'
    hdi.rws4k_30ms = di.rws4k_30ms if 0 != di.rws4k_30ms else '-'
    hdi.rws4k_100ms = di.rws4k_100ms if 0 != di.rws4k_100ms else '-'
    hdi.bw_write_max = di.bw_write_max
    hdi.bw_read_max = di.bw_read_max
    return hdi


report_funcs = []


def report(names):
    def closure(func):
        report_funcs.append((names.split(","), func))
        return func
    return closure


@report('hdd_test_rrd4k,hdd_test_rws4k')
def make_hdd_report(processed_results, path, lab_info):
    make_plots(processed_results, path)
    di = get_disk_info(processed_results)
    render_html(path, di, lab_info)


def make_io_report(results, path, lab_url=None, creds=None):
    if lab_url is not None:
        username, password, tenant_name = parse_creds(creds)
        creds = {'username': username,
                 'password': password,
                 "tenant_name": tenant_name}
        data = collect_lab_data(lab_url, creds)
        lab_info = total_lab_info(data)
    else:
        lab_info = {
            "total_disk": "None",
            "total_memory": "None",
            "nodes_count": "None",
            "processor_count": "None"
        }

    try:
        processed_results = process_disk_info(results)

        for fields, func in report_funcs:
            for field in fields:
                if field not in processed_results:
                    break
            else:
                func(processed_results, path, lab_info)
                break
        else:
            logger.warning("No report generator found for this load")

    except Exception as exc:
        logger.error("Failed to generate html report:" + str(exc))
    else:
        logger.info("Html report were stored in " + path)
