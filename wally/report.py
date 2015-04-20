import os
import sys

from wally import charts
from wally.statistic import med_dev, round_3_digit, round_deviation
from wally.utils import parse_creds
from wally.suits.io.results_loader import filter_data
from wally.meta_info import total_lab_info, collect_lab_data


def render_html(charts_urls, dest, lab_description, info):
    templ = open("report.html", 'r').read()
    open(dest, 'w').write(templ.format(urls=charts_urls,
                                       data=info, lab_info=lab_description))


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
                                            "bw_per_vm")
                                    ])
    return str(ch)


def make_io_report(results, path, lab_url=None, creds=None):
    if lab_url is not None:
        username, password, tenant_name = parse_creds(creds)
        creds = {'username': username,
                 'password': password,
                 "tenant_name": tenant_name}
        data = collect_lab_data(lab_url, creds)
        lab_info = total_lab_info(data)
    else:
        lab_info = ""

    for suite_type, test_suite_data in results:
        if suite_type != 'io' or test_suite_data is None:
            continue

        io_test_suite_res = test_suite_data['res']

        charts_url = []
        max_info = {}

        name_filters = [
            ('hdd_test_rrd4k', ('concurence', 'lat', 'iops'),
             'rand_read_4k', 'random read 4k'),
            # ('hdd_test_swd1m', ('concurence', 'lat', 'bw'), 'seq_write_1m'),
            # ('hdd_test_srd1m', ('concurence', 'lat', 'bw'), 'seq_read_1m'),
            ('hdd_test_rws4k', ('concurence', 'lat', 'iops'),
             'rand_write_4k', 'random write 4k')
        ]

        for name_filter, fields, fname, desc in name_filters:
            th_filter = filter_data(name_filter, fields)

            data = sorted(th_filter(io_test_suite_res.values()))
            if len(data) == 0:
                continue

            concurence, latv, iops_or_bw_v = zip(*data)
            iops_or_bw_v, iops_or_bw_dev_v = zip(*map(med_dev, iops_or_bw_v))
            latv, _ = zip(*map(med_dev, latv))

            url = io_chart(desc, concurence, latv, iops_or_bw_v,
                           iops_or_bw_dev_v,
                           fields[2], fname)
            max_lat = "%s msec" % round_3_digit(max(latv) / 1000)
            max_iops_or_bw = max(iops_or_bw_v)
            max_iops_or_bw_dev = iops_or_bw_dev_v[
                iops_or_bw_v.index(max_iops_or_bw)]
            r = round_deviation((max_iops_or_bw, max_iops_or_bw_dev))
            max_info[fname] = {fields[2]: r,
                                 "lat": max_lat}
            charts_url.append(url)

        if len(charts_url) != 0:
            render_html(charts_url, path, lab_info, max_info)


def main(args):
    make_io_report(results=[('a', 'b')],
                   path=os.path.dirname(args[0]),
                   lab_url='http://172.16.52.112:8000',
                   creds='admin:admin@admin')
    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
