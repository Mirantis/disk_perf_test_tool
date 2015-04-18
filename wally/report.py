import os
import sys

from wally import charts
from wally.statistic import med_dev
from wally.utils import parse_creds
from wally.suits.io.results_loader import filter_data
from wally.meta_info import total_lab_info, collect_lab_data


# from collections import OrderedDict
# from wally.suits.io import formatter
# def pgbench_chart_data(results):
#     """
#     Format pgbench results for chart
#     """
#     data = {}
#     charts_url = []

#     formatted_res = formatters.format_pgbench_stat(results)
#     for key, value in formatted_res.items():
#         num_cl, num_tr = key.split(' ')
#         data.setdefault(num_cl, {}).setdefault(build, {})
#         data[keys[z]][build][
#             ' '.join(keys)] = value

#     for name, value in data.items():
#         title = name
#         legend = []
#         dataset = []

#         scale_x = []

#         for build_id, build_results in value.items():
#             vals = []
#             OD = OrderedDict
#             ordered_build_results = OD(sorted(build_results.items(),
#                                        key=lambda t: t[0]))
#             scale_x = ordered_build_results.keys()
#             for key in scale_x:
#                 res = build_results.get(key)
#                 if res:
#                     vals.append(res)
#             if vals:
#                 dataset.append(vals)
#                 legend.append(build_id)

#         if dataset:
#             charts_url.append(str(charts.render_vertical_bar
#                               (title, legend, dataset, scale_x=scale_x)))
#     return charts_url

# def build_lines_chart(results, z=0):
#     data = {}
#     charts_url = []

#     for build, res in results:
#         formatted_res = formatters.get_formatter(build)(res)
#         for key, value in formatted_res.items():
#             keys = key.split(' ')
#             data.setdefault(key[z], {})
#             data[key[z]].setdefault(build, {})[keys[1]] = value

#     for name, value in data.items():
#         title = name
#         legend = []
#         dataset = []
#         scale_x = []
#         for build_id, build_results in value.items():
#             legend.append(build_id)

#             OD = OrderedDict
#             ordered_build_results = OD(sorted(build_results.items(),
#                                        key=lambda t: ssize_to_b(t[0])))

#             if not scale_x:
#                 scale_x = ordered_build_results.keys()
#             dataset.append(zip(*ordered_build_results.values())[0])

#         chart = charts.render_lines(title, legend, dataset, scale_x)
#         charts_url.append(str(chart))

#     return charts_url

# def build_vertical_bar(results, z=0):
#     data = {}
#     charts_url = []
#     for build, res in results:
#         formatted_res = formatter.get_formatter(build)(res)
#         for key, value in formatted_res.items():
#             keys = key.split(' ')
#             data.setdefault(keys[z], {}).setdefault(build, {})
#             data[keys[z]][build][
#                 ' '.join(keys)] = value

#     for name, value in data.items():
#         title = name
#         legend = []
#         dataset = []

#         scale_x = []

#         for build_id, build_results in value.items():
#             vals = []
#             OD = OrderedDict
#             ordered_build_results = OD(sorted(build_results.items(),
#                                        key=lambda t: t[0]))
#             scale_x = ordered_build_results.keys()
#             for key in scale_x:
#                 res = build_results.get(key)
#                 if res:
#                     vals.append(res)
#             if vals:
#                 dataset.append(vals)
#                 legend.append(build_id)

#         if dataset:
#             charts_url.append(str(charts.render_vertical_bar
#                               (title, legend, dataset, scale_x=scale_x)))
#     return charts_url


def render_html(charts_urls, dest, lab_description, info):
    templ = open("report.html", 'r').read()
    body = "<a href='#lab_desc'>Lab description</a>" \
           "<ol>{0}</ol>" \
           "<div>{1}</div>" \
           '<a name="lab_desc"></a>' \
           "<div><ul>{2}</ul></div>"
    table = "<table><tr><td>{0}</td><td>{1}</td></tr>" \
            "<tr><td>{2}</td><td>{3}</td></tr></table>"
    ul = []
    ol = []
    li = '<li>{0} : {1}</li>'

    for elem in info:
        ol.append(li.format(elem.keys(), elem.values()))

    for key in lab_description:
        value = lab_description[key]
        ul.append("<li>{0} : {1}</li>".format(key, value))

    charts_urls = ['<img src="{0}">'.format(url) for url in charts_urls]

    body = body.format('\n'.join(ol),
                       table.format(*charts_urls),
                       '\n'.join(ul))

    open(dest, 'w').write(templ % {'body': body})


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
        info = []

        name_filters = [
            ('hdd_test_rrd4k', ('concurence', 'lat', 'iops'), 'rand_read_4k'),
            ('hdd_test_swd1m', ('concurence', 'lat', 'bw'), 'seq_write_1m'),
            ('hdd_test_srd1m', ('concurence', 'lat', 'bw'), 'seq_read_1m'),
            ('hdd_test_rws4k', ('concurence', 'lat', 'bw'), 'rand_write_1m')
        ]

        for name_filter, fields, fname in name_filters:
            th_filter = filter_data(name_filter, fields)

            data = sorted(th_filter(io_test_suite_res.values()))
            if len(data) == 0:
                continue

            concurence, latv, iops_or_bw_v = zip(*data)
            iops_or_bw_v, iops_or_bw_dev_v = zip(*map(med_dev, iops_or_bw_v))
            latv, _ = zip(*map(med_dev, latv))

            url = io_chart(name_filter, concurence, latv, iops_or_bw_v,
                           iops_or_bw_dev_v,
                           fields[2], fname)
            info.append(dict(zip(fields, (concurence, latv, iops_or_bw_v))))
            charts_url.append(url)

        if len(charts_url) != 0:
            render_html(charts_url, path, lab_info, info)


def main(args):
    make_io_report(results=[('a', 'b')],
                   path=os.path.dirname(args[0]),
                   lab_url='http://172.16.52.112:8000',
                   creds='admin:admin@admin')
    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
