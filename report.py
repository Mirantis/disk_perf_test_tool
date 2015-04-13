import sys
from collections import OrderedDict

import matplotlib.pyplot as plt

import formatters
from chart import charts
from utils import ssize_to_b
from statistic import med_dev, approximate_curve

from disk_perf_test_tool.tests.io_results_loader import (load_files,
                                                         filter_data)


OPERATIONS = (('async', ('randwrite asynchronous', 'randread asynchronous',
                         'write asynchronous', 'read asynchronous')),
              ('sync', ('randwrite synchronous', 'randread synchronous',
                        'write synchronous', 'read synchronous')),
              ('direct', ('randwrite direct', 'randread direct',
                          'write direct', 'read direct')))

sync_async_view = {'s': 'synchronous',
                   'a': 'asynchronous',
                   'd': 'direct'}


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


def build_vertical_bar(results, z=0):
    data = {}
    charts_url = []
    for build, res in results:
        formatted_res = formatters.get_formatter(build)(res)
        for key, value in formatted_res.items():
            keys = key.split(' ')
            data.setdefault(keys[z], {}).setdefault(build, {})
            data[keys[z]][build][
                ' '.join(keys)] = value

    for name, value in data.items():
        title = name
        legend = []
        dataset = []

        scale_x = []

        for build_id, build_results in value.items():
            vals = []
            OD = OrderedDict
            ordered_build_results = OD(sorted(build_results.items(),
                                       key=lambda t: t[0]))
            scale_x = ordered_build_results.keys()
            for key in scale_x:
                res = build_results.get(key)
                if res:
                    vals.append(res)
            if vals:
                dataset.append(vals)
                legend.append(build_id)

        if dataset:
            charts_url.append(str(charts.render_vertical_bar
                              (title, legend, dataset, scale_x=scale_x)))
    return charts_url


def build_lines_chart(results, z=0):
    data = {}
    charts_url = []

    for build, res in results:
        formatted_res = formatters.get_formatter(build)(res)
        for key, value in formatted_res.items():
            keys = key.split(' ')
            data.setdefault(key[z], {})
            data[key[z]].setdefault(build, {})[keys[1]] = value

    for name, value in data.items():
        title = name
        legend = []
        dataset = []
        scale_x = []
        for build_id, build_results in value.items():
            legend.append(build_id)

            OD = OrderedDict
            ordered_build_results = OD(sorted(build_results.items(),
                                       key=lambda t: ssize_to_b(t[0])))

            if not scale_x:
                scale_x = ordered_build_results.keys()
            dataset.append(zip(*ordered_build_results.values())[0])

        chart = charts.render_lines(title, legend, dataset, scale_x)
        charts_url.append(str(chart))

    return charts_url


def render_html(charts_urls, dest):
    templ = open("report.html", 'r').read()
    body = "<div><ol>%s</ol></div>"
    li = "<li><img src='%s'></li>"
    ol = []
    for chart in charts_urls:
        ol.append(li % chart)
    html = templ % {'body': body % '\n'.join(ol)}
    open(dest, 'w').write(html)


def build_io_chart(res):
    pass


# def render_html_results(ctx):
#     charts = []
#     for res in ctx.results:
#         if res[0] == "io":
#             charts.append(build_io_chart(res))

#     bars = build_vertical_bar(ctx.results)
#     lines = build_lines_chart(ctx.results)

    # render_html(bars + lines, dest)


def make_io_report(results):
    for suite_type, test_suite_data in results:
        if suite_type != 'io':
            continue

        io_test_suite_res = test_suite_data['res']

        charts_url = []

        name_filters = [
            #('hdd_test_rws4k', ('concurence', 'lat', 'iops')),
            #('hdd_test_rrs4k', ('concurence', 'lat', 'iops')),
            ('hdd_test_rrd4k', ('concurence', 'lat', 'iops')),
            ('hdd_test_swd1m', ('concurence', 'lat', 'bw_mean')),
        ]

        for name_filter, fields in name_filters:
            th_filter = filter_data(name_filter, fields)

            data_iter = sorted(th_filter(io_test_suite_res.values()))

            concurence, latv, iops_or_bw_v = zip(*data_iter)
            iops_or_bw_v, iops_or_bw_dev_v = zip(*map(med_dev, iops_or_bw_v))

            _, ax1 = plt.subplots()

            ax1.plot(concurence, iops_or_bw_v)
            ax1.errorbar(concurence, iops_or_bw_v, iops_or_bw_dev_v,
                         linestyle='None',
                         label="iops_or_bw_v",
                         marker="*")

            # ynew = approximate_line(ax, ay, ax, True)

            ax2 = ax1.twinx()

            ax2.errorbar(concurence,
                         [med_dev(lat)[0] / 1000 for lat in latv],
                         [med_dev(lat)[1] / 1000 for lat in latv],
                         linestyle='None',
                         label="iops_or_bw_v",
                         marker="*")
            ax2.plot(concurence, [med_dev(lat)[0] / 1000 for lat in latv])
            plt.show()
            exit(0)

            # bw_only = []

            # for conc, _, _, (bw, _) in data:
            #     bw_only.append(bw)
            #     bw_d_per_th.append((bw / conc, 0))

            # lines = [(zip(*lat_d)[0], 'msec', 'rr', 'lat'), (bw_sum, None, None, 'bw_sum')]

            # chart_url = charts.render_vertical_bar(
            #                 chart_name, ["bw"], [bw_d_per_th], label_x="KBps",
            #                 scale_x=ordered_data.keys(),
            #                 lines=lines)

            # charts_url.append(str(chart_url))

        render_html(charts_url, "results.html")


def main(args):
    make_io_report('/tmp/report', load_files(args[1:]))
    return 0


if __name__ == '__main__':
    exit(main(sys.argv))
