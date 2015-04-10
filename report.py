import argparse
from collections import OrderedDict
import itertools
import math
import re

from chart import charts
import formatters
from utils import ssize_to_b


OPERATIONS = (('async', ('randwrite asynchronous', 'randread asynchronous',
                         'write asynchronous', 'read asynchronous')),
              ('sync', ('randwrite synchronous', 'randread synchronous',
                        'write synchronous', 'read synchronous')))

sync_async_view = {'s': 'synchronous',
                   'a': 'asynchronous'}


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--storage', help='storage location', dest="url")
    parser.add_argument('-e', '--email', help='user email',
                        default="aaa@gmail.com")
    parser.add_argument('-p', '--password', help='user password',
                        default="1234")
    return parser.parse_args(argv)


def pgbench_chart_data(results):
    """
    Format pgbench results for chart
    """
    data = {}
    charts_url = []

    formatted_res = formatters.format_pgbench_stat(results)
    for key, value in formatted_res.items():
        num_cl, num_tr = key.split(' ')
        data.setdefault(num_cl, {}).setdefault(build, {})
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


def render_html_results(ctx, dest):
    charts = []
    for res in ctx.results:
        if res[0] == "io":
            charts.append(build_io_chart(res))

    bars = build_vertical_bar(ctx.results)
    lines = build_lines_chart(ctx.results)

    render_html(bars + lines, dest)


def calc_dev(l):
    sum_res = sum(l)
    mean = sum_res/len(l)
    sum_sq = sum([(r - mean) ** 2 for r in l])
    if len(l) > 1:
        return math.sqrt(sum_sq / (len(l) - 1))
    else:
        return 0


def main():
    from tests.disk_test_agent import parse_output
    out = parse_output(
        open("results/io_scenario_check_th_count.txt").read()).next()
    results = out['res']

    charts_url = []
    charts_data = {}
    for test_name, test_res in results.items():
        blocksize = test_res['blocksize']
        op_type = "sync" if test_res['sync'] else "direct"
        chart_name = "Block size: %s %s" % (blocksize, op_type)
        lat = sum(test_res['lat']) / len(test_res['lat']) / 1000
        lat_dev = calc_dev(test_res['lat'])
        iops = sum(test_res['iops']) / len(test_res['iops'])
        iops_dev = calc_dev(test_res['iops'])
        bw = sum(test_res['bw_mean']) / len(test_res['bw_mean'])
        bw_dev = calc_dev(test_res['bw_mean'])
        conc = test_res['concurence']
        vals = ((lat, lat_dev), (iops, iops_dev), (bw, bw_dev))
        charts_data.setdefault(chart_name, {})[conc] = vals

    for chart_name, chart_data in charts_data.items():
        legend = ["bw"]
        ordered_data = OrderedDict(sorted(chart_data.items(),
                                          key=lambda t: t[0]))

        lat_d, iops_d, bw_d = zip(*ordered_data.values())
        bw_sum = [vals[2][0] * conc for conc, vals in ordered_data.items()]

        chart_url = str(charts.render_vertical_bar(
            chart_name, legend, [bw_d], label_x="KBps",
            scale_x=ordered_data.keys(),
            lines=[(zip(*lat_d)[0], 'msec', 'rr', 'lat'), (bw_sum, None, None, 'bw_sum')]))
        charts_url.append(chart_url)
        render_html(charts_url, "results.html")
    return 0


if __name__ == '__main__':
    exit(main())
