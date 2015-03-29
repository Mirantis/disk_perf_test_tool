import argparse
from collections import OrderedDict

from chart import charts
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


def build_vertical_bar(results):
    data = {}
    charts_url = []

    for build, results in results.items():
        for key, value in results.results.items():
            keys = key.split(' ')
            if not data.get(keys[2]):
                data[keys[2]] = {}
            if not data[keys[2]].get(build):
                data[keys[2]][build] = {}
            data[keys[2]][build][
                ' '.join([keys[0], sync_async_view[keys[1]]])] = value

    for name, value in data.items():
        for op_type, operations in OPERATIONS:
            title = "Block size: " + name
            legend = []
            dataset = []

            scale_x = []

            for build_id, build_results in value.items():
                vals = []

                for key in operations:
                    res = build_results.get(key)
                    if res:
                        vals.append(res)
                        scale_x.append(key)
                if vals:
                    dataset.append(vals)
                    legend.append(build_id)

            if dataset:
                charts_url.append(str(charts.render_vertical_bar
                                  (title, legend, dataset, scale_x=scale_x)))
    return charts_url


def build_lines_chart(results):
    data = {}
    charts_url = []

    for build, results in results.items():
        for key, value in results.results.items():
            keys = key.split(' ')
            if not data.get(' '.join([keys[0], keys[1]])):
                data[' '.join([keys[0], keys[1]])] = {}
            if not data[' '.join([keys[0], keys[1]])].get(build):
                data[' '.join([keys[0], keys[1]])][build] = {}
            data[' '.join([keys[0], keys[1]])][build][keys[2]] = value

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


def render_html(charts_urls):
    templ = open("report.html", 'r').read()
    body = "<div><ol>%s</ol></div>"
    li = "<li><img src='%s'></li>"
    ol = []
    for chart in charts_urls:
        ol.append(li % chart)
    html = templ % {'body': body % '\n'.join(ol)}
    open('results.html', 'w').write(html)


# def report(url, email=None, password=None):
#     results = storage.recent_builds()
#     bars = build_vertical_bar(results)
#     lines = build_lines_chart(results)
#
#     render_html(bars + lines)

#
# def main(argv):
#     opts = parse_args(argv)
#     report(opts.url)
#     return 0
#
#
# if __name__ == '__main__':
#     exit(main(sys.argv[1:]))
