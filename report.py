import argparse
from collections import OrderedDict
import sys

import charts
import storage_api


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--storage', help='storage location', dest="url")
    parser.add_argument('-e', '--email', help='user email',
                        default="aaa@gmail.com")
    parser.add_argument('-p', '--password', help='user password',
                        default="1234")
    return parser.parse_args(argv)


def report(url, email=None, password=None):
    storage = storage_api.create_storage(url, email, password)
    results = storage.recent_builds()

    data = {}

    # render vertical bar
    for build, results in results.items():
        for key, value in results.items():
            keys = key.split(' ')
            if not data.get(keys[2]):
                data[keys[2]] = {}
            if not data[keys[2]].get(build):
                data[keys[2]][build] = {}
            data[keys[2]][build][' '.join([keys[0], keys[1]])] = value

    for name, value in data.items():
        title = name
        legend = []
        dataset = []
        scale_x = []
        for build_id, build_results in value.items():
            legend.append(build_id)
            ordered_build_results = OrderedDict(sorted(build_results.items(),
                                                key=lambda t: t[0]))
            if not scale_x:
                scale_x = ordered_build_results.keys()
            dataset.append(ordered_build_results.values())

        bar = charts.render_vertical_bar(title, legend, dataset,
                                         scale_x=scale_x)
        print "Vertical bar for %s:\n %s" % (name, str(bar))


def main(argv):
    opts = parse_args(argv)
    report(opts.url)
    return 0


if __name__ == '__main__':
    exit(main(sys.argv[1:]))