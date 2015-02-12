from urlparse import urlparse
from flask import Flask, render_template, url_for, request, g
from flask_bootstrap import Bootstrap
from config import TEST_PATH
from report import build_vertical_bar, build_lines_chart
from storage_api import create_storage, Measurement
from logging import getLogger, INFO

import json
import os.path
from web_app.keystone import KeystoneAuth

app = Flask(__name__)
Bootstrap(app)


def get_resource_as_string(name, charset='utf-8'):
    with app.open_resource(name) as f:
        return f.read().decode(charset)

app.jinja_env.globals['get_resource_as_string'] = get_resource_as_string


def load_test(test_name):
    test_name += '.json'

    with open(TEST_PATH + "/" + test_name, 'rt') as f:
        raw = f.read()

        if raw != "":
            test = json.loads(raw)
        else:
            test = []
    import time
    creation_time = os.path.getmtime(TEST_PATH + "/" + test_name)

    for t in test:
        t['date'] = time.ctime(creation_time)

    return test


def collect_tests():
    result = []

    for file in os.listdir(TEST_PATH):
        if file.endswith(".json"):
            result.append(file.split('.')[0])

    return result


def collect_builds():
    builds = []
    build_set = set()
    tests = collect_tests()

    for t in tests:
        test = load_test(t)

        for build in test:
            if build["type"] not in build_set:
                build_set.add(build["type"])
                builds.append(build)

    return builds


def builds_list():
    data = []

    for build in collect_builds():
        d = {}
        d["type"] = build['type']
        d["url"] = url_for("render_test", test_name=build['type'])
        d["date"] = build['date']
        d["name"] = build['name']
        data.append(d)

    return data


def create_measurement(build):
    m = Measurement()
    m.build = build.pop("build_id")
    m.build_type = build.pop("type")
    m.md5 = build.pop("iso_md5")
    m.date = build.pop("date")
    m.date = build.pop("name")
    m.results = {k: v for k, v in build.items()}

    return m


def total_lab_info(data):
    d = {}
    d['nodes_count'] = len(data['nodes'])
    d['total_memory'] = 0
    d['total_disk'] = 0
    d['processor_count'] = 0

    for node in data['nodes']:
        d['total_memory'] += node['memory']['total']
        d['processor_count'] += len(node['processors'])

        for disk in node['disks']:
            d['total_disk'] += disk['size']

    to_gb = lambda x: x / (1024 ** 3)
    d['total_memory'] = format(to_gb(d['total_memory']), ',d')
    d['total_disk'] = format(to_gb(d['total_disk']), ',d')
    return d


def collect_lab_data(meta):
    u = urlparse(meta['__meta__'])
    cred = {"username": "admin", "password": "admin", "tenant_name": "admin"}
    keystone = KeystoneAuth(root_url=meta['__meta__'], creds=cred, admin_node_ip=u.hostname)
    lab_info = keystone.do(method='get', path="")
    nodes = []
    result = {}

    for node in lab_info:
        d = {}
        d['name'] = node['name']
        p = []
        i = []
        disks = []
        devices = []

        for processor in node['meta']['cpu']['spec']:
             p.append(processor)

        for iface in node['meta']['interfaces']:
            i.append(iface)

        m = node['meta']['memory'].copy()

        for disk in node['meta']['disks']:
            disks.append(disk)

        d['memory'] = m
        d['disks'] = disks
        d['devices'] = devices
        d['interfaces'] = i
        d['processors'] = p

        nodes.append(d)

    result['nodes'] = nodes
    result['name'] = 'Perf-1 Env'

    return result


@app.route("/", methods=['GET', 'POST'])
def index():
    data = builds_list()
    return render_template("index.html", tests=data)


@app.route("/tests/<test_name>", methods=['GET'])
def render_test(test_name):
    tests = []
    header_keys = ['build_id', 'iso_md5', 'type', 'date']
    table = [[]]

    if test_name == 'GA':
        builds_to_compare = ['GA']
    else:
        builds_to_compare = ['GA', 'master', test_name]

    builds = collect_builds()
    results = {}
    meta = {"__meta__": "http://172.16.52.112:8000/api/nodes"}
    data = collect_lab_data(meta)
    lab_meta = total_lab_info(data)

    for build in builds:
        if build['type'] in builds_to_compare:
            type = build['type']
            m = create_measurement(build)
            results[type] = m

    bars = build_vertical_bar(results)
    lines = build_lines_chart(results)
    urls = bars + lines
    if len(tests) > 0:
        sorted_keys = sorted(tests[0].keys())

        for key in sorted_keys:
            if key not in header_keys:
                header_keys.append(key)

        for test in tests:
            row = []

            for header in header_keys:
                if isinstance(test[header], list):
                    row.append(str(test[header][0]) + unichr(0x00B1) + str(test[header][1]))
                else:
                    row.append(test[header])

            table.append(row)

    return render_template("test.html", urls=urls, table_url=url_for('render_table', test_name=test_name),
                           index_url=url_for('index'), lab_meta=lab_meta)


@app.route("/tests/table/<test_name>/")
def render_table(test_name):
    builds = collect_builds()

    if test_name == 'GA':
        b = ['GA']
    else:
        b = ['GA', 'master', test_name]

    builds = filter(lambda x: x["type"] in b, builds)
    header_keys = ['build_id', 'iso_md5', 'type' ,'date']
    table = [[]]
    meta = {"__meta__": "http://172.16.52.112:8000/api/nodes"}
    data = collect_lab_data(meta)

    if len(builds) > 0:
        sorted_keys = sorted(builds[0].keys())

        for key in sorted_keys:
            if key not in header_keys:
                header_keys.append(key)

        for test in builds:
            row = []

            for header in header_keys:
                if isinstance(test[header], list):
                    row.append(str(test[header][0]) + unichr(0x00B1) + str(test[header][1]))
                else:
                    row.append(test[header])

            table.append(row)

    return render_template("table.html", headers=header_keys, table=table,
                           back_url=url_for('render_test', test_name=test_name), lab=data)


@app.route("/api/tests/<test_name>", methods=['POST'])
def add_test(test_name):
    tests = json.loads(request.data)

    if not hasattr(g, "storage"):
        path = "file://" + TEST_PATH + '/' + test_name + ".json"
        print path
        g.storage = create_storage(path, "", "")

    for test in tests:
        g.storage.store(test)
    return "Created", 201


@app.route("/api/tests", methods=['GET'])
def get_all_tests():
    return json.dumps(collect_builds())


@app.route("/api/tests/<test_name>", methods=['GET'])
def get_test(test_name):
    builds = collect_builds()

    for build in builds:
        if build["type"] == test_name:
            return json.dumps(build)
    return "Not Found", 404


if __name__ == "__main__":
    logger = getLogger("logger")
    app.logger.setLevel(INFO)
    app.logger.addHandler(logger)
    app.run(host='0.0.0.0', debug=True)