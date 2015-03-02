# <koder>: order imports in usual way
import json
import os.path

from logging import getLogger, INFO
from flask import render_template, url_for, make_response, request
from report import build_vertical_bar, build_lines_chart
from web_app import app
from persistance.storage_api import builds_list, prepare_build_data, \
    get_data_for_table, add_data, get_builds_data, \
    get_build_info, get_build_detailed_info
from web_app.app import app
from werkzeug.routing import Rule


def merge_builds(b1, b2):
    d = {}

    for pair in b2.items():
        if pair[0] in b1 and type(pair[1]) is list:
                b1[pair[0]].extend(pair[1])
        else:
            b1[pair[0]] = pair[1]


app.url_map.add(Rule('/', endpoint='index'))
app.url_map.add(Rule('/images/<image_name>', endpoint='get_image'))
app.url_map.add(Rule('/tests/<test_name>', endpoint='render_test'))
app.url_map.add(Rule('/tests/table/<test_name>/', endpoint='render_table'))
app.url_map.add(Rule('/api/tests/<test_name>',
                     endpoint='add_test', methods=['POST']))
app.url_map.add(Rule('/api/tests', endpoint='get_all_tests'))
app.url_map.add(Rule('/api/tests/<test_name>', endpoint='get_test'))


@app.endpoint('index')
def index():
    data = builds_list()

    for elem in data:
        elem['url'] = url_for('render_test', test_name=elem['url'])

    return render_template("index.html", tests=data)


@app.endpoint('get_image')
def get_image(image_name):
    with open("static/images/" + image_name, 'rb') as f:
        image_binary = f.read()

    response = make_response(image_binary)
    response.headers['Content-Type'] = 'image/png'
    response.headers['Content-Disposition'] = 'attachment; filename=img.png'

    return response


@app.endpoint('render_test')
def render_test(test_name):
    results = prepare_build_data(test_name)
    lab_meta = get_build_detailed_info(test_name)

    bars = build_vertical_bar(results)
    lines = build_lines_chart(results)
    urls = bars + lines

    urls = [url_for("get_image", image_name=os.path.basename(url))
            if not url.startswith('http') else url for url in urls]

    return render_template("test.html", urls=urls,
                           table_url=url_for('render_table',
                           test_name=test_name),
                           index_url=url_for('index'), lab_meta=lab_meta)


@app.endpoint('render_table')
def render_table(test_name):
    builds = get_data_for_table(test_name)
    data = get_build_info(test_name)

    header_keys = ['build_id', 'iso_md5', 'type', 'date']
    table = [[]]
    if len(builds) > 0:
        sorted_keys = sorted(builds[0].keys())

        for key in sorted_keys:
            if key not in header_keys:
                header_keys.append(key)

        for test in builds:
            row = []

            for header in header_keys:
                if isinstance(test[header], list):
                    row.append(str(test[header][0]) + unichr(0x00B1)
                               + str(test[header][1]))
                else:
                    row.append(test[header])

            table.append(row)

    return render_template("table.html", headers=header_keys, table=table,
                           back_url=url_for('render_test',
                           test_name=test_name), lab=data)


@app.endpoint('add_test')
def add_test(test_name):
    add_data(request.data)
    return "Created", 201


@app.endpoint('get_all_tests')
def get_all_tests():
    return json.dumps(get_builds_data())


@app.endpoint('get_test')
def get_test(test_name):
    builds = get_builds_data(test_name)

    return json.dumps(builds)


if __name__ == "__main__":
    logger = getLogger("logger")
    app.logger.setLevel(INFO)
    app.logger.addHandler(logger)
    app.run(host='0.0.0.0', debug=True)
