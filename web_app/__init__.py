from flask import Flask, render_template, url_for, request, g
from flask_bootstrap import Bootstrap
import json
import os.path
from storage_api import create_storage, TEST_PATH

app = Flask(__name__)
Bootstrap(app)


def collect_tests():
    result = []

    for file in os.listdir(TEST_PATH):
        if file.endswith(".json"):
            result.append(file.split('.')[0])

    return result


def load_test(test_name):
    test_name += '.json'

    with open("../" + test_name, 'rt') as f:
        raw = f.read()

        if raw == '':
            raise Exception("Test is emoty")

        test = json.loads(raw)

    return test


@app.route("/", methods=['GET', 'POST'])
def index():
    data = []
    for test in collect_tests():
        d = {}
        d["name"] = test
        d["url"] = url_for("render_test", test_name=test)
        data.append(d)

    return render_template("index.html", tests=data)


@app.route("/tests/<test_name>", methods=['GET'])
def render_test(test_name):
    tests = load_test(test_name)
    header_keys = ['build_id', 'iso_md5', 'type']
    table = []

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

    return render_template("table.html", headers=header_keys, table=table)


@app.route("/tests/<test_name>", methods=['POST'])
def add_test(test_name):
    tests = json.loads(request.data)

    if not hasattr(g, "storage"):
        g.storage = create_storage("file://" + os.path.dirname(__file__) + "/test_results/sample.json", "", "")

    for test in tests:
        g.storage.store(test)
    return "Created", 201



if __name__ == "__main__":
    app.run(debug=True)