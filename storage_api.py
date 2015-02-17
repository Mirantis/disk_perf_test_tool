from urlparse import urlparse

import json
import math
from config import TEST_PATH
from flask import url_for
import os

class Measurement(object):
    def __init__(self):
        self.build = ""
        self.build_type = 0  # GA/Master/Other
        self.md5 = ""
        self.results = {
            "": (float, float)
        }

    def __str__(self):
        return self.build + " " + self.build_type + " " + \
            self.md5 + " " + str(self.results)


def prepare_build_data(build):
    for item in build.items():
        if type(item[1]) is list:
            m = mean(item[1])
            s = stdev(item[1])
            build[item[0]] = [m, s]

            
def mean(l):
    n = len(l)

    return sum(l) / n


def stdev(l):
    m = mean(l)
    return math.sqrt(sum(map(lambda x: (x - m) ** 2, l)))


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

    for build in builds:
        prepare_build_data(build)

    return builds


def builds_list():
    data = []

    for build in collect_builds():
        d = {}
        d["type"] = build['type']
        d["url"] = url_for("render_test", test_name=build['name'])
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



collect_builds()