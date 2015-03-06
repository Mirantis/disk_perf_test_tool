# class displays measurement. Moved from storage_api_v_1
# to avoid circular imports.
import math


class Measurement(object):
    def __init__(self):
        self.build = ""
        self.build_type = 0  # GA/Master/Other
        self.md5 = ""
        self.name = ""
        self.date = None
        self.results = {
            "": (float, float)
        }

    def __str__(self):
        return self.build + " " + self.build_type + " " + \
            self.md5 + " " + str(self.results)


def mean(l):
    n = len(l)

    return sum(l) / n


def stdev(l):
    m = mean(l)
    return math.sqrt(sum(map(lambda x: (x - m) ** 2, l)))


def process_build_data(build):
    """ Function computes mean of all the data from particular build"""
    for item in build.items():
        if type(item[1]) is list:
            m = mean(item[1])
            s = stdev(item[1])
            build[item[0]] = [m, s]


def create_measurement(data):
    """ Function creates measurement from data was extracted from database."""

    build_data = data[0]

    m = Measurement()
    m.build = build_data.build_id
    m.build_type = build_data.type
    m.name = build_data.name
    m.results = {}

    for i in range(1, len(data), 2):
        result = data[i]
        param_combination = data[i + 1]

        if not str(param_combination) in m.results:
            m.results[str(param_combination)] = [result.bandwith]
        else:
            m.results[str(param_combination)] += [result.bandwith]

    for k in m.results.keys():
        m.results[k] = [mean(m.results[k]), stdev(m.results[k])]

    return m