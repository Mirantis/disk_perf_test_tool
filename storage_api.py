from urlparse import urlparse

import json
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


def create_storage(url, email=None, password=None):
    u = urlparse(url)
    if u.scheme == 'file':
        storage = DiskStorage(u.path)

    return storage


class Storage(object):

    def store(self, data):
        pass

    def retrieve(self, id):
        pass


class DiskStorage(Storage):
    def __init__(self, file_name):
        self.file_name = file_name

        if not os.path.exists(file_name):
            with open(file_name, "w+") as f:
                f.write(json.dumps([]))

    def store(self, data):
        with open(self.file_name, "rt") as f:
            raw_data = f.read()
            document = json.loads(raw_data)
            document.append(data)

        with open(self.file_name, "w+") as f:
            f.write(json.dumps(document))

    def retrieve(self, id):
        with open(self.file_name, "rt") as f:
            raw_data = f.read()
            document = json.loads(raw_data)

            for row in document:
                if row["build_id"] == id:
                    m = Measurement()
                    m.build = row.pop("build_id")
                    m.build_type = row.pop("type")
                    m.md5 = row.pop("iso_md5")
                    m.results = {k.split(" "): row[k] for k in row.keys()}

                    return m
        return None

    def recent_builds(self):
        with open(self.file_name, "rt") as f:
            raw_data = f.read()
            document = json.loads(raw_data)
            d = {}
            result = {}
            build_types = {"GA", "master"}

            for i in range(len(document) - 1, -1, - 1):
                if document[i]["type"] in build_types:
                    if document[i]["type"] not in d:
                        d[document[i]["type"]] = document[i]
                elif "other" not in d:
                    d["other"] = document[i]

            for k in d.keys():
                m = Measurement()
                m.build = d[k].pop("build_id")
                m.build_type = d[k].pop("type")
                m.md5 = d[k].pop("iso_md5")
                m.results = {k: v for k, v in d[k].items()}
                result[m.build_type] = m

        return result


#if __name__ == "__main__":
#    storage = create_storage("file:///home/gstepanov/rally-results-processor/sample.json", "", "")
#    print storage.recent_builds()
