import json
import requests


def add_test(test_name, test_data, url):
    if not url.endswith("/"):
        url += '/api/tests/' + test_name

    import pdb
    pdb.set_trace()
    requests.post(url=url, data=json.dumps(test_data))


def get_test(test_name, url):
    if not url.endswith("/"):
        url += '/api/tests/' + test_name

    result = requests.get(url=url)

    return json.loads(result.content)


def get_all_tests(url):
    if not url.endswith('/'):
        url += '/api/tests'

    result = requests.get(url=url)
    return json.loads(result.content)


if __name__ == '__main__':
    url = "http://0.0.0.0:5000/api/tests"
    test = get_test("GA", url=url)
    print test

    tests = get_all_tests(url=url)
    print tests

    # test["type"] = "some build name"
    # add_test("bla_bla", [test], url=url)

    tests = get_all_tests(url=url)
    print len(tests)
