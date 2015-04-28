import json
import requests


def add_test(test_name, test_data, url):
    if not url.endswith("/"):
        url += '/api/tests/' + test_name
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
