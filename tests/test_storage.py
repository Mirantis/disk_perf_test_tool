import shutil
import tempfile
import contextlib


from oktest import ok, main, test


from wally.storage import make_storage


@contextlib.contextmanager
def in_temp_dir():
    dname = tempfile.mkdtemp()
    try:
        yield dname
    finally:
        shutil.rmtree(dname)


def test_basic():
    with in_temp_dir() as root:
        values = {
            "int": 1,
            "str/1": "test",
            "bytes/2": b"test",
            "none/s/1": None,
            "bool/xx/1/2/1": None,
            "float/s/1": 1.234,
            "list": [1, 2, "3"],
            "dict": {1: 3, "2": "4", "1.2": 1.3}
        }

        with make_storage(root, existing=False) as storage:
            for path, val in values.items():
                storage[path] = val


        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                ok(storage[path])  == val


def test_large_arrays():
    pass


def test_array_append():
    pass


def test_performance():
    pass
