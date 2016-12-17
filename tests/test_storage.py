import array
import shutil
import tempfile
import contextlib


import pytest
from oktest import ok


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
                ok(storage[path]) == val
                ok(storage.get(path)) == val


def test_path_list():
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

    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            for path, val in values.items():
                storage[path.split('/')] = val

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                ok(storage[path.split('/')]) == val
                ok(storage.get(path.split('/'))) == val

    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            for path, val in values.items():
                storage[path] = val

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                ok(storage[path.split('/')]) == val
                ok(storage.get(path.split('/'))) == val


def test_list():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage["x/some_path1"] = "1"
            storage["x/some_path2"] = [1, 2, 3]
            storage["x/some_path3"] = [1, 2, 3, 4]

            storage["x/y/some_path11"] = "1"
            storage["x/y/some_path22"] = [1, 2, 3]

        with make_storage(root, existing=True) as storage:
            assert 'x' in storage
            assert 'x/y' in storage

            assert {(False, 'x')} == set(storage.list())

            assert {(True, 'some_path1'),
                    (True, 'some_path2'),
                    (True, 'some_path3'),
                    (False, "y")} == set(storage.list("x"))

            assert {(True, 'some_path11'), (True, 'some_path22')} == set(storage.list("x/y"))


def test_overwrite():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage["some_path"] = "1"
            storage["some_path"] = [1, 2, 3]

        with make_storage(root, existing=True) as storage:
            assert storage["some_path"] == [1, 2, 3]


def test_multy_level():
    with in_temp_dir() as root:
        values = {
            "dict1": {1: {3: 4, 6: [12, {123, 3}, {4: 3}]}, "2": "4", "1.2": 1.3}
        }

        with make_storage(root, existing=False) as storage:
            for path, val in values.items():
                storage[path] = val

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                ok(storage[path]) == val


def test_arrays():
    with in_temp_dir() as root:
        val_l = list(range(10000)) * 10
        val_i = array.array("i", val_l)
        val_f = array.array("f", map(float, val_l))
        val_2f = val_f + val_f

        with make_storage(root, existing=False) as storage:
            storage.set_array(val_i, "array_i")
            storage.set_array(val_f, "array_f")
            storage.set_array(val_f, "array_x2")
            storage.append(val_f, "array_x2")

        with make_storage(root, existing=True) as storage:
            ok(val_i) == storage.get_array("i", "array_i")
            ok(val_f) == storage.get_array("f", "array_f")
            ok(val_2f) == storage.get_array("f", "array_x2")


class LoadMe:
    def __init__(self, **vals):
        self.__dict__.update(vals)


def test_load_user_obj():
    obj = LoadMe(x=1, y=12, z=[1,2,3], t="asdad", gg={"a": 1, "g": [["x"]]})

    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage["obj"] = obj

        with make_storage(root, existing=True) as storage:
            obj2 = storage.load(LoadMe, "obj")
            assert isinstance(obj2, LoadMe)
            ok(obj2.__dict__) == obj.__dict__


def test_path_not_exists():
    with in_temp_dir() as root:
        pass

    with pytest.raises(IOError):
        with make_storage(root, existing=True) as storage:
            pass

    with in_temp_dir() as root:
        pass

    with make_storage(root, existing=False) as storage:
        with pytest.raises(IOError):
            storage["x"]


def test_incorrect_user_object():
    obj = LoadMe(x=1, y=LoadMe(t=12))

    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            with pytest.raises(ValueError):
                storage["obj"] = obj


def test_substorage():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage["x/y"] = "data"
            storage.sub_storage("t")["r"] = "sub_data"

        with make_storage(root, existing=True) as storage:
            ok(storage["t/r"]) == "sub_data"
            ok(storage.sub_storage("x")["y"]) == "data"
