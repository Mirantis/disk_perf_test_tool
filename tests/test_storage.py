import yaml
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
                storage.put(val, path)

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
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
                storage.put(val, *path.split('/'))

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                ok(storage.get(*path.split('/'))) == val
                ok(storage.get(path)) == val

    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            for path, val in values.items():
                storage.put(val, path)

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                ok(storage.get(path)) == val
                ok(storage.get(*path.split('/'))) == val


def test_overwrite():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage.put("1", "some_path")
            storage.put([1, 2, 3], "some_path")

        with make_storage(root, existing=True) as storage:
            assert storage.get("some_path") == [1, 2, 3]


def test_multy_level():
    with in_temp_dir() as root:
        values = {
            "dict1": {1: {3: 4, 6: [12, {123, 3}, {4: 3}]}, "2": "4", "1.2": 1.3}
        }

        with make_storage(root, existing=False) as storage:
            for path, val in values.items():
                storage.put(val, path)

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                ok(storage.get(path)) == val


def test_arrays():
    with in_temp_dir() as root:
        val_l = list(range(10000)) * 10
        val_i = array.array("i", val_l)
        val_f = array.array("f", map(float, val_l))
        val_2f = val_f + val_f

        with make_storage(root, existing=False) as storage:
            storage.put_array(val_i, "array_i")
            storage.put_array(val_f, "array_f")
            storage.put_array(val_f, "array_x2")
            storage.append(val_f, "array_x2")

        with make_storage(root, existing=True) as storage:
            ok(val_i) == storage.get_array("i", "array_i")
            ok(val_f) == storage.get_array("f", "array_f")
            ok(val_2f) == storage.get_array("f", "array_x2")


class LoadMe(yaml.YAMLObject):
    yaml_tag = '!LoadMe'

    def __init__(self, **vals):
        self.__dict__.update(vals)

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_mapping(cls.yaml_tag, data.__dict__)

    @classmethod
    def from_yaml(cls, loader, node):
        return LoadMe(**loader.construct_mapping(node))


def test_load_user_obj():
    obj = LoadMe(x=1, y=12, z=[1,2,3], t="asdad", gg={"a": 1, "g": [["x"]]})

    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage.put(obj, "obj")

        with make_storage(root, existing=True) as storage:
            obj2 = storage.load(LoadMe, "obj")
            assert isinstance(obj2, LoadMe)
            ok(obj2.__dict__) == obj.__dict__


def test_path_not_exists():
    with in_temp_dir() as root:
        pass

    with make_storage(root, existing=False) as storage:
        with pytest.raises(KeyError):
            storage.get("x")


def test_substorage():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage.put("data", "x/y")
            storage.sub_storage("t").put("sub_data", "r")

        with make_storage(root, existing=True) as storage:
            ok(storage.get("t/r")) == "sub_data"
            ok(storage.sub_storage("x").get("y")) == "data"
