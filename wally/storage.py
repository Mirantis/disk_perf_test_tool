"""
This module contains interfaces for storage classes
"""

import os
import abc
import array
import shutil
from typing import Any, Iterator, TypeVar, Type, IO, Tuple, cast, List, Dict, Union, Iterable


import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper  # type: ignore
except ImportError:
    from yaml import Loader, Dumper  # type: ignore


class IStorable(metaclass=abc.ABCMeta):
    """Interface for type, which can be stored"""

basic_types = {list, dict, tuple, set, type(None), int, str, bytes, bool, float}
for btype in basic_types:
    # pylint: disable=E1101
    IStorable.register(btype)  # type: ignore


ObjClass = TypeVar('ObjClass')


class ISimpleStorage(metaclass=abc.ABCMeta):
    """interface for low-level storage, which doesn't support serialization
    and can operate only on bytes"""

    @abc.abstractmethod
    def __setitem__(self, path: str, value: bytes) -> None:
        pass

    @abc.abstractmethod
    def __getitem__(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def __delitem__(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        pass

    @abc.abstractmethod
    def get_stream(self, path: str, mode: str = "rb+") -> IO:
        pass

    @abc.abstractmethod
    def sub_storage(self, path: str) -> 'ISimpleStorage':
        pass

    @abc.abstractmethod
    def clear(self, path: str) -> None:
        pass


class ISerializer(metaclass=abc.ABCMeta):
    """Interface for serialization class"""
    @abc.abstractmethod
    def pack(self, value: IStorable) -> bytes:
        pass

    @abc.abstractmethod
    def unpack(self, data: bytes) -> IStorable:
        pass


class FSStorage(ISimpleStorage):
    """Store all data in files on FS"""

    def __init__(self, root_path: str, existing: bool) -> None:
        self.root_path = root_path
        self.existing = existing
        if existing:
            if not os.path.isdir(self.root_path):
                raise IOError("No storage found at {!r}".format(root_path))

    def j(self, path: str) -> str:
        return os.path.join(self.root_path, path)

    def __setitem__(self, path: str, value: bytes) -> None:
        jpath = self.j(path)
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        with open(jpath, "wb") as fd:
            fd.write(value)

    def __delitem__(self, path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def __getitem__(self, path: str) -> bytes:
        with open(self.j(path), "rb") as fd:
            return fd.read()

    def __contains__(self, path: str) -> bool:
        return os.path.exists(self.j(path))

    def list(self, path: str = "") -> Iterator[Tuple[bool, str]]:
        jpath = self.j(path)
        if not os.path.exists(jpath):
            return

        for entry in os.scandir(jpath):
            if not entry.name in ('..', '.'):
                yield entry.is_file(), entry.name

    def get_stream(self, path: str, mode: str = "rb+") -> IO[bytes]:
        jpath = self.j(path)

        if "cb" == mode:
            create_on_fail = True
            mode = "rb+"
        else:
            create_on_fail = False

        os.makedirs(os.path.dirname(jpath), exist_ok=True)

        try:
            fd = open(jpath, mode)
        except IOError:
            if not create_on_fail:
                raise
            fd = open(jpath, "wb")

        return cast(IO[bytes], fd)

    def sub_storage(self, path: str) -> 'FSStorage':
        return self.__class__(self.j(path), self.existing)

    def clear(self, path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(self.j(path))


class YAMLSerializer(ISerializer):
    """Serialize data to yaml"""
    def pack(self, value: Any) -> bytes:
        if type(value) not in basic_types:
            # for name, val in value.__dict__.items():
            #     if type(val) not in basic_types:
            #         raise ValueError(("Can't pack {!r}. Attribute {} has value {!r} (type: {}), but only" +
            #                           " basic types accepted as attributes").format(value, name, val, type(val)))
            value = value.__dict__
        return yaml.dump(value, Dumper=Dumper, encoding="utf8")

    def unpack(self, data: bytes) -> IStorable:
        return yaml.load(data, Loader=Loader)


class Storage:
    """interface for storage"""
    def __init__(self, storage: ISimpleStorage, serializer: ISerializer) -> None:
        self.storage = storage
        self.serializer = serializer

    def sub_storage(self, *path: str) -> 'Storage':
        return self.__class__(self.storage.sub_storage("/".join(path)), self.serializer)

    def __setitem__(self, path: Union[str, Iterable[str]], value: Any) -> None:
        if not isinstance(path, str):
            path = "/".join(path)

        self.storage[path] = self.serializer.pack(cast(IStorable, value))

    def __getitem__(self, path: Union[str, Iterable[str]]) -> IStorable:
        if not isinstance(path, str):
            path = "/".join(path)

        return self.serializer.unpack(self.storage[path])

    def __delitem__(self, path: Union[str, Iterable[str]]) -> None:
        if not isinstance(path, str):
            path = "/".join(path)
        del self.storage[path]

    def __contains__(self, path: Union[str, Iterable[str]]) -> bool:
        if not isinstance(path, str):
            path = "/".join(path)
        return path in self.storage

    def store_raw(self, val: bytes, *path: str) -> None:
        self.storage["/".join(path)] = val

    def clear(self, *path: str) -> None:
        self.storage.clear("/".join(path))

    def get_raw(self, *path: str) -> bytes:
        return self.storage["/".join(path)]

    def list(self, *path: str) -> Iterator[Tuple[bool, str]]:
        return self.storage.list("/".join(path))

    def set_array(self, value: array.array, *path: str) -> None:
        with self.get_stream("/".join(path), "wb") as fd:
            value.tofile(fd)  # type: ignore

    def get_array(self, typecode: str, *path: str) -> array.array:
        res = array.array(typecode)
        path_s = "/".join(path)
        with self.get_stream(path_s, "rb") as fd:
            fd.seek(0, os.SEEK_END)
            size = fd.tell()
            fd.seek(0, os.SEEK_SET)
            assert size % res.itemsize == 0, "Storage object at path {} contains no array of {} or corrupted."\
                .format(path_s, typecode)
            res.fromfile(fd, size // res.itemsize)  # type: ignore
        return res

    def append(self, value: array.array, *path: str) -> None:
        with self.get_stream("/".join(path), "cb") as fd:
            fd.seek(0, os.SEEK_END)
            value.tofile(fd)  # type: ignore

    def construct(self, path: str, raw_val: Dict, obj_class: Type[ObjClass]) -> ObjClass:
        "Internal function, used to construct user type from raw unpacked value"
        if obj_class in (int, str, dict, list, None):
            raise ValueError("Can't load into build-in value - {!r} into type {}")

        if not isinstance(raw_val, dict):
            raise ValueError("Can't load path {!r} into python type. Raw value not dict".format(path))

        if not all(isinstance(key, str) for key in raw_val.keys()):
            raise ValueError("Can't load path {!r} into python type.".format(path) +
                             "Raw not all keys in raw value is strings")

        obj = obj_class.__new__(obj_class)  # type: ObjClass
        obj.__dict__.update(raw_val)
        return obj

    def load_list(self, obj_class: Type[ObjClass], *path: str) -> List[ObjClass]:
        path_s = "/".join(path)
        raw_val = self[path_s]
        assert isinstance(raw_val, list)
        return [self.construct(path_s, val, obj_class) for val in cast(list, raw_val)]

    def load(self, obj_class: Type[ObjClass], *path: str) -> ObjClass:
        path_s = "/".join(path)
        return self.construct(path_s, cast(Dict, self[path_s]), obj_class)

    def get_stream(self, path: str, mode: str = "r") -> IO:
        return self.storage.get_stream(path, mode)

    def get(self, path: Union[str, Iterable[str]], default: Any = None) -> Any:
        if not isinstance(path, str):
            path = "/".join(path)

        try:
            return self[path]
        except Exception:
            return default

    def __enter__(self) -> 'Storage':
        return self

    def __exit__(self, x: Any, y: Any, z: Any) -> None:
        return


def make_storage(url: str, existing: bool = False) -> Storage:
    return Storage(FSStorage(url, existing), YAMLSerializer())

