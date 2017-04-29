"""
This module contains interfaces for storage classes
"""

import os
import re
import abc
import shutil
import sqlite3
import logging
from typing import Any, TypeVar, Type, IO, Tuple, cast, List, Dict, Iterable, Iterator

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper  # type: ignore
except ImportError:
    from yaml import Loader, Dumper  # type: ignore
import numpy

from .common_types import IStorable


logger = logging.getLogger("wally")


class ISimpleStorage(metaclass=abc.ABCMeta):
    """interface for low-level storage, which doesn't support serialization
    and can operate only on bytes"""

    @abc.abstractmethod
    def put(self, value: bytes, path: str) -> None:
        pass

    @abc.abstractmethod
    def get(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def rm(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def get_fd(self, path: str, mode: str = "rb+") -> IO:
        pass

    @abc.abstractmethod
    def get_fname(self, path: str) -> str:
        pass

    @abc.abstractmethod
    def sub_storage(self, path: str) -> 'ISimpleStorage':
        pass

    @abc.abstractmethod
    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        pass


class ITSStorage(metaclass=abc.ABCMeta):
    """interface for low-level storage, which doesn't support serialization
    and can operate only on bytes"""

    @abc.abstractmethod
    def put(self, value: bytes, path: str) -> None:
        pass

    @abc.abstractmethod
    def get(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def rm(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def get_fd(self, path: str, mode: str = "rb+") -> IO:
        pass

    @abc.abstractmethod
    def sub_storage(self, path: str) -> 'ISimpleStorage':
        pass

    @abc.abstractmethod
    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        pass


class ISerializer(metaclass=abc.ABCMeta):
    """Interface for serialization class"""
    @abc.abstractmethod
    def pack(self, value: IStorable) -> bytes:
        pass

    @abc.abstractmethod
    def unpack(self, data: bytes) -> Any:
        pass


class FSStorage(ISimpleStorage):
    """Store all data in files on FS"""

    def __init__(self, root_path: str, existing: bool) -> None:
        self.root_path = root_path
        self.existing = existing
        self.ignored = {'.', '..'}

    def j(self, path: str) -> str:
        return os.path.join(self.root_path, path)

    def put(self, value: bytes, path: str) -> None:
        jpath = self.j(path)
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        with open(jpath, "wb") as fd:
            fd.write(value)

    def get(self, path: str) -> bytes:
        try:
            with open(self.j(path), "rb") as fd:
                return fd.read()
        except FileNotFoundError as exc:
            raise KeyError(path) from exc

    def rm(self, path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.unlink(path)

    def __contains__(self, path: str) -> bool:
        return os.path.exists(self.j(path))

    def get_fname(self, path: str) -> str:
        return self.j(path)

    def get_fd(self, path: str, mode: str = "rb+") -> IO[bytes]:
        jpath = self.j(path)

        if "cb" == mode:
            create_on_fail = True
            mode = "rb+"
            os.makedirs(os.path.dirname(jpath), exist_ok=True)
        elif "ct" == mode:
            create_on_fail = True
            mode = "rt+"
            os.makedirs(os.path.dirname(jpath), exist_ok=True)
        else:
            create_on_fail = False

        try:
            fd = open(jpath, mode)
        except IOError:
            if not create_on_fail:
                raise

            if 't' in mode:
                fd = open(jpath, "wt")
            else:
                fd = open(jpath, "wb")

        return cast(IO[bytes], fd)

    def sub_storage(self, path: str) -> 'FSStorage':
        return self.__class__(self.j(path), self.existing)

    def sync(self):
        pass

    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        path = self.j(path)

        if not os.path.exists(path):
            return

        if not os.path.isdir(path):
            raise OSError("{!r} is not a directory".format(path))

        for fobj in os.scandir(path):
            if fobj.path not in self.ignored:
                if fobj.is_dir():
                    yield False, fobj.name
                else:
                    yield True, fobj.name


class YAMLSerializer(ISerializer):
    """Serialize data to yaml"""
    def pack(self, value: IStorable) -> bytes:
        try:
            return yaml.dump(value, Dumper=Dumper, encoding="utf8")
        except Exception as exc:
            raise ValueError("Can't pickle object {!r} to yaml".format(type(value))) from exc

    def unpack(self, data: bytes) -> Any:
        return yaml.load(data, Loader=Loader)


class SAFEYAMLSerializer(ISerializer):
    """Serialize data to yaml"""
    def pack(self, value: IStorable) -> bytes:
        try:
            return yaml.safe_dump(value, encoding="utf8")
        except Exception as exc:
            raise ValueError("Can't pickle object {!r} to yaml".format(type(value))) from exc

    def unpack(self, data: bytes) -> Any:
        return yaml.safe_load(data)


ObjClass = TypeVar('ObjClass', bound=IStorable)


class _Raise:
    pass


class Storage:
    """interface for storage"""

    def __init__(self, sstorage: ISimpleStorage, serializer: ISerializer) -> None:
        self.sstorage = sstorage
        self.serializer = serializer
        self.cache = {}

    def sub_storage(self, *path: str) -> 'Storage':
        fpath = "/".join(path)
        return self.__class__(self.sstorage.sub_storage(fpath), self.serializer)

    def put(self, value: Any, *path: str) -> None:
        dct_value = cast(IStorable, value).raw() if isinstance(value, IStorable) else value
        serialized = self.serializer.pack(dct_value)  # type: ignore
        fpath = "/".join(path)
        self.sstorage.put(serialized, fpath)

    def put_list(self, value: Iterable[IStorable], *path: str) -> None:
        serialized = self.serializer.pack([obj.raw() for obj in value])  # type: ignore
        fpath = "/".join(path)
        self.sstorage.put(serialized, fpath)

    def get(self, path: str, default: Any = _Raise) -> Any:
        try:
            vl = self.sstorage.get(path)
        except:
            if default is _Raise:
                raise
            return default

        return self.serializer.unpack(vl)

    def rm(self, *path: str) -> None:
        fpath = "/".join(path)
        self.sstorage.rm(fpath)

    def __contains__(self, path: str) -> bool:
        return path in self.sstorage

    def put_raw(self, val: bytes, *path: str) -> str:
        fpath = "/".join(path)
        self.sstorage.put(val, fpath)
        # TODO: dirty hack
        return self.resolve_raw(fpath)

    def resolve_raw(self, fpath) -> str:
        return cast(FSStorage, self.sstorage).j(fpath)

    def get_raw(self, *path: str) -> bytes:
        return self.sstorage.get("/".join(path))

    def append_raw(self, value: bytes, *path: str) -> None:
        with self.sstorage.get_fd("/".join(path), "rb+") as fd:
            fd.seek(0, os.SEEK_END)
            fd.write(value)

    def get_fd(self, path: str, mode: str = "r") -> IO:
        return self.sstorage.get_fd(path, mode)

    def get_fname(self, path: str) -> str:
        return self.sstorage.get_fname(path)

    def load_list(self, obj_class: Type[ObjClass], *path: str) -> List[ObjClass]:
        path_s = "/".join(path)
        if path_s not in self.cache:
            raw_val = cast(List[Dict[str, Any]], self.get(path_s))
            assert isinstance(raw_val, list)
            self.cache[path_s] = [cast(ObjClass, obj_class.fromraw(val)) for val in raw_val]
        return self.cache[path_s]

    def load(self, obj_class: Type[ObjClass], *path: str) -> ObjClass:
        path_s = "/".join(path)
        if path_s not in self.cache:
            self.cache[path_s] = cast(ObjClass, obj_class.fromraw(self.get(path_s)))
        return self.cache[path_s]

    def sync(self) -> None:
        self.sstorage.sync()

    def __enter__(self) -> 'Storage':
        return self

    def __exit__(self, x: Any, y: Any, z: Any) -> None:
        self.sync()

    def list(self, *path: str) -> Iterator[Tuple[bool, str]]:
        return self.sstorage.list("/".join(path))

    def _iter_paths(self,
                    root: str,
                    path_parts: List[str],
                    groups: Dict[str, str]) -> Iterator[Tuple[bool, str, Dict[str, str]]]:

        curr = path_parts[0]
        rest = path_parts[1:]

        for is_file, name in self.list(root):
            if rest and is_file:
                continue

            rr = re.match(pattern=curr + "$", string=name)
            if rr:
                if root:
                    path = root + "/" + name
                else:
                    path = name

                new_groups = rr.groupdict().copy()
                new_groups.update(groups)

                if rest:
                    yield from self._iter_paths(path, rest, new_groups)
                else:
                    yield is_file, path, new_groups


def make_storage(url: str, existing: bool = False) -> Storage:
    return Storage(FSStorage(url, existing), SAFEYAMLSerializer())

