"""
This module contains interfaces for storage classes
"""

import os
import abc
from typing import Any, Iterable, TypeVar, Type, IO, Tuple, cast, List


class IStorable(metaclass=abc.ABCMeta):
    """Interface for type, which can be stored"""
    @abc.abstractmethod
    def __getstate__(self) -> Any:
        pass

    @abc.abstractmethod
    def __setstate__(self, Any):
        pass


# all builtin types can be stored
IStorable.register(list)  # type: ignore
IStorable.register(dict)  # type: ignore
IStorable.register(tuple)  # type: ignore
IStorable.register(set)  # type: ignore
IStorable.register(None)  # type: ignore
IStorable.register(int)  # type: ignore
IStorable.register(str)  # type: ignore
IStorable.register(bytes)  # type: ignore
IStorable.register(bool)  # type: ignore


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
    def list(self, path: str) -> Iterable[Tuple[bool, str]]:
        pass

    @abc.abstractmethod
    def get_stream(self, path: str) -> IO:
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
        if existing:
            if not os.path.isdir(self.root_path):
                raise ValueError("No storage found at {!r}".format(root_path))

    def __setitem__(self, path: str, value: bytes) -> None:
        path = os.path.join(self.root_path, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fd:
            fd.write(value)

    def __delitem__(self, path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def __getitem__(self, path: str) -> bytes:
        path = os.path.join(self.root_path, path)
        with open(path, "rb") as fd:
            return fd.read()

    def __contains__(self, path: str) -> bool:
        path = os.path.join(self.root_path, path)
        return os.path.exists(path)

    def list(self, path: str) -> Iterable[Tuple[bool, str]]:
        path = os.path.join(self.root_path, path)
        for entry in os.scandir(path):
            if not entry.name in ('..', '.'):
                yield entry.is_file(), entry.name

    def get_stream(self, path: str, mode: str = "rb") -> IO:
        path = os.path.join(self.root_path, path)
        return open(path, mode)


class YAMLSerializer(ISerializer):
    """Serialize data to yaml"""
    def pack(self, value: IStorable) -> bytes:
        raise NotImplementedError()

    def unpack(self, data: bytes) -> IStorable:
        raise NotImplementedError()


class Storage:
    """interface for storage"""
    def __init__(self, storage: ISimpleStorage, serializer: ISerializer) -> None:
        self.storage = storage
        self.serializer = serializer

    def __setitem__(self, path: str, value: IStorable) -> None:
        self.storage[path] = self.serializer.pack(value)

    def __getitem__(self, path: str) -> IStorable:
        return self.serializer.unpack(self.storage[path])

    def __delitem__(self, path: str) -> None:
        del self.storage[path]

    def __contains__(self, path: str) -> bool:
        return path in self.storage

    def list(self, path: str) -> Iterable[Tuple[bool, str]]:
        return self.storage.list(path)

    def construct(self, path: str, raw_val: IStorable, obj_class: Type[ObjClass]) -> ObjClass:
        if obj_class in (int, str, dict, list, None):
            if not isinstance(raw_val, obj_class):
                raise ValueError("Can't load path {!r} into type {}. Real type is {}"
                                 .format(path, obj_class, type(raw_val)))
            return cast(ObjClass, raw_val)

        if not isinstance(raw_val, dict):
            raise ValueError("Can't load path {!r} into python type. Raw value not dict".format(path))

        if not all(isinstance(str, key) for key in raw_val.keys):
            raise ValueError("Can't load path {!r} into python type.".format(path) +
                             "Raw not all keys in raw value is strings")

        obj = obj_class.__new__(obj_class)  # type: ObjClass
        obj.__dict__.update(raw_val)
        return obj

    def load_list(self, path: str, obj_class: Type[ObjClass]) -> List[ObjClass]:
        raw_val = self[path]
        assert isinstance(raw_val, list)
        return [self.construct(path, val, obj_class) for val in cast(list, raw_val)]

    def load(self, path: str, obj_class: Type[ObjClass]) -> ObjClass:
        return self.construct(path, self[path], obj_class)

    def get_stream(self, path: str) -> IO:
        return self.storage.get_stream(path)

    def get(self, path: str, default: Any = None) -> Any:
        try:
            return self[path]
        except KeyError:
            return default


def make_storage(url: str, existing: bool = False) -> Storage:
    return Storage(FSStorage(url, existing), YAMLSerializer())

