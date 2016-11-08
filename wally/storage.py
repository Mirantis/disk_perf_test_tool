"""
This module contains interfaces for storage classes
"""

import abc
from typing import Any, Iterable, TypeVar, Type, IO


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


class IStorage(metaclass=abc.ABCMeta):
    """interface for storage"""
    @abc.abstractmethod
    def __init__(self, path: str, existing_storage: bool = False) -> None:
        pass

    @abc.abstractmethod
    def __setitem__(self, path: str, value: IStorable) -> None:
        pass

    @abc.abstractmethod
    def __getitem__(self, path: str) -> IStorable:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def list(self, path: str) -> Iterable[str]:
        pass

    @abc.abstractmethod
    def load(self, path: str, obj_class: Type[ObjClass]) -> ObjClass:
        pass

    @abc.abstractmethod
    def get_stream(self, path: str) -> IO:
        pass


class ISimpleStorage(metaclass=abc.ABCMeta):
    """interface for low-level storage, which doesn't support serialization
    and can operate only on bytes"""

    @abc.abstractmethod
    def __init__(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def __setitem__(self, path: str, value: bytes) -> None:
        pass

    @abc.abstractmethod
    def __getitem__(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def list(self, path: str) -> Iterable[str]:
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


# TODO(koder): this is concrete storage and serializer classes to be implemented
class FSStorage(IStorage):
    """Store all data in files on FS"""

    @abc.abstractmethod
    def __init__(self, root_path: str, serializer: ISerializer, existing: bool = False) -> None:
        pass


class YAMLSerializer(ISerializer):
    """Serialize data to yaml"""
    pass


def make_storage(url: str, existing: bool = False) -> IStorage:
    return FSStorage(url, YAMLSerializer(), existing)

