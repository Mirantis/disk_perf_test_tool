import abc
from typing import Any, Union, List, Dict


class IStorable(metaclass=abc.ABCMeta):
    """Interface for type, which can be stored"""

    @abc.abstractmethod
    def raw(self) -> Dict[str, Any]:
        pass

    @abc.abstractclassmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'IStorable':
        pass


class Storable(IStorable):
    """Default implementation"""

    def raw(self) -> Dict[str, Any]:
        return self.__dict__

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'IStorable':
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj


Basic = Union[int, str, bytes, bool, None]
StorableType = Union[IStorable, Dict[str, Any], List[Any], int, str, bytes, bool, None]
