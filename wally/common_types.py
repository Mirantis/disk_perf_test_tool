import abc
from typing import Any, Union, List, Dict, NamedTuple


IP = str
IPAddr = NamedTuple("IPAddr", [("host", IP), ("port", int)])


class IStorable(metaclass=abc.ABCMeta):
    """Interface for type, which can be stored"""

    @abc.abstractmethod
    def raw(self) -> Dict[str, Any]:
        pass

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'IStorable':
        pass


Basic = Union[int, str, bytes, bool, None]
StorableType = Union[IStorable, Dict[str, Any], List[Any], int, str, bytes, bool, None]


class Storable(IStorable):
    """Default implementation"""

    def raw(self) -> Dict[str, Any]:
        return {name: val for name, val in self.__dict__.items() if not name.startswith("_")}

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'IStorable':
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj


class ConnCreds(IStorable):
    def __init__(self, host: str, user: str, passwd: str = None, port: str = '22',
                 key_file: str = None, key: bytes = None) -> None:
        self.user = user
        self.passwd = passwd
        self.addr = IPAddr(host, int(port))
        self.key_file = key_file
        self.key = key

    def __str__(self) -> str:
        return "{}@{}:{}".format(self.user, self.addr.host, self.addr.port)

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        return {
            'user': self.user,
            'host': self.addr.host,
            'port': self.addr.port,
            'passwd': self.passwd,
            'key_file': self.key_file
        }

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'ConnCreds':
        return cls(**data)
