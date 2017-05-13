from typing import Any, Dict, NamedTuple

from cephlib.storage import IStorable


IP = str
IPAddr = NamedTuple("IPAddr", [("host", IP), ("port", int)])


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
