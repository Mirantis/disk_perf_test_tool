import re
import getpass
import logging
from typing import List, Dict, Any


from . import utils
from .common_types import IPAddr
from .result_classes import IStorable


logger = logging.getLogger("wally")


class URIsNamespace:
    class ReParts:
        user_rr = "[^:]*?"
        host_rr = "[^:@]*?"
        port_rr = "\\d+"
        key_file_rr = "[^:@]*"
        passwd_rr = ".*?"

    re_dct = ReParts.__dict__

    for attr_name, val in re_dct.items():
        if attr_name.endswith('_rr'):
            new_rr = "(?P<{0}>{1})".format(attr_name[:-3], val)
            setattr(ReParts, attr_name, new_rr)

    re_dct = ReParts.__dict__

    templs = [
        "^{host_rr}$",
        "^{host_rr}:{port_rr}$",
        "^{host_rr}::{key_file_rr}$",
        "^{host_rr}:{port_rr}:{key_file_rr}$",
        "^{user_rr}@{host_rr}$",
        "^{user_rr}@{host_rr}:{port_rr}$",
        "^{user_rr}@{host_rr}::{key_file_rr}$",
        "^{user_rr}@{host_rr}:{port_rr}:{key_file_rr}$",
        "^{user_rr}:{passwd_rr}@{host_rr}$",
        "^{user_rr}:{passwd_rr}@{host_rr}:{port_rr}$",
    ]

    uri_reg_exprs = []  # type: List[str]
    for templ in templs:
        uri_reg_exprs.append(templ.format(**re_dct))


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
    def fromraw(cls, data) -> 'ConnCreds':
        return cls(**data)


def parse_ssh_uri(uri: str) -> ConnCreds:
    """Parse ssh connection URL from one of following form
        [ssh://]user:passwd@host[:port]
        [ssh://][user@]host[:port][:key_file]
    """

    if uri.startswith("ssh://"):
        uri = uri[len("ssh://"):]

    for rr in URIsNamespace.uri_reg_exprs:
        rrm = re.match(rr, uri)
        if rrm is not None:
            params = {"user": getpass.getuser()}  # type: Dict[str, str]
            params.update(rrm.groupdict())
            params['host'] = utils.to_ip(params['host'])
            return ConnCreds(**params)  # type: ignore

    raise ValueError("Can't parse {0!r} as ssh uri value".format(uri))


