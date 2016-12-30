import re
import getpass
import logging
from typing import List, Dict


from . import utils
from .common_types import ConnCreds


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


