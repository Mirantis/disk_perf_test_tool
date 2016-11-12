import re
import time
import errno
import socket
import logging
import os.path
import getpass
import selectors
from io import BytesIO
from typing import Union, Optional, cast, Dict, List, Tuple

import paramiko

from . import utils


logger = logging.getLogger("wally")
IPAddr = Tuple[str, int]


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


class ConnCreds:
    conn_uri_attrs = ("user", "passwd", "host", "port", "key_file")

    def __init__(self, host: str, user: str, passwd: str = None, port: int = 22, key_file: str = None) -> None:
        self.user = user
        self.passwd = passwd
        self.host = host
        self.port = port
        self.key_file = key_file

    def __str__(self) -> str:
        return str(self.__dict__)


def parse_ssh_uri(uri: str) -> ConnCreds:
    """Parse ssh connection URL from one of following form
        [ssh://]user:passwd@host[:port]
        [ssh://][user@]host[:port][:key_file]
    """

    if uri.startswith("ssh://"):
        uri = uri[len("ssh://"):]

    res = ConnCreds("", getpass.getuser())

    for rr in URIsNamespace.uri_reg_exprs:
        rrm = re.match(rr, uri)
        if rrm is not None:
            res.__dict__.update(rrm.groupdict())
            return res

    raise ValueError("Can't parse {0!r} as ssh uri value".format(uri))


NODE_KEYS = {}  # type: Dict[IPAddr, paramiko.RSAKey]


def set_key_for_node(host_port: IPAddr, key: bytes) -> None:
    with BytesIO(key) as sio:
        NODE_KEYS[host_port] = paramiko.RSAKey.from_private_key(sio)


def ssh_connect(creds: ConnCreds,
                conn_timeout: int = 60,
                tcp_timeout: int = 15,
                default_banner_timeout: int = 30) -> Tuple[paramiko.SSHClient, str, str]:

    ssh = paramiko.SSHClient()
    ssh.load_host_keys('/dev/null')
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.known_hosts = None

    end_time = time.time() + conn_timeout  # type: float

    while True:
        try:
            time_left = end_time - time.time()
            c_tcp_timeout = min(tcp_timeout, time_left)

            banner_timeout_arg = {}  # type: Dict[str, int]
            if paramiko.__version_info__ >= (1, 15, 2):
                banner_timeout_arg['banner_timeout'] = int(min(default_banner_timeout, time_left))

            if creds.passwd is not None:
                ssh.connect(creds.host,
                            timeout=c_tcp_timeout,
                            username=creds.user,
                            password=cast(str, creds.passwd),
                            port=creds.port,
                            allow_agent=False,
                            look_for_keys=False,
                            **banner_timeout_arg)
            elif creds.key_file is not None:
                ssh.connect(creds.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            key_filename=cast(str, creds.key_file),
                            look_for_keys=False,
                            port=creds.port,
                            **banner_timeout_arg)
            elif (creds.host, creds.port) in NODE_KEYS:
                ssh.connect(creds.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            pkey=NODE_KEYS[(creds.host, creds.port)],
                            look_for_keys=False,
                            port=creds.port,
                            **banner_timeout_arg)
            else:
                key_file = os.path.expanduser('~/.ssh/id_rsa')
                ssh.connect(creds.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            key_filename=key_file,
                            look_for_keys=False,
                            port=creds.port,
                            **banner_timeout_arg)
            return ssh, "{0.host}:{0.port}".format(creds), creds.host
        except paramiko.PasswordRequiredException:
            raise
        except (socket.error, paramiko.SSHException):
            if time.time() > end_time:
                raise
            time.sleep(1)


def wait_ssh_available(addrs: List[IPAddr],
                       timeout: int = 300,
                       tcp_timeout: float = 1.0) -> None:
    addrs = set(addrs)
    for _ in utils.Timeout(timeout):
        with selectors.DefaultSelector() as selector:  # type: selectors.BaseSelector
            for addr in addrs:
                sock = socket.socket()
                sock.setblocking(False)
                try:
                    sock.connect(addr)
                except BlockingIOError:
                    pass
                selector.register(sock, selectors.EVENT_READ, data=addr)

            etime = time.time() + tcp_timeout
            ltime = etime - time.time()
            while ltime > 0:
                for key, _ in selector.select(timeout=ltime):
                    selector.unregister(key.fileobj)
                    try:
                        key.fileobj.getpeername()
                        addrs.remove(key.data)
                    except OSError as exc:
                        if exc.errno == errno.ENOTCONN:
                            pass
                ltime = etime - time.time()

        if not addrs:
            break


