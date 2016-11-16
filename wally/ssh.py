import time
import errno
import socket
import logging
import os.path
import selectors
from io import BytesIO
from typing import cast, Dict, List, Set

import paramiko

from . import utils
from .ssh_utils import ConnCreds, IPAddr

logger = logging.getLogger("wally")
NODE_KEYS = {}  # type: Dict[IPAddr, paramiko.RSAKey]


def set_key_for_node(host_port: IPAddr, key: bytes) -> None:
    with BytesIO(key) as sio:
        NODE_KEYS[host_port] = paramiko.RSAKey.from_private_key(sio)


def connect(creds: ConnCreds,
            conn_timeout: int = 60,
            tcp_timeout: int = 15,
            default_banner_timeout: int = 30) -> paramiko.SSHClient:

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
                ssh.connect(creds.addr.host,
                            timeout=c_tcp_timeout,
                            username=creds.user,
                            password=cast(str, creds.passwd),
                            port=creds.addr.port,
                            allow_agent=False,
                            look_for_keys=False,
                            **banner_timeout_arg)
            elif creds.key_file is not None:
                ssh.connect(creds.addr.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            key_filename=cast(str, creds.key_file),
                            look_for_keys=False,
                            port=creds.addr.port,
                            **banner_timeout_arg)
            elif creds.key is not None:
                with BytesIO(creds.key) as sio:
                    ssh.connect(creds.addr.host,
                                username=creds.user,
                                timeout=c_tcp_timeout,
                                pkey=paramiko.RSAKey.from_private_key(sio),
                                look_for_keys=False,
                                port=creds.addr.port,
                                **banner_timeout_arg)
            elif (creds.addr.host, creds.addr.port) in NODE_KEYS:
                ssh.connect(creds.addr.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            pkey=NODE_KEYS[creds.addr],
                            look_for_keys=False,
                            port=creds.addr.port,
                            **banner_timeout_arg)
            else:
                key_file = os.path.expanduser('~/.ssh/id_rsa')
                ssh.connect(creds.addr.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            key_filename=key_file,
                            look_for_keys=False,
                            port=creds.addr.port,
                            **banner_timeout_arg)
            return ssh
        except paramiko.PasswordRequiredException:
            raise
        except (socket.error, paramiko.SSHException):
            if time.time() > end_time:
                raise
            time.sleep(1)


def wait_ssh_available(addrs: List[IPAddr],
                       timeout: int = 300,
                       tcp_timeout: float = 1.0) -> None:

    addrs_set = set(addrs)  # type: Set[IPAddr]

    for _ in utils.Timeout(timeout):
        selector = selectors.DefaultSelector()  # type: selectors.BaseSelector
        with selector:
            for addr in addrs_set:
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
                # convert to greater or equal integer
                for key, _ in selector.select(timeout=int(ltime + 0.99999)):
                    selector.unregister(key.fileobj)
                    try:
                        key.fileobj.getpeername()  # type: ignore
                        addrs_set.remove(key.data)
                    except OSError as exc:
                        if exc.errno == errno.ENOTCONN:
                            pass
                ltime = etime - time.time()

        if not addrs_set:
            break


