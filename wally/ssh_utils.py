import re
import json
import time
import socket
import logging
import os.path
import getpass
from io import BytesIO
import subprocess
from typing import Union, Optional, cast, Dict, List, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor

import paramiko

import agent

from . import interfaces, utils


logger = logging.getLogger("wally")


class URIsNamespace(object):
    class ReParts(object):
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

    def __init__(self) -> None:
        self.user = None  # type: Optional[str]
        self.passwd = None  # type: Optional[str]
        self.host = None  # type: str
        self.port = 22  # type: int
        self.key_file = None  # type: Optional[str]

    def __str__(self) -> str:
        return str(self.__dict__)


SSHCredsType = Union[str, ConnCreds]


def parse_ssh_uri(uri: str) -> ConnCreds:
    # [ssh://]+
    # user:passwd@ip_host:port
    # user:passwd@ip_host
    # user@ip_host:port
    # user@ip_host
    # ip_host:port
    # ip_host
    # user@ip_host:port:path_to_key_file
    # user@ip_host::path_to_key_file
    # ip_host:port:path_to_key_file
    # ip_host::path_to_key_file

    if uri.startswith("ssh://"):
        uri = uri[len("ssh://"):]

    res = ConnCreds()
    res.port = 22
    res.key_file = None
    res.passwd = None
    res.user = getpass.getuser()

    for rr in URIsNamespace.uri_reg_exprs:
        rrm = re.match(rr, uri)
        if rrm is not None:
            res.__dict__.update(rrm.groupdict())
            return res

    raise ValueError("Can't parse {0!r} as ssh uri value".format(uri))


class LocalHost(interfaces.IHost):
    def __str__(self):
        return "<Local>"

    def get_ip(self) -> str:
        return 'localhost'

    def put_to_file(self, path: str, content: bytes) -> None:
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, "wb") as fd:
            fd.write(content)

    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        proc = subprocess.Popen(cmd, shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        stdout_data, _ = proc.communicate()
        if proc.returncode != 0:
            templ = "SSH:{0} Cmd {1!r} failed with code {2}. Output: {3}"
            raise OSError(templ.format(self, cmd, proc.returncode, stdout_data))

        return stdout_data


class SSHHost(interfaces.IHost):
    def __init__(self, ssh_conn, node_name: str, ip: str) -> None:
        self.conn = ssh_conn
        self.node_name = node_name
        self.ip = ip

    def get_ip(self) -> str:
        return self.ip

    def __str__(self) -> str:
        return self.node_name

    def put_to_file(self, path: str, content: bytes) -> None:
        with self.conn.open_sftp() as sftp:
            with sftp.open(path, "wb") as fd:
                fd.write(content)

    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        transport = self.conn.get_transport()
        session = transport.open_session()

        try:
            session.set_combine_stderr(True)

            stime = time.time()

            if not nolog:
                logger.debug("SSH:{0} Exec {1!r}".format(self, cmd))

            session.exec_command(cmd)
            session.settimeout(1)
            session.shutdown_write()
            output = ""

            while True:
                try:
                    ndata = session.recv(1024)
                    output += ndata
                    if "" == ndata:
                        break
                except socket.timeout:
                    pass

                if time.time() - stime > timeout:
                    raise OSError(output + "\nExecution timeout")

            code = session.recv_exit_status()
        finally:
            found = False

            if found:
                session.close()

        if code != 0:
            templ = "SSH:{0} Cmd {1!r} failed with code {2}. Output: {3}"
            raise OSError(templ.format(self, cmd, code, output))

        return output


NODE_KEYS = {}  # type: Dict[Tuple[str, int], paramiko.RSAKey]


def set_key_for_node(host_port: Tuple[str, int], key: bytes) -> None:
    sio = BytesIO(key)
    NODE_KEYS[host_port] = paramiko.RSAKey.from_private_key(sio)
    sio.close()


def ssh_connect(creds: SSHCredsType, conn_timeout: int = 60) -> interfaces.IHost:
    if creds == 'local':
        return LocalHost()

    tcp_timeout = 15
    default_banner_timeout = 30

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

            creds = cast(ConnCreds, creds)

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
            return SSHHost(ssh, "{0.host}:{0.port}".format(creds), creds.host)
        except paramiko.PasswordRequiredException:
            raise
        except (socket.error, paramiko.SSHException):
            if time.time() > end_time:
                raise
            time.sleep(1)


def connect(uri: str, **params) -> interfaces.IHost:
    if uri == 'local':
        res = LocalHost()
    else:
        creds = parse_ssh_uri(uri)
        creds.port = int(creds.port)
        res = ssh_connect(creds, **params)
    return res


SetupResult = Tuple[interfaces.IRPC, Dict[str, Any]]


RPCBeforeConnCallback = Callable[[interfaces.IHost, int], None]


def setup_rpc(node: interfaces.IHost,
              rpc_server_code: bytes,
              port: int = 0,
              rpc_conn_callback: RPCBeforeConnCallback = None) -> SetupResult:
    code_file = node.run("mktemp").strip()
    log_file = node.run("mktemp").strip()
    node.put_to_file(code_file, rpc_server_code)
    cmd = "python {code_file} server --listen-addr={listen_ip}:{port} --daemon " + \
          "--show-settings --stdout-file={out_file}"
    params_js = node.run(cmd.format(code_file=code_file,
                                    listen_addr=node.get_ip(),
                                    out_file=log_file,
                                    port=port)).strip()
    params = json.loads(params_js)
    params['log_file'] = log_file

    if rpc_conn_callback:
        ip, port = rpc_conn_callback(node, port)
    else:
        ip = node.get_ip()
        port = int(params['addr'].split(":")[1])

    return agent.connect((ip, port)), params


def wait_ssh_awailable(addrs: List[Tuple[str, int]],
                       timeout: int = 300,
                       tcp_timeout: float = 1.0,
                       max_threads: int = 32) -> None:
    addrs = addrs[:]
    tout = utils.Timeout(timeout)

    def check_sock(addr):
        s = socket.socket()
        s.settimeout(tcp_timeout)
        try:
            s.connect(addr)
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        while addrs:
            check_result = pool.map(check_sock, addrs)
            addrs = [addr for ok, addr in zip(check_result, addrs) if not ok]  # type: List[Tuple[str, int]]
            tout.tick()



