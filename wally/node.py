import re
import getpass
from typing import Tuple
from .inode import INode, NodeInfo

from .ssh_utils import parse_ssh_uri, run_over_ssh, connect


class Node(INode):
    """Node object"""

    def __init__(self, node_info: NodeInfo) -> None:
        INode.__init__(self)

        self.info = node_info
        self.roles = node_info.roles
        self.bind_ip = node_info.bind_ip

        assert self.ssh_conn_url.startswith("ssh://")
        self.ssh_conn_url = node_info.ssh_conn_url

        self.ssh_conn = None
        self.rpc_conn_url = None
        self.rpc_conn = None
        self.os_vm_id = None
        self.hw_info = None

        if self.ssh_conn_url is not None:
            self.ssh_cred = parse_ssh_uri(self.ssh_conn_url)
            self.node_id = "{0.host}:{0.port}".format(self.ssh_cred)
        else:
            self.ssh_cred = None
            self.node_id = None

    def __str__(self) -> str:
        template = "<Node: url={conn_url!r} roles={roles}" + \
                   " connected={is_connected}>"
        return template.format(conn_url=self.ssh_conn_url,
                               roles=", ".join(self.roles),
                               is_connected=self.ssh_conn is not None)

    def __repr__(self) -> str:
        return str(self)

    def connect_ssh(self, timeout: int=None) -> None:
        self.ssh_conn = connect(self.ssh_conn_url)

    def connect_rpc(self) -> None:
        raise NotImplementedError()

    def prepare_rpc(self) -> None:
        raise NotImplementedError()

    def get_ip(self) -> str:
        """get node connection ip address"""

        if self.ssh_conn_url == 'local':
            return '127.0.0.1'
        return self.ssh_cred.host

    def get_user(self) -> str:
        """"get ssh connection username"""
        if self.ssh_conn_url == 'local':
            return getpass.getuser()
        return self.ssh_cred.user

    def run(self, cmd: str, stdin_data: str=None, timeout: int=60, nolog: bool=False) -> Tuple[int, str]:
        """Run command on node. Will use rpc connection, if available"""

        if self.rpc_conn is None:
            return run_over_ssh(self.ssh_conn, cmd,
                                stdin_data=stdin_data, timeout=timeout,
                                nolog=nolog, node=self)
        assert not stdin_data
        proc_id = self.rpc_conn.cli.spawn(cmd)
        exit_code = None
        output = ""

        while exit_code is None:
            exit_code, stdout_data, stderr_data = self.rpc_conn.cli.get_updates(proc_id)
            output += stdout_data + stderr_data

        return exit_code, output

    def discover_hardware_info(self) -> None:
        raise NotImplementedError()

    def get_file_content(self, path: str) -> str:
        raise NotImplementedError()

    def forward_port(self, ip: str, remote_port: int, local_port: int = None) -> int:
        raise NotImplementedError()

    def get_interface(self, ip: str) -> str:
        """Get node external interface for given IP"""
        data = self.run("ip a", nolog=True)
        curr_iface = None

        for line in data.split("\n"):
            match1 = re.match(r"\d+:\s+(?P<name>.*?):\s\<", line)
            if match1 is not None:
                curr_iface = match1.group('name')

            match2 = re.match(r"\s+inet\s+(?P<ip>[0-9.]+)/", line)
            if match2 is not None:
                if match2.group('ip') == ip:
                    assert curr_iface is not None
                    return curr_iface

        raise KeyError("Can't found interface for ip {0}".format(ip))

    def sync_hw_info(self) -> None:
        pass

    def sync_sw_info(self) -> None:
        pass