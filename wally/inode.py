import abc
from typing import Set, Dict, Tuple, Any, Optional

from .ssh_utils import parse_ssh_uri


class FuelNodeInfo:
    """FUEL master node additional info"""
    def __init__(self,
                 version: str,
                 fuel_ext_iface: str,
                 openrc: Dict[str, str]):

        self.version = version
        self.fuel_ext_iface = fuel_ext_iface
        self.openrc = openrc


class NodeInfo:
    """Node information object"""
    def __init__(self,
                 ssh_conn_url: str,
                 roles: Set[str],
                 bind_ip: str=None,
                 ssh_key: str=None):
        self.ssh_conn_url = ssh_conn_url
        self.roles = roles

        if bind_ip is None:
            bind_ip = parse_ssh_uri(self.ssh_conn_url).host

        self.bind_ip = bind_ip
        self.ssh_key = ssh_key


class INode(metaclass=abc.ABCMeta):
    """Node object"""

    def __init__(self, node_info: NodeInfo):
        self.rpc = None
        self.node_info = node_info
        self.hwinfo = None
        self.roles = []

    @abc.abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    @abc.abstractmethod
    def connect_ssh(self, timeout: int=None) -> None:
        pass

    @abc.abstractmethod
    def connect_rpc(self) -> None:
        pass

    @abc.abstractmethod
    def prepare_rpc(self) -> None:
        pass

    @abc.abstractmethod
    def get_ip(self) -> str:
        pass

    @abc.abstractmethod
    def get_user(self) -> str:
        pass

    @abc.abstractmethod
    def run(self, cmd: str, stdin_data: str=None, timeout: int=60, nolog: bool=False) -> str:
        pass

    @abc.abstractmethod
    def discover_hardware_info(self) -> None:
        pass

    @abc.abstractmethod
    def copy_file(self, local_path: str, remote_path: Optional[str]=None) -> str:
        pass

    @abc.abstractmethod
    def get_file_content(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def put_to_file(self, path:str, content: bytes) -> None:
        pass

    @abc.abstractmethod
    def forward_port(self, ip: str, remote_port: int, local_port: int=None) -> int:
        pass

    @abc.abstractmethod
    def get_interface(self, ip: str) -> str:
        pass

    @abc.abstractmethod
    def stat_file(self, path:str) -> Any:
        pass
