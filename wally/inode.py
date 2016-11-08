import abc
from typing import Set, Dict, Optional

from .ssh_utils import parse_ssh_uri
from . import hw_info
from .interfaces import IRemoteNode, IHost


class FuelNodeInfo:
    """FUEL master node additional info"""
    def __init__(self,
                 version: str,
                 fuel_ext_iface: str,
                 openrc: Dict[str, str]) -> None:

        self.version = version  # type: str
        self.fuel_ext_iface = fuel_ext_iface  # type: str
        self.openrc = openrc  # type: Dict[str, str]


class NodeInfo:
    """Node information object"""
    def __init__(self,
                 ssh_conn_url: str,
                 roles: Set[str],
                 bind_ip: str = None,
                 ssh_key: str = None) -> None:
        self.ssh_conn_url = ssh_conn_url  # type: str
        self.roles = roles  # type: Set[str]

        if bind_ip is None:
            bind_ip = parse_ssh_uri(self.ssh_conn_url).host

        self.bind_ip = bind_ip  # type: str
        self.ssh_key = ssh_key  # type: Optional[str]


class INode(IRemoteNode, metaclass=abc.ABCMeta):
    """Node object"""

    def __init__(self, node_info: NodeInfo):
        IRemoteNode.__init__(self)
        self.node_info = node_info  # type: NodeInfo
        self.hwinfo = None  # type: hw_info.HWInfo
        self.swinfo = None  # type: hw_info.SWInfo
        self.os_vm_id = None  # type: str
        self.ssh_conn = None  # type: IHost
        self.ssh_conn_url = None  # type: str
        self.rpc_conn = None
        self.rpc_conn_url = None  # type: str

    @abc.abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def node_id(self) -> str:
        pass
