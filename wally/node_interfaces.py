import abc
from typing import Any, Set, Optional, List, Dict, Callable


class NodeInfo:
    """Node information object, result of dicovery process or config parsing"""

    def __init__(self,
                 ssh_conn_url: str,
                 roles: Set[str],
                 hops: List['NodeInfo'] = None,
                 ssh_key: bytes = None) -> None:

        self.hops = []  # type: List[NodeInfo]
        if hops is not None:
            self.hops = hops

        self.ssh_conn_url = ssh_conn_url  # type: str
        self.rpc_conn_url = None  # type: str
        self.roles = roles  # type: Set[str]
        self.os_vm_id = None  # type: Optional[int]
        self.ssh_key = ssh_key  # type: Optional[bytes]
        self.params = {}  # type: Dict[str, Any]


class ISSHHost(metaclass=abc.ABCMeta):
    """Minimal interface, required to setup RPC connection"""
    info = None  # type: NodeInfo

    @abc.abstractmethod
    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        pass

    @abc.abstractmethod
    def get_ip(self) -> str:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def put_to_file(self, path: str, content: bytes) -> None:
        pass


class IRPCNode(metaclass=abc.ABCMeta):
    """Remote filesystem interface"""
    info = None  # type: NodeInfo

    @abc.abstractmethod
    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        pass

    @abc.abstractmethod
    def copy_file(self, local_path: str, remote_path: str = None) -> str:
        pass

    @abc.abstractmethod
    def get_file_content(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def put_to_file(self, path:str, content: bytes) -> None:
        pass

    @abc.abstractmethod
    def forward_port(self, ip: str, remote_port: int, local_port: int = None) -> int:
        pass

    @abc.abstractmethod
    def get_interface(self, ip: str) -> str:
        pass

    @abc.abstractmethod
    def stat_file(self, path:str) -> Any:
        pass

    @abc.abstractmethod
    def node_id(self) -> str:
        pass


    @abc.abstractmethod
    def disconnect(self) -> str:
        pass



RPCBeforeConnCallback = Callable[[NodeInfo, int], None]