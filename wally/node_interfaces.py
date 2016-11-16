import abc
from typing import Any, Set, Optional, List, Dict, Callable, NamedTuple
from .ssh_utils import ConnCreds
from .common_types import IPAddr


RPCCreds = NamedTuple("RPCCreds", [("addr", IPAddr), ("key_file", str), ("cert_file", str)])


class NodeInfo:
    """Node information object, result of discovery process or config parsing"""
    def __init__(self, ssh_creds: ConnCreds, roles: Set[str]) -> None:

        # ssh credentials
        self.ssh_creds = ssh_creds
        # credentials for RPC connection
        self.rpc_creds = None  # type: Optional[RPCCreds]
        self.roles = roles
        self.os_vm_id = None  # type: Optional[int]
        self.params = {}  # type: Dict[str, Any]

    def node_id(self) -> str:
        return "{0.host}:{0.port}".format(self.ssh_creds.addr)


class ISSHHost(metaclass=abc.ABCMeta):
    """Minimal interface, required to setup RPC connection"""
    info = None  # type: NodeInfo

    @abc.abstractmethod
    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    @abc.abstractmethod
    def put_to_file(self, path: str, content: bytes) -> None:
        pass

    def __enter__(self) -> 'ISSHHost':
        return self

    def __exit__(self, x, y, z) -> bool:
        self.disconnect()
        return False


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
    def get_interface(self, ip: str) -> str:
        pass

    @abc.abstractmethod
    def stat_file(self, path:str) -> Any:
        pass

    @abc.abstractmethod
    def disconnect(self) -> str:
        pass

    def __enter__(self) -> 'IRPCNode':
        return self

    def __exit__(self, x, y, z) -> bool:
        self.disconnect()
        return False

