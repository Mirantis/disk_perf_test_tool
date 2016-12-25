import abc
from typing import Any, Set, Dict, NamedTuple, Optional
from .ssh_utils import ConnCreds
from .common_types import IPAddr
from .result_classes import IStorable


RPCCreds = NamedTuple("RPCCreds", [("addr", IPAddr), ("key_file", str), ("cert_file", str)])


class NodeInfo(IStorable):
    """Node information object, result of discovery process or config parsing"""

    yaml_tag = 'node_info'

    def __init__(self, ssh_creds: ConnCreds, roles: Set[str], params: Dict[str, Any] = None) -> None:
        # ssh credentials
        self.ssh_creds = ssh_creds

        # credentials for RPC connection
        self.rpc_creds = None  # type: Optional[RPCCreds]

        self.roles = roles
        self.os_vm_id = None  # type: Optional[int]
        self.params = {}  # type: Dict[str, Any]
        if params is not None:
            self.params = params

    def node_id(self) -> str:
        return "{0.host}:{0.port}".format(self.ssh_creds.addr)

    def __str__(self) -> str:
        return self.node_id()

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        dct = self.__dict__.copy()

        if self.rpc_creds is not None:
            dct['rpc_creds'] = list(self.rpc_creds)

        dct['ssh_creds'] = self.ssh_creds.raw()
        dct['roles'] = list(self.roles)
        return dct

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'NodeInfo':
        data = data.copy()
        if data['rpc_creds'] is not None:
            data['rpc_creds'] = RPCCreds(*data['rpc_creds'])

        data['ssh_creds'] = ConnCreds.fromraw(data['ssh_creds'])
        data['roles'] = set(data['roles'])
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj


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
    def put_to_file(self, path: Optional[str], content: bytes) -> str:
        pass

    def __enter__(self) -> 'ISSHHost':
        return self

    def __exit__(self, x, y, z) -> bool:
        self.disconnect()
        return False


class IRPCNode(metaclass=abc.ABCMeta):
    """Remote filesystem interface"""
    info = None  # type: NodeInfo
    conn = None  # type: Any
    rpc_log_file = None  # type: str

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def run(self, cmd: str, timeout: int = 60, nolog: bool = False, check_timeout: float = 0.01) -> str:
        pass

    @abc.abstractmethod
    def copy_file(self, local_path: str, remote_path: str = None) -> str:
        pass

    @abc.abstractmethod
    def get_file_content(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def put_to_file(self, path: Optional[str], content: bytes) -> str:
        pass

    @abc.abstractmethod
    def stat_file(self, path:str) -> Any:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    @abc.abstractmethod
    def upload_plugin(self, name: str, code: bytes, version: str = None) -> None:
        pass

    def __enter__(self) -> 'IRPCNode':
        return self

    def __exit__(self, x, y, z) -> bool:
        self.disconnect()
        return False

