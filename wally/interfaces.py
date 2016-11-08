import abc
from typing import Any, Set, Dict


class IRemoteShell(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        pass


class IHost(IRemoteShell, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_ip(self) -> str:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def put_to_file(self, path: str, content: bytes) -> None:
        pass


class IRemoteFS(metaclass=abc.ABCMeta):
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


class IRPC(metaclass=abc.ABCMeta):
    pass


class IRemoteNode(IRemoteFS, IRemoteShell, metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        self.roles = set()  # type: Set[str]
        self.rpc = None  # type: IRPC
        self.rpc_params = None  # type: Dict[str, Any]

    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    @abc.abstractmethod
    def connect_ssh(self, timeout: int = None) -> None:
        pass

    @abc.abstractmethod
    def get_ip(self) -> str:
        pass

    @abc.abstractmethod
    def get_user(self) -> str:
        pass

