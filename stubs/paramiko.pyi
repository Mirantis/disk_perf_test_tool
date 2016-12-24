from io import BytesIO
from typing import Any, Tuple


__version_info__ = None  # type: Tuple[int, int, int]


class PasswordRequiredException(Exception):
    pass


class SSHException(Exception):
    pass


class RSAKey:
    @classmethod
    def from_private_key(cls, data: BytesIO, password: str = None) -> 'RSAKey': ...

    @classmethod
    def from_private_key_file(cls, fname: str, password: str = None) -> 'RSAKey': ...



class AutoAddPolicy:
    pass


class SSHClient:
    def __init__(self) -> None:
        self.known_hosts = None  # type: Any

    def load_host_keys(self, path: str) -> None: ...
    def set_missing_host_key_policy(self, policy: AutoAddPolicy) -> None: ...
    def connect(self, *args: Any, **kwargs: Any): ...
    def get_transport(self) -> Any: ...
    def open_sftp(self) -> Any: ...
